#!/usr/bin/env python3
"""Autonomous SAM/RGB-D lid grasp with short-horizon visual replanning.

No home or automatic return command exists in this entry point.  Production
motion is impossible until an accepted camera->robot calibration and a static
ESDF are supplied.  ``--dry-run`` performs the same live perception and
planning preflight without sending arm or gripper commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.camera_id import configure_camera_map_by_udid
from rollout.autonomous_mpc import (
    AtomicRunState,
    AutonomousStop,
    ChunkExecutor,
    CompositePoseValidator,
    ESDFGrid,
    MuJoCoIKValidator,
    Pose,
    ReplanPolicy,
    SceneSnapshot,
    check_pregrasp,
    decide_replan,
    minimum_jerk_segment,
    plan_lift_translate_descend,
    plan_to_dict,
    validate_calibration,
)
from rollout.sam_segmentation import detect_blue_cross_center
from rollout.scene_semantics import LABEL_BACKGROUND, LABEL_LID, LABEL_ROBOT
from rollout.scene_volume import automatic_grid, integrate_projective_depth


def _json(path):
    return json.loads(Path(path).read_text())


def _fingerprint(paths):
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _camera_to_robot(point_camera, transform):
    homogeneous = np.asarray((*point_camera, 1.0), dtype=float)
    return (np.asarray(transform, dtype=float) @ homogeneous)[:3]


def _instance_id(artifacts):
    if not artifacts:
        return None
    mask_path = artifacts.get("lid_mask")
    if mask_path and Path(mask_path).exists():
        return hashlib.sha256(Path(mask_path).read_bytes()).hexdigest()[:16]
    lid = artifacts.get("lid")
    return None if lid is None else str(lid.get("prompt"))


def _runner_args(config, output_dir):
    return SimpleNamespace(
        torque_config=config["torque_config"],
        scene_config=config["scene_config"],
        sam_endpoint=config["sam_endpoint"],
        output_dir=str(Path(output_dir) / "head"),
        holding_kp=[7.0] * 6,
        holding_kd=[0.3] * 6,
        motion_kp=[7.0] * 6,
        motion_kd=[0.3] * 6,
        gain_ramp_s=1.0,
        mode_settle_s=0.5,
        hold_settle_s=0.25,
        right_can_interface="can_right",
    )


class LivePerception:
    def __init__(self, runner, right_observer, transform, output_dir):
        self.runner = runner
        self.right = right_observer
        self.transform = transform
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.esdf = None
        self.esdf_sequence = 0

    def _mask(self, path, shape):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise AutonomousStop(f"could not read SAM mask: {path}")
        if mask.shape != shape:
            mask = cv2.resize(
                mask,
                (shape[1], shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask > 0

    def _rebuild_esdf(self):
        """Build a fresh camera-frame ESDF with robot/lid dynamic labels."""

        artifacts = self.runner.last_observation_artifacts or {}
        raw_path = artifacts.get("raw_image")
        lid_path = artifacts.get("lid_mask")
        gripper_path = artifacts.get("gripper_mask")
        if not raw_path or not lid_path or not gripper_path:
            raise AutonomousStop("head SAM artifacts are incomplete")
        image = cv2.imread(str(raw_path))
        depth = np.asarray(self.runner.last_depth, dtype=float)
        matrix = np.asarray(self.runner.last_camera_matrix, dtype=float)
        if image is None or depth.ndim != 2 or matrix.shape != (3, 3):
            raise AutonomousStop("fresh RGB-D observation is incomplete")
        if image.shape[:2] != depth.shape:
            image = cv2.resize(
                image,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        labels = np.full(depth.shape, LABEL_BACKGROUND, dtype=np.uint8)
        labels[self._mask(lid_path, depth.shape)] = LABEL_LID
        gripper_mask = self._mask(gripper_path, depth.shape)
        labels[gripper_mask] = LABEL_ROBOT

        # The gripper-specific mask is precise near the target.  A second SAM
        # prompt removes the rest of both arms from the static collision layer.
        result = self.runner.sam.segment(
            image,
            frame_id=self.runner.frame_id,
            timestamp=float(self.runner.last_head_timestamp),
            prompt="robot arms and grippers",
            confidence_threshold=0.05,
        )
        self.runner.frame_id += 1
        for candidate in result.candidates:
            mask = np.asarray(candidate.mask, dtype=np.uint8)
            if mask.shape != depth.shape:
                mask = cv2.resize(
                    mask,
                    (depth.shape[1], depth.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            robot_mask = mask > 0
            if np.count_nonzero(robot_mask & gripper_mask) >= 20:
                labels[robot_mask] = LABEL_ROBOT

        grid = automatic_grid(
            depth,
            matrix,
            voxel_size_m=0.010,
            truncation_m=0.030,
        )
        volume = integrate_projective_depth(
            depth,
            matrix,
            grid,
            truncation_m=0.030,
            surface_labels=labels,
        )
        static_esdf = volume.esdf_m.copy()
        dynamic = (
            (volume.semantic_labels == LABEL_ROBOT)
            | (volume.semantic_labels == LABEL_LID)
        ) & volume.observed
        static_occupied = (
            (volume.semantic_labels == LABEL_BACKGROUND)
            & volume.observed
            & (volume.tsdf <= 0.0)
        )
        if np.any(static_occupied):
            distance = distance_transform_edt(
                ~static_occupied, sampling=grid.voxel_size_m
            )
            static_esdf[dynamic] = distance[dynamic]
        self.esdf = ESDFGrid(
            static_esdf,
            grid.origin_xyz_m,
            grid.voxel_size_m,
            T_esdf_robot=np.linalg.inv(self.transform),
            body_radius_m=0.025,
        )
        path = self.output_dir / f"live_esdf_{self.esdf_sequence:04d}.npz"
        self.esdf_sequence += 1
        np.savez_compressed(
            path,
            esdf_m=static_esdf,
            observed=volume.observed,
            semantic_labels=volume.semantic_labels,
            origin_xyz_m=grid.origin_xyz_m,
            voxel_size_m=grid.voxel_size_m,
            T_camera_robot=np.linalg.inv(self.transform),
        )
        return str(path)

    def head_snapshot(self):
        _, _, image, timestamp = self.runner.observe(0.0)
        if not self.runner.last_geometry_quality.accepted:
            raise AutonomousStop(
                "RGB-D target geometry failed the live quality gate"
            )
        esdf_artifact = self._rebuild_esdf()
        target_camera = self.runner.last_target_3d.point_camera_xyz_m
        target_robot = _camera_to_robot(target_camera, self.transform)
        pose = Pose.from_se3(self.runner.rpc.get_right_ee_pose())
        clearance = self.esdf.clearance(pose)
        snapshot = SceneSnapshot(
            timestamp_s=float(timestamp),
            target_xyz_m=tuple(float(value) for value in target_robot),
            target_instance_id=_instance_id(
                self.runner.last_observation_artifacts
            ),
            ee_pose=pose,
            lid_visible=True,
            blue_marker_visible=True,
            estimated_gripper_lid_m=float(
                np.linalg.norm(np.asarray(pose.xyz) - target_robot)
            ),
            predicted_clearance_m=clearance,
        )
        return snapshot, {"sam_overlay": image, "esdf_npz": esdf_artifact}

    def right_snapshot(self, head):
        geometry, _, image, timestamp = self.right.observe(require_lid=False)
        artifacts = self.right.last_observation_artifacts or {}
        raw = artifacts.get("raw_image")
        marker = None
        if raw and Path(raw).exists():
            marker_value = detect_blue_cross_center(cv2.imread(str(raw)))
            if marker_value is not None:
                marker = tuple(float(value) for value in marker_value)
        return SceneSnapshot(
            timestamp_s=float(timestamp),
            target_xyz_m=head.target_xyz_m,
            target_instance_id=head.target_instance_id,
            ee_pose=Pose.from_se3(self.runner.rpc.get_right_ee_pose()),
            lid_visible=geometry is not None,
            blue_marker_visible=marker is not None,
            right_marker_px=marker,
            estimated_gripper_lid_m=head.estimated_gripper_lid_m,
            predicted_clearance_m=head.predicted_clearance_m,
        ), image


def _offline(config, args, state, calibration):
    scene = _json(args.offline_scene)
    camera_udid = config["camera_udids"]["head"]
    transform = validate_calibration(calibration, camera_udid=camera_udid)
    target = _camera_to_robot(scene["target_camera_xyz_m"], transform)
    start = Pose(
        tuple(scene["ee_pose_wxyz_xyz"][:4]),
        tuple(scene["ee_pose_wxyz_xyz"][4:]),
    )
    clearance = float(scene.get("constant_clearance_m", 0.050))
    plan = plan_lift_translate_descend(
        start,
        target,
        lift_m=float(config["planner"]["lift_m"]),
        pregrasp_height_m=float(config["planner"]["pregrasp_height_m"]),
        final_height_m=float(config["planner"]["final_height_m"]),
        validator=lambda pose: clearance,
    )
    state.update("DRY_RUN_COMPLETE", plan=plan_to_dict(plan), offline=True)
    print(json.dumps(state.payload, indent=2))
    return 0


def _reset_calibration(path):
    payload = _json(path)
    payload.update(
        accepted=False,
        accepted_at_s=None,
        T_robot_camera=None,
        T_esdf_robot=None,
    )
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def run_live(config, args, state, calibration):
    from src.run_realtime_sam_grasp import LiveSamGrasp
    from src.run_staged_sam_pregrasp import RightLidObserver

    expected = config["camera_udids"]
    mapping, live_udids = configure_camera_map_by_udid(expected)
    state.event("camera_udid_preflight", mapping=mapping, connected=live_udids)
    transform = validate_calibration(
        calibration, camera_udid=expected["head"]
    )
    torque = _json(config["torque_config"])
    runner = right = None
    try:
        runner = LiveSamGrasp(_runner_args(config, args.output_dir))
        right = RightLidObserver(runner, Path(args.output_dir) / "right")
        runner.start()
        right.start()
        perception = LivePerception(
            runner,
            right,
            transform,
            Path(args.output_dir) / "scene",
        )
        reference, image = perception.head_snapshot()  # real-image SAM preflight
        state.event(
            "sam_preflight",
            image=image,
            target_xyz_m=reference.target_xyz_m,
            instance=reference.target_instance_id,
        )
        def make_plan(snapshot):
            simulation = MuJoCoIKValidator(
                config["mujoco_model"],
                runner.rpc.get_right_joint_positions(),
                left_q=runner.rpc.get_left_joint_positions(),
            )
            validator = CompositePoseValidator(perception.esdf, simulation)
            candidate = plan_lift_translate_descend(
                snapshot.ee_pose,
                snapshot.target_xyz_m,
                lift_m=float(config["planner"]["lift_m"]),
                pregrasp_height_m=float(config["planner"]["pregrasp_height_m"]),
                final_height_m=float(config["planner"]["final_height_m"]),
                validator=validator,
            )
            candidate.metadata["mujoco_q_waypoints"] = simulation.q_waypoints
            candidate.metadata["mujoco_model"] = config["mujoco_model"]
            return candidate

        plan = make_plan(reference)
        minimum = float(config["planner"]["minimum_clearance_m"])
        if plan.minimum_clearance_m is None or plan.minimum_clearance_m < minimum:
            raise AutonomousStop("planned static ESDF clearance is too low")
        state.update("PLANNED", plan=plan_to_dict(plan))
        if args.dry_run:
            state.update("DRY_RUN_COMPLETE")
            print(json.dumps(state.payload, indent=2))
            return 0

        runner.rpc.open_right_gripper()
        executor = ChunkExecutor(
            runner.rpc,
            torque_limit_nm=torque["thresholds"]["right"],
            consecutive_torque_samples=torque["consecutive_samples"],
        )
        policy = ReplanPolicy(
            maximum_observation_age_s=float(
                config["replan"]["maximum_observation_age_s"]
            ),
            maximum_target_shift_m=float(
                config["replan"]["maximum_target_shift_m"]
            ),
            maximum_position_error_m=float(
                config["replan"]["maximum_position_error_m"]
            ),
            maximum_rotation_error_deg=float(
                config["replan"]["maximum_rotation_error_deg"]
            ),
            minimum_clearance_m=minimum,
        )
        maximum_replans = int(config["replan"]["maximum_replans"])
        replans = 0
        while True:
            replan_now = False
            for index, chunk in enumerate(plan.chunks()):
                commanded = executor.execute(chunk)
                current, image = perception.head_snapshot()
                decision = decide_replan(
                    now_s=time.time(),
                    reference=reference,
                    current=current,
                    commanded_pose=commanded,
                    policy=policy,
                )
                state.event(
                    "chunk_observation",
                    chunk=index,
                    action=decision.action,
                    reasons=decision.reasons,
                    image=image,
                    actual_pose=current.ee_pose.__dict__,
                )
                if decision.action == "HOLD":
                    executor.sender.hold()
                    raise AutonomousStop("fresh observation unavailable")
                if decision.action == "REPLAN":
                    reference = current
                    replan_now = True
                    break
            if not replan_now:
                reference = current
                break
            replans += 1
            if replans > maximum_replans:
                executor.sender.hold()
                raise AutonomousStop("maximum visual replans exceeded")
            plan = make_plan(reference)
            state.event("replanned", count=replans, plan=plan_to_dict(plan))

        grasp = config["grasp"]
        right_snapshot, right_image = perception.right_snapshot(reference)
        gate = check_pregrasp(
            right_snapshot,
            goal_px=grasp["right_goal_px"],
            tolerance_px=float(grasp["right_tolerance_px"]),
            maximum_gripper_lid_m=float(grasp["maximum_gripper_lid_m"]),
        )
        state.event(
            "pregrasp_gate",
            allowed=gate.allowed,
            reasons=gate.reasons,
            pixel_error=gate.pixel_error,
            image=right_image,
        )
        if not gate.allowed:
            executor.sender.hold()
            raise AutonomousStop(
                f"right-camera pregrasp gate rejected: {gate.reasons}"
            )

        before_target = np.asarray(reference.target_xyz_m)
        runner.rpc.close_right_gripper()
        time.sleep(0.6)
        ratio = float(runner.rpc.get_right_gripper_exact())
        if ratio <= float(grasp["empty_close_ratio"]):
            raise AutonomousStop("gripper fully closed; lid was not captured")
        lift_finish = Pose(
            reference.ee_pose.wxyz,
            (
                reference.ee_pose.xyz[0],
                reference.ee_pose.xyz[1],
                reference.ee_pose.xyz[2]
                + float(grasp["verification_lift_m"]),
            ),
        )
        lift = minimum_jerk_segment(
            reference.ee_pose,
            lift_finish,
            0.75,
            stage="verification_lift",
        )
        executor.execute(lift)
        after, _ = perception.head_snapshot()
        after_right, image = perception.right_snapshot(after)
        followed = float(np.asarray(after.target_xyz_m)[2] - before_target[2])
        success = (
            ratio > float(grasp["empty_close_ratio"])
            and after_right.blue_marker_visible
            and followed >= float(grasp["minimum_lid_follow_m"])
        )
        state.update(
            "SUCCEEDED" if success else "FAILED_VERIFICATION",
            result={
                "gripper_ratio": ratio,
                "blue_marker_visible": after_right.blue_marker_visible,
                "lid_follow_m": followed,
                "right_image": image,
            },
        )
        if not success:
            raise AutonomousStop("post-lift grasp verification failed")
        return 0
    except BaseException as error:
        state.update(
            "ABORTED",
            error={"type": type(error).__name__, "message": str(error)},
        )
        if runner is not None and not args.dry_run:
            try:
                from rollout.autonomous_mpc import AbsoluteCartesianTargetSender

                AbsoluteCartesianTargetSender(runner.rpc).hold()
            except BaseException:
                pass
        raise
    finally:
        if right is not None:
            right.stop()
        if runner is not None:
            runner.stop()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/configs/pasteur_autonomous_lid_grasp.json",
    )
    parser.add_argument("--output-dir", default="/tmp/autonomous_sam_lid_grasp")
    parser.add_argument("--state", help="defaults to OUTPUT_DIR/run_state.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline-scene")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-calibration", action="store_true")
    args = parser.parse_args(argv)

    config = _json(args.config)
    calibration_path = config["calibration"]
    if args.reset_calibration:
        _reset_calibration(calibration_path)
        print(f"reset calibration: {calibration_path}")
        return 0
    calibration = _json(calibration_path)
    state_path = args.state or str(Path(args.output_dir) / "run_state.json")
    state = AtomicRunState(state_path, resume=args.resume)
    state.update(
        "PREFLIGHT",
        dry_run=bool(args.dry_run or args.offline_scene),
        config_fingerprint=_fingerprint(
            [
                args.config,
                config["scene_config"],
                config["task_config"],
                config["torque_config"],
                calibration_path,
            ]
        ),
    )
    if args.offline_scene:
        return _offline(config, args, state, calibration)
    return run_live(config, args, state, calibration)


if __name__ == "__main__":
    raise SystemExit(main())
