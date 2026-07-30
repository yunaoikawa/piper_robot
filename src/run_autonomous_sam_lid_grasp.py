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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.camera_id import configure_camera_map_by_udid
from rollout.autonomous_mpc import (
    AnalyticObstacleSet,
    AtomicRunState,
    AutonomousStop,
    ChunkExecutor,
    CompositePoseValidator,
    ESDFGrid,
    MuJoCoIKValidator,
    Pose,
    ReplanPolicy,
    SceneSnapshot,
    decide_replan,
    minimum_jerk_segment,
    plan_lift_translate_descend,
    plan_to_dict,
    quaternion_angle_deg,
    select_collision_aware_plan,
    validate_calibration,
)
from rollout.torque_safety import torque_stop_enabled_from_config
from rollout.autonomous_grasp_state_machine import (
    GraspGates,
    GraspState,
    decide_state,
)
from rollout.descent_probe import assess_descent_probe, assess_lowest_point
from rollout.daily_scene import (
    DailySceneStore,
    SceneNotConfirmed,
    SceneObject,
)
from rollout.grasp_contact_verification import assess_stable_closure
from rollout.grasp_readiness import PrecloseThresholds, assess_preclose
from rollout.grasp_window import (
    GraspWindowAssessment,
    GraspWindowTemplate,
    assess_grasp_window,
    normalized_pad_target_gap,
    render_grasp_window,
)
from rollout.scene_semantics import LABEL_BACKGROUND, LABEL_LID, LABEL_ROBOT
from rollout.scene_volume import automatic_grid, integrate_projective_depth
from rollout.tool_plane_geometry import MuJoCoToolPlane, transform_plane


@dataclass(frozen=True)
class RightGraspObservation:
    """Generic target-in-tool observation from the wrist camera."""

    scene: SceneSnapshot
    assessment: GraspWindowAssessment | None
    normalized_pad_target_gap: float | None
    image: str
    target_mask: str | None

    @property
    def target_visible(self) -> bool:
        return self.assessment is not None


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
    target = artifacts.get("target") or artifacts.get("lid")
    if target is not None and target.get("prompt"):
        # A segmentation mask naturally changes at every frame. Hashing its
        # pixels made one physical target look like a new instance and caused
        # needless replanning. The semantic prompt is stable for this local,
        # single-target task.
        return str(target["prompt"])
    mask_path = artifacts.get("target_mask") or artifacts.get("lid_mask")
    if mask_path and Path(mask_path).exists():
        return hashlib.sha256(Path(mask_path).read_bytes()).hexdigest()[:16]
    return None


def _calibration_id(calibration):
    return hashlib.sha256(
        json.dumps(calibration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]


def _daily_scene_object_proposals(
    config, task, reference, image, inventory=None
):
    inventory = inventory or {}
    catalog = config.get("daily_scene", {}).get("expected_objects", [])
    objects = []
    for index, expected in enumerate(catalog):
        role = expected.get("role")
        is_target = role == "target_lid"
        shares_target_pose = role in {"target_lid", "target_container"}
        pose = None
        confidence = 0.0
        mask_path = None
        if shares_target_pose and reference.target_xyz_m is not None:
            pose = np.eye(4)
            pose[:3, 3] = np.asarray(reference.target_xyz_m, dtype=float)
            confidence = 1.0
            mask_path = image.get("sam_overlay")
        detected = inventory.get(str(expected.get("instance_id", "")))
        if detected is not None:
            confidence = max(confidence, float(detected["confidence"]))
            mask_path = detected.get("overlay", mask_path)
        objects.append(
            SceneObject(
                instance_id=str(
                    expected.get("instance_id", f"expected-{index}")
                ),
                semantic_name=str(expected["semantic_name"]),
                geometry=dict(expected["geometry"]),
                role=role,
                pose_robot=None if pose is None else pose.tolist(),
                confidence=confidence,
                status="uncertain",
                source=(
                    "sam_rgbd"
                    if is_target
                    else (
                        "sam_rgbd_inventory"
                        if detected is not None
                        else (
                            "target_assembly_proxy"
                            if shares_target_pose
                            else "expected_daily_inventory"
                        )
                    )
                ),
                transparent=bool(expected.get("transparent", False)),
                mask_path=mask_path,
                depth_quality=(
                    "support_plane_proxy"
                    if shares_target_pose
                    else (
                        "rgb_only_instance"
                        if detected is not None
                        else "not_observed"
                    )
                ),
            )
        )
    if not objects:
        target = task["target"]
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(reference.target_xyz_m, dtype=float)
        objects.append(
            SceneObject(
                instance_id=reference.target_instance_id or "target-1",
                semantic_name=target["semantic_name"],
                geometry={"type": target["shape_class"]},
                role="target_lid",
                pose_robot=pose.tolist(),
                confidence=1.0,
                status="uncertain",
                transparent=bool(target.get("transparent", False)),
                mask_path=image.get("sam_overlay"),
                depth_quality="support_plane_proxy",
            )
        )
    return objects


def _runner_args(config, output_dir):
    return SimpleNamespace(
        torque_config=config["torque_config"],
        scene_config=config["scene_config"],
        sam_endpoint=config["sam_endpoint"],
        output_dir=str(Path(output_dir) / "head"),
        holding_kp=[7.0, 7.0, 7.0, 5.0, 5.0, 5.0],
        holding_kd=[0.3] * 6,
        motion_kp=[7.0, 7.0, 7.0, 5.0, 5.0, 5.0],
        motion_kd=[0.3] * 6,
        gain_ramp_s=1.0,
        mode_settle_s=0.5,
        hold_settle_s=0.25,
        right_can_interface="can_right",
        depth_frames=15,
    )


class LivePerception:
    def __init__(
        self,
        runner,
        right_observer,
        transform,
        output_dir,
        grasp_window_template,
        grasp_window_method,
    ):
        self.runner = runner
        self.right = right_observer
        self.transform = transform
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grasp_window_template = grasp_window_template
        self.grasp_window_method = grasp_window_method
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

    def scan_inventory(self, expected_objects):
        """Run closed-set SAM prompts for the operator's daily inventory."""

        artifacts = self.runner.last_observation_artifacts or {}
        raw_path = artifacts.get("raw_image")
        image = cv2.imread(str(raw_path)) if raw_path else None
        if image is None:
            raise AutonomousStop("daily inventory RGB image is unavailable")
        overlay = image.copy()
        detections = {}
        for expected in expected_objects:
            if expected.get("role") == "target_lid":
                continue
            prompt = expected.get("sam_prompt") or expected["semantic_name"]
            result = self.runner.sam.segment(
                image,
                frame_id=self.runner.frame_id,
                timestamp=float(self.runner.last_head_timestamp),
                prompt=str(prompt),
                confidence_threshold=0.05,
            )
            self.runner.frame_id += 1
            if not result.candidates:
                continue
            candidate = max(result.candidates, key=lambda item: item.score)
            mask = np.asarray(candidate.mask, dtype=np.uint8)
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = mask > 0
            if np.count_nonzero(mask) < 20:
                continue
            tint = np.zeros_like(overlay)
            tint[mask] = (40, 180, 255)
            overlay = cv2.addWeighted(overlay, 1.0, tint, 0.35, 0)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
            x, y, _, _ = cv2.boundingRect(mask.astype(np.uint8))
            cv2.putText(
                overlay,
                str(expected["semantic_name"]),
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            detections[str(expected["instance_id"])] = {
                "confidence": float(candidate.score),
            }
        path = self.output_dir / "daily_inventory_sam.png"
        if not cv2.imwrite(str(path), overlay):
            raise AutonomousStop("could not save daily inventory overlay")
        for detection in detections.values():
            detection["overlay"] = str(path)
        return detections, str(path)

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
            blue_marker_visible=False,
            estimated_gripper_lid_m=float(
                np.linalg.norm(np.asarray(pose.xyz) - target_robot)
            ),
            predicted_clearance_m=clearance,
        )
        return snapshot, {"sam_overlay": image, "esdf_npz": esdf_artifact}

    def right_snapshot(self, head):
        geometry, _, image, timestamp = self.right.observe(require_lid=False)
        artifacts = self.right.last_observation_artifacts or {}
        raw_path = artifacts.get("raw_image")
        target_mask_path = artifacts.get("target_mask") or artifacts.get(
            "lid_mask"
        )
        assessment = None
        normalized_gap = None
        if geometry is not None and raw_path and target_mask_path:
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            target_mask = cv2.imread(
                str(target_mask_path), cv2.IMREAD_GRAYSCALE
            )
            if raw is None or target_mask is None:
                raise AutonomousStop(
                    "right target image or SAM mask could not be read"
                )
            target_mask = target_mask > 0
            assessment, tool_frame = assess_grasp_window(
                raw,
                target_mask,
                self.grasp_window_template,
                method=self.grasp_window_method,
            )
            normalized_gap, _ = normalized_pad_target_gap(raw, target_mask)
            overlay = render_grasp_window(
                raw,
                target_mask,
                self.grasp_window_template,
                assessment,
                tool_frame,
            )
            sequence = int(artifacts.get("sequence", self.right.sequence))
            overlay_path = (
                self.output_dir / f"right_grasp_window_{sequence:04d}.png"
            )
            if not cv2.imwrite(str(overlay_path), overlay):
                raise AutonomousStop(
                    f"could not save grasp-window overlay: {overlay_path}"
                )
            image = str(overlay_path)
        scene = SceneSnapshot(
            timestamp_s=float(timestamp),
            target_xyz_m=head.target_xyz_m,
            target_instance_id=head.target_instance_id,
            ee_pose=Pose.from_se3(self.runner.rpc.get_right_ee_pose()),
            lid_visible=geometry is not None,
            # These legacy fields remain only for SceneSnapshot compatibility.
            # No decision below depends on a coloured marker or absolute pixel.
            blue_marker_visible=False,
            right_marker_px=None,
            estimated_gripper_lid_m=head.estimated_gripper_lid_m,
            predicted_clearance_m=head.predicted_clearance_m,
        )
        return RightGraspObservation(
            scene=scene,
            assessment=assessment,
            normalized_pad_target_gap=normalized_gap,
            image=image,
            target_mask=target_mask_path,
        )

    def tool_support_state(self, tool_plane):
        target = self.runner.last_target_3d
        if target is None:
            raise AutonomousStop("support-plane estimate is unavailable")
        plane_point_robot, plane_normal_robot = transform_plane(
            target.point_camera_xyz_m,
            target.plane.normal,
            self.transform,
        )
        return tool_plane.clearance_and_free_normal(
            self.runner.rpc.get_right_joint_positions(),
            self.runner.rpc.get_left_joint_positions(),
            plane_point_robot,
            plane_normal_robot,
        )


def _offline(config, args, state, calibration):
    scene = _json(args.offline_scene)
    task = _json(config["task_config"])
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
        grasp_orientation_wxyz=task["demonstration_reference"][
            "grasp_orientation_wxyz"
        ],
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
    from rollout.right_target_observer import RightTargetObserver

    expected = config["camera_udids"]
    task = _json(config["task_config"])
    selection = _json(config["grasp"]["window_selection"])
    grasp_window_template = GraspWindowTemplate.from_dict(
        selection["template"]
    )
    grasp_window_method = str(selection["selected_method"])
    grasp_orientation = tuple(
        float(value)
        for value in task["demonstration_reference"][
            "grasp_orientation_wxyz"
        ]
    )
    mapping, live_udids = configure_camera_map_by_udid(expected)
    state.event("camera_udid_preflight", mapping=mapping, connected=live_udids)
    transform = validate_calibration(
        calibration, camera_udid=expected["head"]
    )
    calibration_id = _calibration_id(calibration)
    daily_store = DailySceneStore(
        config.get("daily_scene", {}).get(
            "state_path", "runs/pasteur_daily_scene.json"
        )
    )
    torque = _json(config["torque_config"])
    runner = right = None
    try:
        runner = LiveSamGrasp(_runner_args(config, args.output_dir))
        target = task["target"]
        right = RightTargetObserver(
            runner,
            Path(args.output_dir) / "right",
            prompts=(
                target["sam_prompt"],
                target["semantic_name"],
                target["shape_class"].replace("_", " "),
            ),
            grasp_window_template=grasp_window_template,
            grasp_window_method=grasp_window_method,
        )
        runner.start()
        right.start()
        perception = LivePerception(
            runner,
            right,
            transform,
            Path(args.output_dir) / "scene",
            grasp_window_template,
            grasp_window_method,
        )
        tool_plane = MuJoCoToolPlane(config["mujoco_model"])
        reference, image = perception.head_snapshot()  # real-image SAM preflight
        acquisition = decide_state(
            GraspState.ACQUIRE_TARGET,
            GraspGates(target_visible=reference.lid_visible),
        )
        state.event(
            "sam_preflight",
            image=image,
            target_xyz_m=reference.target_xyz_m,
            instance=reference.target_instance_id,
            state=acquisition.state.value,
            action=acquisition.action,
        )
        if acquisition.state != GraspState.COARSE_ALIGN:
            raise AutonomousStop("generic target could not be acquired")

        try:
            daily_scene = daily_store.require_confirmed(
                calibration_id=calibration_id
            )
        except SceneNotConfirmed as error:
            existing = daily_store.load()
            if existing is None or existing.status in {
                "change_reported",
                "confirmed",
            }:
                inventory, inventory_image = perception.scan_inventory(
                    config.get("daily_scene", {}).get(
                        "expected_objects", []
                    )
                )
                existing = daily_store.propose(
                    objects=_daily_scene_object_proposals(
                        config, task, reference, image, inventory
                    ),
                    calibration_id=calibration_id,
                    camera_ids=expected,
                    images={
                        "head_sam": image["sam_overlay"],
                        "daily_inventory": inventory_image,
                    },
                    reason="daily_scan",
                )
            state.update(
                "WAITING_FOR_DAILY_SCENE_CONFIRMATION",
                daily_scene=existing.to_dict(),
                daily_scene_ui=config.get("daily_scene", {}).get(
                    "ui_url", "http://127.0.0.1:8765/"
                ),
            )
            raise AutonomousStop(
                f"{error}; confirm revision {existing.revision} in the daily-scene UI"
            )
        scene_revision = daily_scene.revision
        state.event(
            "daily_scene_bound",
            scene_id=daily_scene.scene_id,
            revision=scene_revision,
            confirmed_by=daily_scene.confirmed_by,
        )

        def make_plan(snapshot):
            analytic_obstacles = AnalyticObstacleSet(
                [item.__dict__ for item in daily_scene.objects]
            )

            def validator_factory():
                return CompositePoseValidator(
                    perception.esdf,
                    MuJoCoIKValidator(
                        config["mujoco_model"],
                        runner.rpc.get_right_joint_positions(),
                        left_q=runner.rpc.get_left_joint_positions(),
                    ),
                    analytic_obstacles=analytic_obstacles,
                )

            candidate = select_collision_aware_plan(
                snapshot.ee_pose,
                snapshot.target_xyz_m,
                validator_factory=validator_factory,
                grasp_orientation_wxyz=grasp_orientation,
                lift_m=float(config["planner"]["lift_m"]),
                pregrasp_height_m=float(config["planner"]["pregrasp_height_m"]),
                final_height_m=float(config["planner"]["final_height_m"]),
                detour_m=float(
                    config["planner"].get("lateral_detour_m", 0.070)
                ),
            )
            candidate.metadata["mujoco_model"] = config["mujoco_model"]
            candidate.metadata["daily_scene_revision"] = scene_revision
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
            enforce_torque_stop=torque_stop_enabled_from_config(torque),
        )
        state.event(
            "torque_policy",
            mode=torque.get("motion_torque_policy", "enforce"),
            thresholds_nm=torque["thresholds"]["right"],
        )

        def execute_with_torque_observation(waypoints):
            warning_count = executor.torque_warning_count
            result = executor.execute(waypoints)
            if executor.torque_warning_count > warning_count:
                state.event(
                    "torque_observer_warning",
                    warning=executor.last_torque_warning,
                    total_warning_count=executor.torque_warning_count,
                )
            return result

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
        def execute_visual_plan(candidate, plan_reference, replans):
            while True:
                replan_now = False
                for index, chunk in enumerate(candidate.chunks()):
                    try:
                        daily_store.require_confirmed(
                            revision=scene_revision,
                            calibration_id=calibration_id,
                        )
                    except SceneNotConfirmed as error:
                        executor.sender.hold()
                        raise AutonomousStop(str(error)) from error
                    commanded = execute_with_torque_observation(chunk)
                    current, image = perception.head_snapshot()
                    decision = decide_replan(
                        now_s=time.time(),
                        reference=plan_reference,
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
                        raise AutonomousStop(
                            "fresh observation unavailable"
                        )
                    if decision.action == "REPLAN":
                        if "target_shift" in decision.reasons or (
                            "target_instance_changed" in decision.reasons
                        ):
                            changed = daily_store.report_change(
                                "automatic RGB-D/SAM target change detection"
                            )
                            executor.sender.hold()
                            state.event(
                                "daily_scene_invalidated",
                                revision=changed.revision,
                                reasons=decision.reasons,
                            )
                            raise AutonomousStop(
                                "bench scene changed; operator reconfirmation required"
                            )
                        plan_reference = current
                        replan_now = True
                        break
                if not replan_now:
                    return current, replans
                replans += 1
                if replans > maximum_replans:
                    executor.sender.hold()
                    raise AutonomousStop(
                        "maximum visual replans exceeded"
                    )
                candidate = make_plan(plan_reference)
                state.event(
                    "replanned",
                    count=replans,
                    plan=plan_to_dict(candidate),
                )

        reference, replans = execute_visual_plan(plan, reference, 0)

        grasp = config["grasp"]
        thresholds = PrecloseThresholds.from_dict(grasp)
        maximum_probes = int(grasp["maximum_descent_probes"])
        required_lowest_point_probes = int(
            grasp["required_lowest_point_probes"]
        )
        maximum_probe_below_support_m = float(
            grasp["maximum_probe_below_support_m"]
        )
        recoveries = 0
        right_observation = None
        readiness = None
        lowest_point_confirmed = False
        lowest_point_streak = 0
        probe_index = 0
        allowed = False
        while probe_index <= maximum_probes:
            right_observation = perception.right_snapshot(reference)
            orientation_error = quaternion_angle_deg(
                right_observation.scene.ee_pose.wxyz,
                grasp_orientation,
            )
            tool_clearance, free_normal = perception.tool_support_state(
                tool_plane
            )
            window_ready = bool(
                right_observation.assessment is not None
                and right_observation.assessment.allowed_to_close
            )
            gap = right_observation.normalized_pad_target_gap
            readiness = assess_preclose(
                target_visible=right_observation.target_visible,
                target_in_window=window_ready,
                normalized_image_gap=gap,
                orientation_error_deg=orientation_error,
                tool_support_clearance_m=tool_clearance,
                thresholds=thresholds,
            )
            gates = GraspGates(
                target_visible=readiness.target_visible,
                coarse_aligned=True,
                tool_horizontal=readiness.tool_horizontal,
                target_in_window=(
                    readiness.target_in_window
                    and readiness.finger_pad_at_target
                ),
                tip_at_support=readiness.tool_tip_at_support,
                lowest_point_reached=lowest_point_confirmed,
                recoveries=recoveries,
                maximum_recoveries=int(
                    grasp["maximum_preclose_recoveries"]
                ),
            )
            coarse = decide_state(GraspState.COARSE_ALIGN, gates)
            orientation = decide_state(coarse.state, gates)
            fine = decide_state(orientation.state, gates)
            preclose = decide_state(fine.state, gates)
            if preclose.state == GraspState.DESCEND_PROBE:
                preclose = decide_state(preclose.state, gates)
            allowed = bool(
                lowest_point_confirmed
                and not (
                    set(readiness.reasons)
                    - {"tool_tip_not_at_support_plane"}
                )
                and preclose.state == GraspState.CLOSE_AND_MONITOR
            )
            state.event(
                "pregrasp_gate",
                probe=probe_index,
                allowed=allowed,
                reasons=readiness.reasons,
                state=preclose.state.value,
                action=preclose.action,
                method=grasp_window_method,
                window=(
                    None
                    if right_observation.assessment is None
                    else right_observation.assessment.to_dict()
                ),
                normalized_pad_target_gap=gap,
                orientation_error_deg=orientation_error,
                tool_support_clearance_m=tool_clearance,
                lowest_point_confirmed=lowest_point_confirmed,
                lowest_point_streak=lowest_point_streak,
                readiness=readiness.to_dict(),
                image=right_observation.image,
            )
            if allowed:
                break
            if lowest_point_confirmed:
                lowest_point_visual_reasons = tuple(
                    sorted(
                        set(readiness.reasons)
                        - {"tool_tip_not_at_support_plane"}
                    )
                )
                recoveries += 1
                if recoveries > int(
                    grasp["maximum_preclose_recoveries"]
                ):
                    executor.sender.hold()
                    raise AutonomousStop(
                        "lowest-point SAM gate rejected closure after "
                        f"{recoveries - 1} realignments: "
                        f"{lowest_point_visual_reasons}"
                    )
                recovery_start = Pose.from_se3(
                    runner.rpc.get_right_ee_pose()
                )
                recovery_finish = Pose(
                    recovery_start.wxyz,
                    tuple(
                        float(value)
                        for value in (
                            np.asarray(recovery_start.xyz)
                            + float(grasp["realignment_lift_m"])
                            * np.asarray(free_normal)
                        )
                    ),
                )
                recovery_waypoints = minimum_jerk_segment(
                    recovery_start,
                    recovery_finish,
                    1.0,
                    stage="lowest_point_sam_recovery_lift",
                )
                recovery_simulation = MuJoCoIKValidator(
                    config["mujoco_model"],
                    runner.rpc.get_right_joint_positions(),
                    left_q=runner.rpc.get_left_joint_positions(),
                )
                for waypoint in recovery_waypoints:
                    recovery_simulation.validate(waypoint.pose)
                execute_with_torque_observation(recovery_waypoints)
                reference, _ = perception.head_snapshot()
                recovery_plan = make_plan(reference)
                reference, replans = execute_visual_plan(
                    recovery_plan, reference, replans
                )
                state.event(
                    "lowest_point_sam_realign",
                    recovery=recoveries,
                    reasons=lowest_point_visual_reasons,
                    lifted_m=float(grasp["realignment_lift_m"]),
                )
                lowest_point_confirmed = False
                lowest_point_streak = 0
                probe_index = 0
                continue
            if not readiness.tool_horizontal:
                executor.sender.hold()
                raise AutonomousStop(
                    "tool lost demonstrated horizontal orientation during "
                    "lowest-point descent"
                )
            if tool_clearance < -maximum_probe_below_support_m:
                executor.sender.hold()
                raise AutonomousStop(
                    "maximum bounded distance below the observed support "
                    "plane exceeded"
                )
            if probe_index >= maximum_probes:
                executor.sender.hold()
                raise AutonomousStop(
                    "maximum bounded descent probes exceeded"
                )

            requested = min(
                float(grasp["descent_probe_step_m"]),
                max(
                    0.0,
                    tool_clearance
                    + maximum_probe_below_support_m,
                ),
            )
            if requested <= 0:
                executor.sender.hold()
                raise AutonomousStop(
                    "support-plane gate is inconsistent with probe distance"
                )
            descent_direction = -np.asarray(free_normal, dtype=float)
            start_pose = right_observation.scene.ee_pose
            finish_xyz = (
                np.asarray(start_pose.xyz)
                + requested * descent_direction
            )
            finish_pose = Pose(
                start_pose.wxyz,
                tuple(float(value) for value in finish_xyz),
            )
            probe_waypoints = minimum_jerk_segment(
                start_pose,
                finish_pose,
                float(grasp["descent_probe_duration_s"]),
                stage="bounded_support_probe",
            )
            simulation = MuJoCoIKValidator(
                config["mujoco_model"],
                runner.rpc.get_right_joint_positions(),
                left_q=runner.rpc.get_left_joint_positions(),
            )
            for waypoint in probe_waypoints:
                simulation.validate(waypoint.pose)
            predicted_clearance = tool_clearance - requested
            if predicted_clearance < -maximum_probe_below_support_m:
                executor.sender.hold()
                raise AutonomousStop(
                    "bounded probe would penetrate the support plane"
                )
            torque_before = np.asarray(
                runner.rpc.get_right_joint_torque(), dtype=float
            )
            pose_before = Pose.from_se3(runner.rpc.get_right_ee_pose())
            execute_with_torque_observation(probe_waypoints)
            pose_after = Pose.from_se3(runner.rpc.get_right_ee_pose())
            torque_after = np.asarray(
                runner.rpc.get_right_joint_torque(), dtype=float
            )
            reference, _ = perception.head_snapshot()
            clearance_after, free_normal_after = (
                perception.tool_support_state(tool_plane)
            )
            probe = assess_descent_probe(
                requested_distance_m=requested,
                measured_delta_xyz_m=(
                    np.asarray(pose_after.xyz)
                    - np.asarray(pose_before.xyz)
                ),
                descent_direction_xyz=descent_direction,
                torque_before_nm=torque_before,
                torque_after_nm=torque_after,
                support_clearance_m=clearance_after,
                maximum_support_clearance_m=(
                    thresholds.maximum_tip_clearance_m
                ),
                minimum_progress_ratio_at_contact=float(
                    grasp[
                        "minimum_descent_progress_ratio_at_contact"
                    ]
                ),
                minimum_torque_change_nm=float(
                    grasp["minimum_torque_change_nm"]
                ),
            )
            lowest_point = assess_lowest_point(
                probe=probe,
                support_clearance_m=clearance_after,
                maximum_support_clearance_m=(
                    thresholds.maximum_tip_clearance_m
                ),
                minimum_progress_ratio=float(
                    grasp[
                        "minimum_descent_progress_ratio_at_contact"
                    ]
                ),
                minimum_torque_change_nm=float(
                    grasp["minimum_torque_change_nm"]
                ),
                previous_consecutive_candidates=lowest_point_streak,
                required_consecutive_candidates=(
                    required_lowest_point_probes
                ),
            )
            lowest_point_streak = lowest_point.consecutive_candidates
            lowest_point_confirmed = lowest_point.confirmed
            state.event(
                "bounded_descent_probe",
                probe=probe_index,
                requested_distance_m=probe.requested_distance_m,
                measured_progress_m=probe.measured_progress_m,
                progress_ratio=probe.progress_ratio,
                maximum_torque_change_nm=probe.maximum_torque_change_nm,
                early_contact=probe.early_contact,
                support_clearance_m=clearance_after,
                lowest_point_candidate=lowest_point.candidate,
                lowest_point_confirmed=lowest_point.confirmed,
                lowest_point_streak=lowest_point.consecutive_candidates,
            )
            probe_index += 1
        if not allowed:
            raise AutonomousStop("preclose feedback loop did not converge")

        before_target = np.asarray(reference.target_xyz_m)
        runner.rpc.close_right_gripper()
        closure_samples = []
        closure = None
        closure_config = task["closure"]
        closure_deadline = time.monotonic() + float(
            closure_config["maximum_monitor_s"]
        )
        while time.monotonic() < closure_deadline:
            time.sleep(float(closure_config["sample_period_s"]))
            closure_samples.append(
                float(runner.rpc.get_right_gripper_exact())
            )
            if len(closure_samples) < 5:
                continue
            closure = assess_stable_closure(
                closure_samples,
                empty_close_ratio=float(grasp["empty_close_ratio"]),
            )
            if not closure.still_closing:
                break
        if closure is None or closure.still_closing:
            raise AutonomousStop(
                "gripper closure did not reach a stable aperture"
            )
        ratio = closure.final_open_ratio
        state.event(
            "stable_closure",
            captured=closure.captured,
            state=decide_state(
                GraspState.CLOSE_AND_MONITOR,
                GraspGates(closure_captured=closure.captured),
            ).state.value,
            final_open_ratio=ratio,
            stable_range=closure.stable_range,
            samples=closure_samples,
        )
        if not closure.captured:
            raise AutonomousStop(
                "gripper stably closed empty; lid was not captured"
            )
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
        execute_with_torque_observation(lift)
        after, _ = perception.head_snapshot()
        after_right = perception.right_snapshot(after)
        followed = float(np.asarray(after.target_xyz_m)[2] - before_target[2])
        success = (
            ratio > float(grasp["empty_close_ratio"])
            and after_right.target_visible
            and followed >= float(grasp["minimum_target_follow_m"])
        )
        verification = decide_state(
            GraspState.VERIFY_LIFT,
            GraspGates(target_followed_lift=success),
        )
        state.update(
            "SUCCEEDED" if success else "FAILED_VERIFICATION",
            result={
                "gripper_ratio": ratio,
                "target_visible": after_right.target_visible,
                "target_follow_m": followed,
                "right_image": after_right.image,
                "state": verification.state.value,
                "action": verification.action,
            },
        )
        if not success:
            raise AutonomousStop("post-lift grasp verification failed")
        return 0
    except BaseException as error:
        if state.payload.get("status") != "WAITING_FOR_DAILY_SCENE_CONFIRMATION":
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
    parser.add_argument("--sam-endpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset-calibration", action="store_true")
    args = parser.parse_args(argv)

    config = _json(args.config)
    if args.sam_endpoint:
        config["sam_endpoint"] = args.sam_endpoint
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
