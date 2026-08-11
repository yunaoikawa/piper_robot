#!/usr/bin/env python3
"""Execute only a newly audited prefix of a scene-anchored right-arm path."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.rpc import RPCClient
from rollout.dish_transport_rehearsal import (
    CartesianAirTransportStreamer,
    ProductionArmKinematics,
    _append_joint_branch_path,
    audit_joint_path,
)
from rollout.gripper_level import JawLevelReference, assess_jaw_level
from rollout.teleop_trajectory_stream import (
    CONTROL_HZ,
    JointTrajectorySample,
    ProductionRightFK,
    TeleopTrajectoryStreamer,
)
from src.run_dish_transport_rehearsal import SynchronizedCameraSet


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _dense(path: np.ndarray, maximum_step_rad: float) -> np.ndarray:
    pieces = [np.asarray(path[0], dtype=float).reshape(1, 6)]
    for first, second in zip(path, path[1:]):
        pieces.append(
            _append_joint_branch_path(first, second, maximum_step_rad)[1:]
        )
    return np.vstack(pieces)


def _capture(
    output: Path,
    camera_names: dict,
    *,
    timeout_s: float,
    maximum_skew_s: float,
) -> dict:
    with SynchronizedCameraSet(
        camera_names["head"], camera_names["left"], camera_names["right"]
    ) as cameras:
        capture = cameras.capture(
            timeout_s=timeout_s, maximum_skew_s=maximum_skew_s
        )
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("head", "left", "right"):
        path = output / f"after_safe_prefix_{name}.jpg"
        if not cv2.imwrite(str(path), capture[f"{name}_bgr"]):
            raise RuntimeError(f"failed to write {path}")
        paths[name] = str(path)
    if capture.get("head_depth_m") is not None:
        depth_path = output / "after_safe_prefix_head_depth.npy"
        np.save(depth_path, np.asarray(capture["head_depth_m"], dtype=np.float32))
        paths["head_depth_m"] = str(depth_path)
    paths["timestamp_skew_s"] = float(capture["timestamp_skew_s"])
    return paths


def _rotation_error_rad(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> float:
    first_xyzw = np.roll(np.asarray(first_wxyz, dtype=float), -1)
    second_xyzw = np.roll(np.asarray(second_wxyz, dtype=float), -1)
    relative = Rotation.from_quat(first_xyzw).inv() * Rotation.from_quat(second_xyzw)
    return float(relative.magnitude())


def _teleop_samples(
    q_path: np.ndarray,
    pose_path: np.ndarray,
    *,
    speed_m_s: float,
    gripper_open_ratio: float,
) -> list[JointTrajectorySample]:
    """Resample an audited branch at the proven teleop command frequency."""

    if speed_m_s <= 0.0:
        raise ValueError("Cartesian speed must be positive")
    q_path = np.asarray(q_path, dtype=float)
    pose_path = np.asarray(pose_path, dtype=float)
    linear = np.linalg.norm(np.diff(pose_path[:, 4:], axis=0), axis=1)
    angular = np.asarray(
        [
            _rotation_error_rad(first[:4], second[:4])
            for first, second in zip(pose_path, pose_path[1:])
        ],
        dtype=float,
    )
    segment = np.maximum(linear, 0.12 * angular)
    cumulative = np.r_[0.0, np.cumsum(segment)]
    if cumulative[-1] <= 1e-9:
        raise ValueError("audited path has no Cartesian motion")
    duration = float(cumulative[-1] / speed_m_s)
    command_count = max(1, int(np.ceil(duration * CONTROL_HZ)))
    distances = np.linspace(0.0, cumulative[-1], command_count + 1)
    times = np.linspace(
        1.0 / CONTROL_HZ,
        duration + 1.0 / CONTROL_HZ,
        command_count + 1,
    )
    q_samples = np.column_stack(
        [np.interp(distances, cumulative, q_path[:, joint]) for joint in range(6)]
    )
    q_samples[-1] = q_path[-1]
    return [
        JointTrajectorySample(
            t_s=float(timestamp),
            stage="audited_local_cartesian_adjustment",
            right_q_physical_rad=q.copy(),
            right_gripper_open_ratio=float(gripper_open_ratio),
        )
        for timestamp, q in zip(times, q_samples)
    ]


def _execute_direct_joint_teleop(
    streamer: TeleopTrajectoryStreamer,
    rpc,
    samples: list[JointTrajectorySample],
    *,
    endpoint_settle_s: float,
) -> dict:
    """Use the successful branch-locked teleop command path.

    The MIT/gain preparation and measured hold come from the shared teleop
    streamer.  Unlike Cartesian replay, the audited joint branch is sent
    directly, matching the production thin-object grasp controller.
    """

    if not samples:
        raise ValueError("cannot execute an empty direct-joint path")
    if not 0.0 <= endpoint_settle_s <= 3.0:
        raise ValueError("endpoint settle must be within [0, 3] seconds")
    motion_error = None
    prepared = None
    maximum_joint_error = 0.0
    settle_commands = 0
    try:
        prepared = streamer._prepare()
        start_error = float(
            np.max(
                np.abs(
                    prepared["start_q"]
                    - np.asarray(samples[0].right_q_physical_rad, dtype=float)
                )
            )
        )
        if start_error > streamer.maximum_start_joint_error_rad:
            raise RuntimeError(
                "right arm is not at the audited direct-joint start: "
                f"max_joint_error={start_error:.4f}rad"
            )
        started = time.monotonic()
        for index, sample in enumerate(samples, start=1):
            streamer._check_torque(f"during {sample.stage}")
            rpc.set_right_joint_target(
                np.asarray(sample.right_q_physical_rad, dtype=float),
                gripper_target=float(sample.right_gripper_open_ratio),
                preview_time=0.05,
            )
            if index % streamer.tracking_check_interval == 0:
                measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
                maximum_joint_error = max(
                    maximum_joint_error,
                    float(
                        np.max(
                            np.abs(
                                measured
                                - np.asarray(sample.right_q_physical_rad, dtype=float)
                            )
                        )
                    ),
                )
            remaining = started + float(sample.t_s) - time.monotonic()
            if remaining < -2.0 / CONTROL_HZ:
                raise RuntimeError("direct-joint teleop stream missed its deadline")
            if remaining > 0.0:
                time.sleep(remaining)
        endpoint = np.asarray(samples[-1].right_q_physical_rad, dtype=float)
        settle_count = int(np.ceil(endpoint_settle_s * CONTROL_HZ))
        settle_started = time.monotonic()
        for settle_index in range(1, settle_count + 1):
            streamer._check_torque("during direct-joint endpoint settle")
            rpc.set_right_joint_target(
                endpoint,
                gripper_target=float(samples[-1].right_gripper_open_ratio),
                preview_time=0.05,
            )
            settle_commands += 1
            remaining = (
                settle_started + settle_index / CONTROL_HZ - time.monotonic()
            )
            if remaining > 0.0:
                time.sleep(remaining)
    except BaseException as error:
        motion_error = error
        raise
    finally:
        try:
            streamer._finish()
        except BaseException as cleanup_error:
            if motion_error is None:
                raise
            note = getattr(motion_error, "add_note", None)
            if note is not None:
                note(f"direct-joint cleanup also failed: {cleanup_error!r}")
    final_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    maximum_joint_error = max(
        maximum_joint_error,
        float(
            np.max(
                np.abs(
                    final_q
                    - np.asarray(samples[-1].right_q_physical_rad, dtype=float)
                )
            )
        ),
    )
    return {
        "commands_sent": True,
        "command_path": "set_right_joint_target_30hz_success_branch",
        "control_hz": CONTROL_HZ,
        "sample_count": len(samples),
        "endpoint_settle_s": endpoint_settle_s,
        "endpoint_settle_command_count": settle_commands,
        "maximum_tracking_joint_error_rad": maximum_joint_error,
        "torque_stop_enforced": False,
        "torque_warning_count": streamer.torque_warning_count,
        "last_torque_warning": streamer.last_torque_warning,
        "final_right_q_physical_rad": final_q.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--stop-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-joint-step-rad", type=float, default=0.04)
    parser.add_argument(
        "--speed-m-s",
        type=float,
        help="Override the configured Cartesian speed for a local adjustment.",
    )
    parser.add_argument(
        "--endpoint-settle-s",
        type=float,
        help="Override the configured endpoint republish window (maximum 3 s).",
    )
    parser.add_argument(
        "--reverse-from-nearest",
        action="store_true",
        help="Return from the nearest source-plan sample to stop-index.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.plan.read_text())
    config = json.loads(_resolve(manifest["config"]).read_text())
    plan = manifest["plans"][0]
    sparse = np.asarray(plan["q_physical_rad"], dtype=float)
    if not 0 <= args.stop_index < len(sparse):
        raise ValueError("stop-index is outside the planned path")

    rpc_settings = config["rpc"]
    rpc = RPCClient(
        str(rpc_settings["host"]),
        int(rpc_settings["port"]),
        timeout_ms=int(rpc_settings["timeout_ms"]),
    )
    measured_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    nearest_index = None
    nearest_error = None
    if args.reverse_from_nearest:
        nearest_index = int(np.argmin(np.linalg.norm(sparse - measured_q, axis=1)))
        nearest_error = float(np.max(np.abs(sparse[nearest_index] - measured_q)))
        if args.stop_index > nearest_index:
            raise ValueError("reverse stop-index follows the nearest source-plan sample")
        selected = np.vstack(
            (measured_q, sparse[nearest_index : args.stop_index - 1 : -1])
        )
    else:
        selected = np.vstack((measured_q, sparse[: args.stop_index + 1]))
    dense_q = _dense(selected, float(args.maximum_joint_step_rad))
    geometry = config["geometry"]
    audit = audit_joint_path(
        _resolve(config["planning_model"]),
        "right",
        dense_q,
        dish_radius_m=float(geometry["dish_diameter_m"]) / 2.0,
        dish_thickness_m=float(geometry["dish_thickness_m"]),
        dish_center_offset_ee_m=geometry["dish_center_offset_ee_m"],
        ignored_environment_bodies=geometry.get("ignored_absent_scene_bodies", []),
        carrying_sample_range=(0, len(dense_q) - 1),
        minimum_dish_clearance_m=float(config["planning"]["minimum_dish_clearance_m"]),
    )
    if not audit["accepted"]:
        raise RuntimeError(f"dense prefix collision audit failed: {audit}")

    production = ProductionArmKinematics(
        _resolve(config["production_model"]), "right"
    )
    poses = np.asarray([production.pose(q) for q in dense_q])
    measured_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    start_position_error = float(np.linalg.norm(measured_pose[4:] - poses[0, 4:]))
    start_rotation_error = _rotation_error_rad(measured_pose[:4], poses[0, :4])
    if start_position_error > 0.01:
        raise RuntimeError(
            f"current right pose is not the audited start: {start_position_error:.4f}m"
        )

    execution = config["execution"]
    requested_translation_m = float(
        np.linalg.norm(poses[-1, 4:] - poses[0, 4:])
    )
    # A broad tracking gate is useful on long transport paths, but it can
    # falsely call a millimetric placement adjustment successful without any
    # robot motion.  For local adjustments, require the endpoint to resolve at
    # least 75% of the requested translation.
    local_endpoint_error_m = max(0.00075, 0.25 * requested_translation_m)
    effective_tracking_position_error_m = min(
        float(execution["maximum_tracking_position_error_m"]),
        local_endpoint_error_m,
    )
    if start_rotation_error > float(execution["maximum_tracking_rotation_error_rad"]):
        raise RuntimeError(
            "current right orientation is not the audited start: "
            f"{np.degrees(start_rotation_error):.2f}deg"
        )
    command_speed_m_s = float(
        execution["speed_m_s"] if args.speed_m_s is None else args.speed_m_s
    )
    teleop_fk = ProductionRightFK(_resolve(config["production_model"]))
    streamer = TeleopTrajectoryStreamer(
        rpc,
        teleop_fk,
        torque_limit_nm=execution["torque_limits_nm"]["right"],
        consecutive_torque_samples=int(execution["torque_consecutive_samples"]),
        enforce_torque_stop=False,
        tracking_check_interval=int(execution["tracking_check_interval"]),
        maximum_tracking_position_error_m=float(
            effective_tracking_position_error_m
        ),
        maximum_tracking_rotation_error_rad=float(
            execution["maximum_tracking_rotation_error_rad"]
        ),
    )
    teleop_samples = _teleop_samples(
        dense_q,
        poses,
        speed_m_s=command_speed_m_s,
        gripper_open_ratio=float(execution["air_gripper_open_ratio"]),
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "piper_robot.audited_scene_path_prefix_run/v1",
        "status": "running",
        "source_plan": str(args.plan.resolve()),
        "stop_index": int(args.stop_index),
        "reverse_from_nearest": bool(args.reverse_from_nearest),
        "nearest_source_plan_index": nearest_index,
        "nearest_source_plan_max_joint_error_rad": nearest_error,
        "dense_sample_count": len(dense_q),
        "collision_audit": audit,
        "start_position_error_m": start_position_error,
        "start_rotation_error_rad": start_rotation_error,
        "requested_translation_m": requested_translation_m,
        "effective_tracking_position_error_m": effective_tracking_position_error_m,
        "command_speed_m_s": command_speed_m_s,
        "control_path": "set_right_joint_target_30hz_success_branch",
    }
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        report["motion"] = _execute_direct_joint_teleop(
            streamer,
            rpc,
            teleop_samples,
            endpoint_settle_s=float(
                2.0 if args.endpoint_settle_s is None else args.endpoint_settle_s
            ),
        )
        final_pose = np.asarray(
            rpc.get_right_ee_pose().parameters(), dtype=float
        )
        endpoint_position_error = float(
            np.linalg.norm(final_pose[4:] - poses[-1, 4:])
        )
        report["endpoint_position_error_m"] = endpoint_position_error
        if endpoint_position_error > effective_tracking_position_error_m:
            raise RuntimeError(
                "right teleop endpoint did not make the requested local progress: "
                f"position={endpoint_position_error:.4f}m, "
                f"limit={effective_tracking_position_error_m:.4f}m"
            )
        jaw = assess_jaw_level(
            final_pose, JawLevelReference(), planned=False
        )
        refinement_config = execution.get("jaw_level_refinement", {})
        report["jaw_level_refinement"] = None
        if not jaw.accepted and refinement_config.get("enabled", True):
            refinement_streamer = CartesianAirTransportStreamer(
                rpc,
                "right",
                torque_limit_nm=execution["torque_limits_nm"]["right"],
                torque_samples=int(execution["torque_consecutive_samples"]),
                control_hz=float(execution["control_hz"]),
                preview_time_s=float(execution["preview_time_s"]),
                tracking_interval=int(execution["tracking_check_interval"]),
                maximum_tracking_position_error_m=float(
                    effective_tracking_position_error_m
                ),
                maximum_tracking_rotation_error_rad=float(
                    execution["maximum_tracking_rotation_error_rad"]
                ),
                final_settle_s=float(
                    execution["endpoint_settle_s"]
                    if args.endpoint_settle_s is None
                    else args.endpoint_settle_s
                ),
            )
            refinement_reference = replace(
                JawLevelReference(),
                maximum_checkpoint_tilt_deg=max(
                    0.75,
                    float(refinement_config["target_maximum_tilt_deg"]),
                ),
                maximum_tip_height_difference_m=max(
                    0.00075,
                    float(
                        refinement_config[
                            "target_maximum_tip_height_difference_m"
                        ]
                    ),
                ),
            )
            report["jaw_level_refinement"] = refinement_streamer.refine_jaw_level(
                refinement_reference,
                gripper_open_ratio=float(execution["air_gripper_open_ratio"]),
                maximum_attempts=int(refinement_config["maximum_attempts"]),
                maximum_correction_deg=float(
                    refinement_config["maximum_correction_deg"]
                ),
                maximum_total_correction_deg=float(
                    refinement_config["maximum_total_correction_deg"]
                ),
                correction_duration_s=float(
                    refinement_config["correction_duration_s"]
                ),
                settle_s=float(refinement_config["settle_s"]),
                maximum_xyz_drift_m=float(
                    refinement_config["maximum_xyz_drift_m"]
                ),
                maximum_xyz_correction_per_attempt_m=float(
                    refinement_config["maximum_xyz_correction_per_attempt_m"]
                ),
                maximum_xyz_command_offset_m=float(
                    refinement_config["maximum_xyz_command_offset_m"]
                ),
                hard_xyz_drift_m=float(refinement_config["hard_xyz_drift_m"]),
            )
            final_pose = refinement_streamer.measured_pose()
            jaw = assess_jaw_level(
                final_pose, JawLevelReference(), planned=False
            )
        report["final_pose_wxyz_xyz"] = final_pose.tolist()
        report["jaw_level"] = jaw.to_dict()
        report["images"] = _capture(
            output / "images",
            config["camera_names"],
            timeout_s=float(execution["camera_timeout_s"]),
            maximum_skew_s=float(execution["maximum_camera_timestamp_skew_s"]),
        )
        report["status"] = (
            "held_for_visual_replan"
            if jaw.accepted
            else "held_jaw_level_rejected"
        )
    except BaseException as error:
        streamer.hold_measured()
        report["status"] = "error_hold"
        report["error"] = repr(error)
        raise
    finally:
        (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
