#!/usr/bin/env python3
"""Aim a carrier wrist camera at a transport destination without moving the load."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.rpc import RPCClient
from rollout.dish_transport_rehearsal import (
    CartesianAirTransportStreamer,
    PRODUCTION_BRANCH,
    ProductionArmKinematics,
    _append_joint_branch_path,
    audit_joint_path,
)
from rollout.gripper_level import JawLevelReference, leveled_pose
from src.run_dish_transport_rehearsal import SynchronizedCameraSet, _level_reference


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _camera_relative_to_ee(model_path: Path, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return physical-right wrist camera pose expressed in the right EE frame."""

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    branch = PRODUCTION_BRANCH["right"]
    qpos = [model.joint(f"{branch}_joint{i}").qposadr[0] for i in range(1, 7)]
    data.qpos[qpos] = q
    mujoco.mj_forward(model, data)
    camera = model.camera(f"{branch}_egocentric")
    ee = model.site(f"{branch}_ee")
    world_R_ee = np.asarray(data.site_xmat[ee.id], dtype=float).reshape(3, 3)
    world_p_ee = np.asarray(data.site_xpos[ee.id], dtype=float)
    world_R_camera = np.asarray(data.cam_xmat[camera.id], dtype=float).reshape(3, 3)
    world_p_camera = np.asarray(data.cam_xpos[camera.id], dtype=float)
    return (
        world_R_ee.T @ (world_p_camera - world_p_ee),
        world_R_ee.T @ world_R_camera,
    )


def _aim_score(
    yaw_rad: float,
    base_R: np.ndarray,
    carried_center: np.ndarray,
    destination: np.ndarray,
    load_offset_ee: np.ndarray,
    camera_p_ee: np.ndarray,
    camera_R_ee: np.ndarray,
) -> float:
    world_R_ee = Rotation.from_rotvec([0.0, 0.0, yaw_rad]).as_matrix() @ base_R
    world_p_ee = carried_center - world_R_ee @ load_offset_ee
    world_p_camera = world_p_ee + world_R_ee @ camera_p_ee
    camera_forward = world_R_ee @ camera_R_ee @ np.asarray([0.0, 0.0, -1.0])
    target_ray = destination - world_p_camera
    target_ray /= np.linalg.norm(target_ray)
    return float(math.acos(np.clip(camera_forward @ target_ray, -1.0, 1.0)))


def _capture(output: Path, names: dict, timeout_s: float, maximum_skew_s: float) -> dict:
    with SynchronizedCameraSet(names["head"], names["left"], names["right"]) as cameras:
        capture = cameras.capture(timeout_s=timeout_s, maximum_skew_s=maximum_skew_s)
    output.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("head", "left", "right"):
        path = output / f"camera_aim_{name}.jpg"
        if not cv2.imwrite(str(path), capture[f"{name}_bgr"]):
            raise RuntimeError(f"failed to write {path}")
        files[name] = str(path)
    files["timestamp_skew_s"] = float(capture["timestamp_skew_s"])
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--orientation-samples", type=int, default=41)
    parser.add_argument("--vertical-samples", type=int, default=11)
    parser.add_argument("--maximum-absolute-yaw-deg", type=float, default=110.0)
    parser.add_argument(
        "--forced-yaw-deg",
        type=float,
        help="Use an empirically verified signed yaw instead of the camera model.",
    )
    parser.add_argument("--lift-search-step-m", type=float, default=0.01)
    parser.add_argument("--maximum-aim-lift-m", type=float, default=0.12)
    parser.add_argument("--retreat-plan-index", type=int, default=7)
    parser.add_argument(
        "--use-current-anchor",
        action="store_true",
        help="Apply an incremental camera correction from the measured pose.",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.plan.read_text())
    config = json.loads(_resolve(manifest["config"]).read_text())
    source_plan = manifest["plans"][0]
    geometry = config["geometry"]
    load_offset = np.asarray(geometry["dish_center_offset_ee_m"], dtype=float)
    final_pose = np.asarray(source_plan["poses_wxyz_xyz"][-1], dtype=float)
    final_R = Rotation.from_quat(final_pose[[1, 2, 3, 0]]).as_matrix()
    destination = final_pose[4:] + final_R @ load_offset

    rpc_config = config["rpc"]
    rpc = RPCClient(
        rpc_config["host"], int(rpc_config["port"]), timeout_ms=int(rpc_config["timeout_ms"])
    )
    current_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    current_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    sparse_q = np.asarray(source_plan["q_physical_rad"], dtype=float)
    retreat_index = int(args.retreat_plan_index)
    if args.use_current_anchor:
        nearest_index = None
        nearest_error = None
        prefix_q = current_q.reshape(1, 6)
        anchor_q = current_q
    else:
        nearest_index = int(np.argmin(np.linalg.norm(sparse_q - current_q, axis=1)))
        nearest_error = float(np.max(np.abs(sparse_q[nearest_index] - current_q)))
        if nearest_error > 0.15:
            raise RuntimeError(
                f"current joint state is not on the source plan: {nearest_error:.3f}rad"
            )
        if not 0 <= retreat_index <= nearest_index:
            raise ValueError("retreat-plan-index must precede the current source-plan index")
        prefix_parts = [current_q.reshape(1, 6)]
        reverse_waypoints = [
            sparse_q[nearest_index],
            *sparse_q[nearest_index - 1 : retreat_index - 1 : -1],
        ]
        for waypoint in reverse_waypoints:
            prefix_parts.append(
                _append_joint_branch_path(prefix_parts[-1][-1], waypoint, 0.04)[1:]
            )
        prefix_q = np.vstack(prefix_parts)
        anchor_q = prefix_q[-1]

    reference = _level_reference(config["level_config"])
    kinematics = ProductionArmKinematics(_resolve(config["production_model"]), "right")
    anchor_pose = kinematics.pose(anchor_q)
    anchor_R = Rotation.from_quat(anchor_pose[[1, 2, 3, 0]]).as_matrix()
    carried_center = anchor_pose[4:] + anchor_R @ load_offset
    level = leveled_pose(anchor_pose, reference)
    level_R = Rotation.from_quat(level[[1, 2, 3, 0]]).as_matrix()
    camera_p_ee, camera_R_ee = _camera_relative_to_ee(
        _resolve(config["production_model"]), anchor_q
    )
    yaw_limit = math.radians(float(args.maximum_absolute_yaw_deg))
    count = int(args.orientation_samples)
    vertical_count = int(args.vertical_samples)
    if count < 5 or vertical_count < 2:
        raise ValueError("orientation and vertical sample counts are too small")
    lift_step = float(args.lift_search_step_m)
    maximum_lift = float(args.maximum_aim_lift_m)
    if lift_step <= 0.0 or maximum_lift < 0.0:
        raise ValueError("lift search bounds are invalid")
    trials = []
    selected = None
    for lift in np.arange(0.0, maximum_lift + lift_step * 0.5, lift_step):
        lifted_center = carried_center + np.asarray([0.0, 0.0, float(lift)])
        if args.forced_yaw_deg is None:
            optimized = minimize_scalar(
                _aim_score,
                bounds=(-yaw_limit, yaw_limit),
                method="bounded",
                options={"xatol": math.radians(0.05)},
                args=(
                    level_R,
                    lifted_center,
                    destination,
                    load_offset,
                    camera_p_ee,
                    camera_R_ee,
                ),
            )
            if not optimized.success:
                trials.append(
                    {"lift_m": float(lift), "accepted": False, "reason": "aim"}
                )
                continue
            yaw = float(optimized.x)
        else:
            yaw = math.radians(float(args.forced_yaw_deg))
            if abs(yaw) > yaw_limit:
                raise ValueError("forced yaw exceeds maximum-absolute-yaw-deg")
        axis_error = _aim_score(
            yaw,
            level_R,
            lifted_center,
            destination,
            load_offset,
            camera_p_ee,
            camera_R_ee,
        )
        desired_R = Rotation.from_rotvec([0.0, 0.0, yaw]).as_matrix() @ level_R
        requested = []
        current_xyzw = Rotation.from_matrix(anchor_R).as_quat()
        for alpha in np.linspace(0.0, 1.0, vertical_count):
            center = carried_center + np.asarray([0.0, 0.0, float(lift) * alpha])
            xyz = center - anchor_R @ load_offset
            requested.append(np.r_[current_xyzw[[3, 0, 1, 2]], xyz])
        rotations = Slerp(
            [0.0, 1.0],
            Rotation.concatenate(
                [Rotation.from_matrix(anchor_R), Rotation.from_matrix(desired_R)]
            ),
        )(np.linspace(0.0, 1.0, count))[1:]
        for rotation in rotations:
            matrix = rotation.as_matrix()
            xyzw = rotation.as_quat()
            xyz = lifted_center - matrix @ load_offset
            requested.append(np.r_[xyzw[[3, 0, 1, 2]], xyz])
        requested = np.asarray(requested)
        try:
            rotation_q, ik = kinematics.solve_path(
                requested,
                seed_q=anchor_q,
                level_reference=reference,
                allow_multistart=False,
                maximum_position_error_m=0.006,
                maximum_rotation_error_rad=math.radians(4.0),
                maximum_joint_delta_rad=0.20,
            )
            q_path = np.vstack((prefix_q[:-1], rotation_q))
            poses = np.asarray([kinematics.pose(q) for q in q_path])
            audit = audit_joint_path(
                _resolve(config["planning_model"]),
                "right",
                q_path,
                dish_radius_m=float(geometry["dish_diameter_m"]) / 2.0,
                dish_thickness_m=float(geometry["dish_thickness_m"]),
                dish_center_offset_ee_m=load_offset,
                ignored_environment_bodies=geometry.get("ignored_absent_scene_bodies", []),
                carrying_sample_range=(0, len(q_path) - 1),
                minimum_dish_clearance_m=float(
                    config["planning"]["minimum_dish_clearance_m"]
                ),
            )
        except (RuntimeError, ValueError) as error:
            trials.append(
                {"lift_m": float(lift), "accepted": False, "reason": repr(error)}
            )
            continue
        trials.append(
            {
                "lift_m": float(lift),
                "accepted": bool(audit["accepted"]),
                "optimized_yaw_deg": math.degrees(yaw),
                "predicted_camera_axis_error_deg": math.degrees(axis_error),
                "first_disallowed_sample": audit.get("first_disallowed_sample"),
                "minimum_dish_environment_distance_m": audit.get(
                    "minimum_dish_environment_distance_m"
                ),
            }
        )
        if audit["accepted"]:
            selected = (float(lift), yaw, axis_error, q_path, poses, ik, audit)
            break
    if selected is None:
        raise RuntimeError(f"no collision-free camera-aim lift found: {trials}")
    selected_lift, yaw, axis_error, q_path, poses, ik, audit = selected

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "piper_robot.carrier_camera_destination_aim/v1",
        "status": "planned",
        "source_plan": str(args.plan.resolve()),
        "destination_physical_xyz_m": destination.tolist(),
        "carried_center_physical_xyz_m": carried_center.tolist(),
        "nearest_source_plan_index": nearest_index,
        "nearest_source_plan_max_joint_error_rad": nearest_error,
        "retreat_source_plan_index": retreat_index,
        "used_current_anchor": bool(args.use_current_anchor),
        "selected_lift_m": selected_lift,
        "optimized_yaw_deg": math.degrees(yaw),
        "predicted_camera_axis_error_deg": math.degrees(axis_error),
        "yaw_source": (
            "empirically_forced" if args.forced_yaw_deg is not None else "model_optimized"
        ),
        "lift_search_trials": trials,
        "ik": ik,
        "collision_audit": audit,
    }
    report_path = output / "run.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not audit["accepted"]:
        raise RuntimeError(f"camera-aim collision audit failed: {audit}")
    if args.plan_only:
        report["status"] = "audited_plan_only"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0

    execution = config["execution"]
    streamer = CartesianAirTransportStreamer(
        rpc,
        "right",
        torque_limit_nm=execution["torque_limits_nm"]["right"],
        torque_samples=int(execution["torque_consecutive_samples"]),
        control_hz=float(execution["control_hz"]),
        preview_time_s=float(execution["preview_time_s"]),
        tracking_interval=int(execution["tracking_check_interval"]),
        maximum_tracking_position_error_m=float(execution["maximum_tracking_position_error_m"]),
        maximum_tracking_rotation_error_rad=float(execution["maximum_tracking_rotation_error_rad"]),
        final_settle_s=float(execution["endpoint_settle_s"]),
    )
    try:
        report["motion"] = streamer.execute(
            poses,
            speed_m_s=float(execution["speed_m_s"]),
            gripper_open_ratio=float(execution["air_gripper_open_ratio"]),
            stage="carrier_camera_aim/microscope",
        )
        report["final_pose_wxyz_xyz"] = streamer.measured_pose().tolist()
        report["images"] = _capture(
            output / "images",
            config["camera_names"],
            float(execution["camera_timeout_s"]),
            float(execution["maximum_camera_timestamp_skew_s"]),
        )
        report["status"] = "held_for_image_verification"
    except BaseException as error:
        streamer.hold_measured()
        report["status"] = "error_hold"
        report["error"] = repr(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
