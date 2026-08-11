#!/usr/bin/env python3
"""Plan a small, collision-audited Cartesian adjustment from a stopped hold.

The semantic scene remains the collision authority, but the requested delta is
defined in the production robot's metric Cartesian frame.  This avoids using a
scene-Z displacement as a proxy for physical down near a placement surface.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import mink
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rollout.dish_transport_rehearsal import (
    ProductionArmKinematics,
    audit_joint_path,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _unit_horizontal(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=float).reshape(3).copy()
    result[2] = 0.0
    norm = float(np.linalg.norm(result))
    if norm < 1e-9:
        raise ValueError("gripper approach axis has no horizontal component")
    return result / norm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-q-rad", required=True, nargs=6, type=float)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--retreat-inward-m",
        type=float,
        help="move horizontally from the fingertips toward the wrist",
    )
    group.add_argument(
        "--down-m",
        type=float,
        help="move vertically down in the production robot frame",
    )
    group.add_argument(
        "--delta-world-m",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--maximum-joint-step-rad", type=float, default=0.04)
    args = parser.parse_args()

    if args.samples < 2:
        raise ValueError("--samples must be at least two")
    source_manifest = json.loads(args.source_plan.read_text())
    source_plan = source_manifest["plans"][0]
    config_path = _resolve(source_manifest["config"])
    config = json.loads(config_path.read_text())
    geometry = config["geometry"]
    production = ProductionArmKinematics(_resolve(config["production_model"]), "right")
    start_q = np.asarray(args.start_q_rad, dtype=float).reshape(6)
    start_pose = production.pose(start_q)

    if args.retreat_inward_m is not None:
        if args.retreat_inward_m <= 0.0:
            raise ValueError("--retreat-inward-m must be positive")
        rotation = mink.SE3(start_pose).as_matrix()[:3, :3]
        # EE local +X points inward, from the fingertips toward the wrist.
        direction = _unit_horizontal(rotation @ np.asarray([1.0, 0.0, 0.0]))
        delta = float(args.retreat_inward_m) * direction
        adjustment = "horizontal_retreat_toward_wrist"
    elif args.down_m is not None:
        if args.down_m <= 0.0:
            raise ValueError("--down-m must be positive")
        delta = np.asarray([0.0, 0.0, -float(args.down_m)])
        adjustment = "physical_robot_frame_down"
    else:
        delta = np.asarray(args.delta_world_m, dtype=float).reshape(3)
        if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) <= 0.0:
            raise ValueError("--delta-world-m must be finite and non-zero")
        adjustment = "physical_robot_frame_xyz"

    target_pose = start_pose.copy()
    target_pose[4:] += delta
    poses = np.repeat(start_pose.reshape(1, 7), args.samples, axis=0)
    poses[:, 4:] = np.linspace(start_pose[4:], target_pose[4:], args.samples)

    q_path = [start_q.copy()]
    ik_reports = []
    seed = start_q.copy()
    for index, pose in enumerate(poses[1:], start=1):
        seed, report = production.solve_pose(
            pose,
            seed,
            allow_multistart=False,
            maximum_position_error_m=0.001,
            maximum_rotation_error_rad=math.radians(1.0),
            maximum_joint_delta_rad=float(args.maximum_joint_step_rad),
        )
        q_path.append(seed.copy())
        ik_reports.append({"sample_index": index, **report})
    q_path = np.asarray(q_path)
    actual_poses = np.asarray([production.pose(q) for q in q_path])

    collision = audit_joint_path(
        _resolve(config["planning_model"]),
        "right",
        q_path,
        dish_radius_m=float(geometry["dish_diameter_m"]) / 2.0,
        dish_thickness_m=float(geometry["dish_thickness_m"]),
        dish_center_offset_ee_m=geometry["dish_center_offset_ee_m"],
        ignored_environment_bodies=geometry.get("ignored_absent_scene_bodies", []),
        carrying_sample_range=(0, len(q_path) - 1),
        minimum_dish_clearance_m=float(
            config.get("planning", {}).get("minimum_dish_clearance_m", 0.01)
        ),
    )
    if not collision["accepted"]:
        raise RuntimeError(f"local adjustment collision audit failed: {collision}")
    collision["physical_cartesian_adjustment"] = {
        "kind": adjustment,
        "delta_world_xyz_m": delta.tolist(),
        "start_pose_wxyz_xyz": start_pose.tolist(),
        "target_pose_wxyz_xyz": target_pose.tolist(),
        "maximum_joint_step_rad": float(np.max(np.abs(np.diff(q_path, axis=0)))),
        "maximum_ik_position_error_m": max(
            (item["position_error_m"] for item in ik_reports), default=0.0
        ),
        "maximum_ik_rotation_error_deg": max(
            (item["rotation_error_deg"] for item in ik_reports), default=0.0
        ),
    }

    manifest = {
        "schema": "piper_robot.local_cartesian_hold_adjustment/v1",
        "config": str(config_path),
        "source_plan": str(args.source_plan.resolve()),
        "plans": [
            {
                "schema": "piper_robot.dish_transport_air_rehearsal/v2",
                "name": adjustment,
                "source": "measured_stopped_hold",
                "destination": adjustment,
                "physical_arm": "right",
                "medoid_hdf5": source_plan["medoid_hdf5"],
                "medoid_sha256": source_plan["medoid_sha256"],
                "coordinate_retarget": "production_metric_cartesian_delta",
                "poses_wxyz_xyz": actual_poses.tolist(),
                "q_physical_rad": q_path.tolist(),
                "checkpoints": [
                    {
                        "name": "current_hold",
                        "pose_index": 0,
                        "pose_wxyz_xyz": actual_poses[0].tolist(),
                    },
                    {
                        "name": "adjusted_hold",
                        "pose_index": len(q_path) - 1,
                        "pose_wxyz_xyz": actual_poses[-1].tolist(),
                    },
                ],
                "maximum_planned_tilt_deg": 0.0,
                "collision_audit": collision,
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)
    print(json.dumps(collision["physical_cartesian_adjustment"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
