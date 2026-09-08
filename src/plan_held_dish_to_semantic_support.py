#!/usr/bin/env python3
"""Plan a level carried-dish continuation to a semantic support surface.

This is intentionally different from demonstration retargeting.  The first
sample is an already stopped/held robot checkpoint, so the virtual dish starts
at the current gripper rather than at a stale demonstration station.  The
destination is derived from a named MuJoCo support body and its top surface.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.arm.home import physical_to_semantic_model_q_offset
from rollout.dish_transport_rehearsal import (
    ProductionArmKinematics,
    audit_joint_path,
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


class _SceneRightKinematics:
    def __init__(self, scene: Path, production_model: Path, tool_offset: np.ndarray):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.ids = np.asarray(
            [self.model.joint(f"right/joint{i}").qposadr[0] for i in range(1, 7)]
        )
        self.site = self.model.site("right/ee")
        self.offset = physical_to_semantic_model_q_offset("right")
        self.tool_offset = np.asarray(tool_offset, dtype=float).reshape(3)
        self.production = ProductionArmKinematics(production_model, "right")

    def evaluate(self, q_physical: np.ndarray):
        self.data.qpos[self.ids] = q_physical + self.offset
        mujoco.mj_forward(self.model, self.data)
        site_rotation = np.asarray(self.data.site_xmat[self.site.id]).reshape(3, 3)
        dish_center = self.data.site_xpos[self.site.id] + site_rotation @ self.tool_offset
        pose = self.production.pose(q_physical)
        physical_rotation = Rotation.from_quat(pose[[1, 2, 3, 0]]).as_matrix()
        return dish_center.copy(), physical_rotation, pose

    def solve_path(
        self,
        centers: np.ndarray,
        start_q: np.ndarray,
        target_rotation: np.ndarray,
        *,
        maximum_joint_step_rad: float,
        maximum_rotation_error_deg: float,
        free_level_yaw: bool,
    ) -> tuple[np.ndarray, dict]:
        path = [np.asarray(start_q, dtype=float).reshape(6)]
        position_errors = []
        rotation_errors = []
        for index, center in enumerate(centers[1:], start=1):
            seed = path[-1]
            lower = np.maximum(
                self.production.lower + 1e-5, seed - maximum_joint_step_rad
            )
            upper = np.minimum(
                self.production.upper - 1e-5, seed + maximum_joint_step_rad
            )
            # The physical controller can report a stopped joint a few
            # hundredths of a radian beyond the conservative CAD range.  Keep
            # that measured state as path sample zero, but seed the optimizer
            # inside its declared bounds.  The solved continuation must move
            # back into (never farther beyond) the production range.
            optimizer_seed = np.clip(seed, lower, upper)

            def residual(q):
                actual_center, actual_rotation, _ = self.evaluate(q)
                rotation_error = (
                    actual_rotation[:, 1] - np.asarray([0.0, 0.0, 1.0])
                    if free_level_yaw
                    else Rotation.from_matrix(
                        target_rotation.T @ actual_rotation
                    ).as_rotvec()
                )
                return np.r_[
                    (actual_center - center) / 0.002,
                    rotation_error / math.radians(1.5),
                    0.005 * (q - seed),
                ]

            result = least_squares(
                residual,
                optimizer_seed,
                bounds=(lower, upper),
                max_nfev=300,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9,
            )
            actual_center, actual_rotation, _ = self.evaluate(result.x)
            position_error = float(np.linalg.norm(actual_center - center))
            rotation_error = (
                math.acos(
                    float(
                        np.clip(actual_rotation[:, 1] @ [0.0, 0.0, 1.0], -1.0, 1.0)
                    )
                )
                if free_level_yaw
                else float(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            target_rotation.T @ actual_rotation
                        ).as_rotvec()
                    )
                )
            )
            if position_error > 0.004 or rotation_error > math.radians(
                maximum_rotation_error_deg
            ):
                raise ValueError(
                    f"scene path IK failed at sample {index}: "
                    f"position={position_error:.6f}m, "
                    f"rotation={math.degrees(rotation_error):.3f}deg"
                )
            path.append(np.asarray(result.x, dtype=float))
            position_errors.append(position_error)
            rotation_errors.append(rotation_error)
        result_path = np.asarray(path)
        return result_path, {
            "accepted": True,
            "maximum_position_error_m": max(position_errors, default=0.0),
            "maximum_rotation_error_deg": math.degrees(
                max(rotation_errors, default=0.0)
            ),
            "maximum_joint_step_rad": float(
                np.max(np.abs(np.diff(result_path, axis=0)))
            ),
        }


def _support_center(model: mujoco.MjModel, body_name: str, dish_height_m: float):
    body = model.body(body_name)
    geom_ids = np.flatnonzero(model.geom_bodyid == body.id)
    if len(geom_ids) != 1:
        raise ValueError(f"{body_name!r} must have exactly one support geom")
    geom_id = int(geom_ids[0])
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError(f"{body_name!r} support geom must be a box")
    center = np.asarray(model.body_pos[body.id], dtype=float).copy()
    center[2] += float(model.geom_pos[geom_id, 2])
    center[2] += float(model.geom_size[geom_id, 2]) + dish_height_m / 2.0
    return center


def _linear(first: np.ndarray, second: np.ndarray, samples: int):
    return np.linspace(first, second, int(samples), endpoint=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-checkpoint", default="midpoint")
    parser.add_argument(
        "--start-q-rad",
        nargs=6,
        type=float,
        help="Use a measured stopped right-arm joint state instead of a source checkpoint.",
    )
    parser.add_argument(
        "--destination-body", default="microscope-1-inferred-stage"
    )
    parser.add_argument("--transit-clearance-m", type=float, default=0.034)
    parser.add_argument(
        "--hover-only",
        action="store_true",
        help="stop at the destination hover instead of descending to contact",
    )
    parser.add_argument("--maximum-joint-step-rad", type=float, default=0.16)
    parser.add_argument("--maximum-rotation-error-deg", type=float, default=3.0)
    parser.add_argument("--vertical-samples", type=int, default=31)
    parser.add_argument("--transport-samples", type=int, default=61)
    parser.add_argument("--placement-samples", type=int, default=21)
    parser.add_argument(
        "--free-level-yaw",
        action="store_true",
        help="keep the jaw plane horizontal while allowing yaw to follow reachability",
    )
    parser.add_argument(
        "--horizontal-first",
        action="store_true",
        help="Translate above the destination before making any final descent.",
    )
    parser.add_argument(
        "--horizontal-fraction",
        type=float,
        default=1.0,
        help=(
            "Execute this fraction of the high horizontal leg [-1, 1], excluding "
            "zero. Negative values make a short Cartesian retreat away from the "
            "destination. Fractions below one omit the destination descent."
        ),
    )
    parser.add_argument(
        "--vertical-fraction",
        type=float,
        default=1.0,
        help=(
            "Execute only this fraction of the initial descent (0, 1]. "
            "Fractions below one deliberately omit the following horizontal leg."
        ),
    )
    args = parser.parse_args()

    source_manifest = json.loads(args.source_plan.read_text())
    source_plan = source_manifest["plans"][0]
    config_path = _resolve(source_manifest["config"])
    config = json.loads(config_path.read_text())
    scene = _resolve(config["planning_model"])
    production = _resolve(config["production_model"])
    geometry = config["geometry"]
    tool_offset = np.asarray(geometry["dish_center_offset_ee_m"], dtype=float)
    dish_height = float(geometry["dish_thickness_m"])

    if args.start_q_rad is None:
        checkpoint = next(
            item
            for item in source_plan["checkpoints"]
            if item["name"] == args.start_checkpoint
        )
        start_index = int(checkpoint["pose_index"])
        start_q = np.asarray(source_plan["q_physical_rad"], dtype=float)[start_index]
        start_source = f"source_checkpoint:{args.start_checkpoint}"
    else:
        start_q = np.asarray(args.start_q_rad, dtype=float)
        start_source = "measured_stopped_q_cli"

    kinematics = _SceneRightKinematics(scene, production, tool_offset)
    start_center, target_rotation, _ = kinematics.evaluate(start_q)
    destination_center = _support_center(
        kinematics.model, args.destination_body, dish_height
    )
    transit_z = destination_center[2] + float(args.transit_clearance_m)
    lower_at_start = start_center.copy()
    lower_at_start[2] = transit_z
    hover_destination = destination_center.copy()
    hover_destination[2] = transit_z
    if args.horizontal_first:
        if not -1.0 <= args.horizontal_fraction <= 1.0 or abs(
            args.horizontal_fraction
        ) < 1e-9:
            raise ValueError(
                "--horizontal-fraction must be in [-1, 1] and non-zero"
            )
        transit_z = max(float(start_center[2]), float(transit_z))
        full_horizontal_destination = destination_center.copy()
        full_horizontal_destination[2] = transit_z
        horizontal_destination = start_center + float(args.horizontal_fraction) * (
            full_horizontal_destination - start_center
        )
        hover_destination[2] = destination_center[2] + float(args.transit_clearance_m)
        pieces = [_linear(start_center, horizontal_destination, args.transport_samples)]
        if args.horizontal_fraction == 1.0:
            pieces.append(
                _linear(horizontal_destination, hover_destination, args.vertical_samples)[1:]
            )
    else:
        if not 0.0 < args.vertical_fraction <= 1.0:
            raise ValueError("--vertical-fraction must be in (0, 1]")
        descent_stop = start_center + float(args.vertical_fraction) * (
            lower_at_start - start_center
        )
        pieces = [_linear(start_center, descent_stop, args.vertical_samples)]
        if args.vertical_fraction == 1.0:
            pieces.append(
                _linear(lower_at_start, hover_destination, args.transport_samples)[1:]
            )
    complete_route = bool(
        args.horizontal_fraction == 1.0 and args.vertical_fraction == 1.0
    )
    if not args.hover_only and complete_route:
        pieces.append(
            _linear(hover_destination, destination_center, args.placement_samples)[1:]
        )
    centers = np.vstack(pieces)
    q_path, ik_report = kinematics.solve_path(
        centers,
        start_q,
        target_rotation,
        maximum_joint_step_rad=float(args.maximum_joint_step_rad),
        maximum_rotation_error_deg=float(args.maximum_rotation_error_deg),
        free_level_yaw=bool(args.free_level_yaw),
    )
    poses = np.asarray([kinematics.production.pose(q) for q in q_path])
    collision = audit_joint_path(
        scene,
        "right",
        q_path,
        dish_radius_m=float(geometry["dish_diameter_m"]) / 2.0,
        dish_thickness_m=dish_height,
        dish_center_offset_ee_m=tool_offset,
        ignored_environment_bodies=geometry.get("ignored_absent_scene_bodies", []),
        carrying_sample_range=(0, len(q_path) - 1),
        minimum_dish_clearance_m=float(
            config.get("planning", {}).get("minimum_dish_clearance_m", 0.01)
        ),
    )
    vertical_stop = args.vertical_samples - 1
    hover_stop = vertical_stop + args.transport_samples - 1
    if args.horizontal_first:
        horizontal_stop = args.transport_samples - 1
        hover_stop = len(q_path) - 1
    else:
        horizontal_stop = vertical_stop
    checkpoint_indices = [
        0,
        horizontal_stop,
        len(q_path) - 1 if args.hover_only else hover_stop,
    ]
    if args.horizontal_first and args.horizontal_fraction < 1.0:
        checkpoint_indices = [0, horizontal_stop]
        checkpoint_names = ["current_hold", "horizontal_inspection_stop"]
    elif not args.horizontal_first and args.vertical_fraction < 1.0:
        checkpoint_indices = [0, vertical_stop]
        checkpoint_names = ["current_hold", "vertical_inspection_stop"]
    else:
        checkpoint_names = [
            "current_hold",
            "above_destination" if args.horizontal_first else "transit_height",
            "stage_hover",
        ]
    if not args.hover_only and complete_route:
        checkpoint_indices.append(len(q_path) - 1)
        checkpoint_names.append("placed")
    collision["scene_anchored_endpoint"] = {
        "start_source": start_source,
        "start_checkpoint": args.start_checkpoint,
        "start_dish_center_scene_xyz_m": start_center.tolist(),
        "destination_body": args.destination_body,
        "destination_dish_center_scene_xyz_m": destination_center.tolist(),
        "destination_constraint": "support_top_center_plus_half_dish_thickness",
        "transit_height_scene_z_m": transit_z,
        "hover_only": bool(args.hover_only),
        "horizontal_first": bool(args.horizontal_first),
        "horizontal_fraction": float(args.horizontal_fraction),
        "vertical_fraction": float(args.vertical_fraction),
        "ik": ik_report,
    }
    manifest = {
        "schema": "piper_robot.scene_anchored_dish_transport_preview/v1",
        "config": str(config_path),
        "source_plan": str(args.source_plan.resolve()),
        "plans": [
            {
                "schema": "piper_robot.dish_transport_air_rehearsal/v2",
                "name": "current_hold_to_microscope_stage",
                "source": "current_stopped_hold",
                "destination": "microscope_stage_top_center",
                "physical_arm": "right",
                "medoid_hdf5": source_plan["medoid_hdf5"],
                "medoid_sha256": source_plan["medoid_sha256"],
                "coordinate_retarget": "semantic_scene_support_anchored",
                "poses_wxyz_xyz": poses.tolist(),
                "q_physical_rad": q_path.tolist(),
                "checkpoints": [
                    {
                        "name": name,
                        "pose_index": index,
                        "pose_wxyz_xyz": poses[index].tolist(),
                    }
                    for name, index in zip(checkpoint_names, checkpoint_indices)
                ],
                "maximum_planned_tilt_deg": ik_report[
                    "maximum_rotation_error_deg"
                ],
                "collision_audit": collision,
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)
    print(json.dumps(collision["scene_anchored_endpoint"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
