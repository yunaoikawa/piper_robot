#!/usr/bin/env python3
"""Apply one metric, task-shaped right-wrist correction through teleop RPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.arm.home import physical_to_semantic_model_q_offset
from robot.rpc import RPCClient
from rollout.grasp_window import (
    GraspWindowTemplate,
    detect_light_pad_tool_frame,
)
from rollout.teleop_trajectory_stream import (
    ProductionRightFK,
    TeleopTrajectoryStreamer,
    sample_joint_knots,
)
from rollout.tool_relative_visual_correction import (
    estimate_model_plane_correction,
)
from rollout.torque_safety import torque_stop_enabled_from_config
from src.optimize_lid_grasp_trajectory import GraspKinematics


def _load(path):
    return json.loads(Path(path).resolve().read_text())


def _close_rpc(rpc):
    rpc.socket.close(linger=0)
    rpc.context.term()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--target-diameter-m", type=float, required=True)
    parser.add_argument("--maximum-step-m", type=float, default=0.035)
    parser.add_argument("--duration-s", type=float, default=1.5)
    parser.add_argument(
        "--production-model",
        default="robot/cone-e-description/robot-welded-base-and-lift.mjcf",
    )
    parser.add_argument(
        "--torque-config",
        default="src/configs/pasteur_lid_torque.json",
    )
    parser.add_argument("--rpc-host", default="localhost")
    parser.add_argument("--rpc-port", type=int, default=8081)
    parser.add_argument(
        "--right-q-rad",
        type=float,
        nargs=6,
        help=(
            "recorded physical-right joints for offline correction "
            "estimation; cannot be combined with --execute"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and args.right_q_rad is not None:
        parser.error("--execute requires a fresh RPC state, not --right-q-rad")

    image = cv2.imread(str(Path(args.image).resolve()))
    mask_u8 = cv2.imread(
        str(Path(args.mask).resolve()), cv2.IMREAD_GRAYSCALE
    )
    if image is None or mask_u8 is None:
        raise ValueError("could not read correction image/mask")
    mask = mask_u8 > 0
    selection = _load(args.selection)
    template = GraspWindowTemplate.from_dict(selection["template"])
    tool_frame = detect_light_pad_tool_frame(image)
    correction = estimate_model_plane_correction(
        mask,
        tool_frame,
        template,
        target_diameter_m=args.target_diameter_m,
        maximum_step_m=args.maximum_step_m,
    )

    rpc = None
    try:
        if args.right_q_rad is None:
            rpc = RPCClient(args.rpc_host, args.rpc_port, timeout_ms=10000)
            current_physical = np.asarray(
                rpc.get_right_joint_positions(), dtype=float
            )
        else:
            current_physical = np.asarray(args.right_q_rad, dtype=float)
        offset = physical_to_semantic_model_q_offset("right")
        current_model = current_physical + offset
        kinematics = GraspKinematics(args.physics_model)
        current_position, current_rotation = kinematics.pose(current_model)
        local_xy = np.asarray(
            correction.bounded_model_local_xy_m, dtype=float
        )
        world_delta = (
            current_rotation[:, 0] * local_xy[0]
            + current_rotation[:, 1] * local_xy[1]
        )
        # The image correction is planar; do not use camera perspective error
        # to change the established support height.
        world_delta[2] = 0.0
        target_model, ik = kinematics.solve(
            current_position + world_delta,
            current_rotation,
            current_model,
            maximum_position_error_m=0.003,
            maximum_rotation_error_deg=6.0,
            maximum_joint_delta_rad=0.12,
        )
        if args.execute and not ik["accepted"]:
            raise RuntimeError(f"visual correction IK rejected: {ik}")
        target_physical = target_model - offset
        result = {
            "schema": "piper_robot.right_visual_correction/v1",
            "commands_sent": False,
            "correction": correction.to_dict(),
            "world_delta_model_m": world_delta.tolist(),
            "current_right_q_physical_rad": current_physical.tolist(),
            "target_right_q_physical_rad": target_physical.tolist(),
            "ik": ik,
            "execution": None,
        }
        if args.execute:
            assert rpc is not None
            torque = _load(args.torque_config)
            streamer = TeleopTrajectoryStreamer(
                rpc,
                ProductionRightFK(args.production_model),
                torque_limit_nm=torque["thresholds"]["right"],
                consecutive_torque_samples=int(
                    torque.get("consecutive_samples", 5)
                ),
                enforce_torque_stop=torque_stop_enabled_from_config(torque),
            )
            samples = sample_joint_knots(
                [
                    {
                        "stage": "visual_correction_start",
                        "right_q_physical_rad": current_physical.tolist(),
                        "right_gripper_open_ratio": 1.0,
                        "minimum_duration_s": 0.1,
                    },
                    {
                        "stage": "visual_correction",
                        "right_q_physical_rad": target_physical.tolist(),
                        "right_gripper_open_ratio": 1.0,
                        "minimum_duration_s": args.duration_s,
                    },
                ]
            )
            result["execution"] = streamer.execute(samples)
            result["commands_sent"] = True
    finally:
        if rpc is not None:
            _close_rpc(rpc)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
