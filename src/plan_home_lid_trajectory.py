#!/usr/bin/env python3
"""Create an observation-only home-to-lid MuJoCo trajectory plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.home_lid_trajectory import (
    load_object_scene,
    plan_home_lid_trajectory,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", required=True)
    parser.add_argument(
        "--model",
        default="robot/pasteur-calibrated-scene/scene.mjcf",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--lift-m", type=float, default=0.040)
    parser.add_argument("--pregrasp-height-m", type=float, default=0.015)
    parser.add_argument("--final-height-m", type=float, default=0.003)
    parser.add_argument("--verification-lift-m", type=float, default=0.005)
    parser.add_argument("--detour-m", type=float, default=0.070)
    parser.add_argument(
        "--maximum-joint-speed-rad-s", type=float, default=0.35
    )
    args = parser.parse_args(argv)
    if min(
        args.lift_m,
        args.pregrasp_height_m,
        args.final_height_m,
        args.verification_lift_m,
        args.detour_m,
    ) < 0:
        parser.error("trajectory distances must be non-negative")
    scene = load_object_scene(args.objects)
    result = plan_home_lid_trajectory(
        scene,
        model_path=args.model,
        lift_m=args.lift_m,
        pregrasp_height_m=args.pregrasp_height_m,
        final_height_m=args.final_height_m,
        verification_lift_m=args.verification_lift_m,
        detour_m=args.detour_m,
        maximum_joint_speed_rad_s=args.maximum_joint_speed_rad_s,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
