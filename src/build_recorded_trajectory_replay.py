#!/usr/bin/env python3
"""Build a collision-audited replay from measured RGB-D keyframes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.recorded_trajectory_replay import build_recorded_replay


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--object-scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--display-only-on-collision",
        action="store_true",
        help=(
            "write a contact-audited direct interpolation for visualization "
            "when collision-free planning fails; never authorizes motion"
        ),
    )
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    report = build_recorded_replay(
        model_path=args.model,
        object_scene_path=args.object_scene,
        measured_keyframes=config["measured_keyframes"],
        output_path=args.output,
        keyframe=config.get("home_keyframe", "home"),
        control_hz=float(config.get("control_hz", 30.0)),
        maximum_joint_speed_rad_s=float(
            config.get("maximum_joint_speed_rad_s", 0.35)
        ),
        minimum_segment_duration_s=float(
            config.get("minimum_segment_duration_s", 0.5)
        ),
        physical_right_model_branch=config.get(
            "physical_right_model_branch",
            "right",
        ),
        planning_seed=int(config.get("planning_seed", 22)),
        planning_seed_candidates=config.get("planning_seed_candidates"),
        planning_edge_step_rad=float(
            config.get("planning_edge_step_rad", 0.035)
        ),
        planning_extension_step_rad=float(
            config.get("planning_extension_step_rad", 0.12)
        ),
        planning_maximum_iterations=int(
            config.get("planning_maximum_iterations", 20000)
        ),
        allow_colliding_display_only=args.display_only_on_collision,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
