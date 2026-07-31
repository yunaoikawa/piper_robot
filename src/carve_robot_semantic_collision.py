#!/usr/bin/env python3
"""Remove depth-proven robot contamination from semantic collision voxels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.semantic_collision_carving import carve_robot_contamination
from robot.arm.home import (
    physical_to_semantic_model_q_offset,
    semantic_model_home_q,
)


def _verified_qpos(config_path: str | None, model_branch: str) -> dict:
    if config_path is None:
        return {}
    config = json.loads(Path(config_path).read_text())
    other_branch = "right" if model_branch == "left" else "left"
    result = {}
    for item in config["measured_keyframes"]:
        manifest = json.loads(
            (Path(item["capture"]) / "manifest.json").read_text()
        )
        right = manifest["robot_state"]["after"][
            "right_joint_positions_rad"
        ]
        right = [
            value + offset
            for value, offset in zip(
                right,
                physical_to_semantic_model_q_offset("right"),
            )
        ]
        values = {
            f"{model_branch}/joint{index + 1}": value
            for index, value in enumerate(right)
        }
        values.update(
            {
                f"{other_branch}/joint{index + 1}": value
                for index, value in enumerate(semantic_model_home_q("left"))
            }
        )
        result[item["name"]] = values
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--alignment-report", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--allowed-body-prefix",
        action="append",
        required=True,
    )
    parser.add_argument("--keyframe", default="home")
    parser.add_argument(
        "--robot-body-prefix",
        action="append",
        default=[],
        help="limit contamination carving to this robot branch prefix",
    )
    parser.add_argument(
        "--verified-keyframes-config",
        help="measured replay profile whose stopped poses are collision-free",
    )
    parser.add_argument(
        "--physical-right-model-branch",
        choices=("left", "right"),
        default="right",
    )
    parser.add_argument(
        "--maximum-removed-fraction",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--robot-clearance-margin-m",
        type=float,
        default=0.0,
        help="carve allow-listed voxels within this verified-pose margin",
    )
    args = parser.parse_args(argv)
    report = carve_robot_contamination(
        args.model,
        args.alignment_report,
        args.output_model,
        args.report,
        allowed_body_prefixes=tuple(args.allowed_body_prefix),
        keyframe=args.keyframe,
        maximum_removed_fraction=args.maximum_removed_fraction,
        robot_body_prefixes=tuple(
            args.robot_body_prefix or ["left/", "right/"]
        ),
        verified_qpos_by_name=_verified_qpos(
            args.verified_keyframes_config,
            args.physical_right_model_branch,
        ),
        robot_clearance_margin_m=args.robot_clearance_margin_m,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
