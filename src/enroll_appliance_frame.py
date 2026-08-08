#!/usr/bin/env python3
"""Create a lab-local appliance frame from a motion-ready semantic scene.

This command never contacts or moves the robot.  Its output may be used by a
motion process only when the semantic scene, scene-to-robot calibration, and
registration bounds are all accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.appliance_frame import (
    appliance_pose_from_scene,
    enroll_local_tag,
    load_accepted_robot_scene_transform,
    matrix4,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--robot-scene-calibration", type=Path, required=True)
    parser.add_argument("--semantic-name", default="incubator")
    parser.add_argument("--minimum-confidence", type=float, default=0.85)
    parser.add_argument("--tag-observation", type=Path)
    parser.add_argument("--tag-id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inspection-only",
        action="store_true",
        help="write a rejected diagnostic enrollment from a non-collision-ready scene",
    )
    args = parser.parse_args()

    scene = _load(args.scene)
    readiness = scene.get("readiness", {})
    collision_ready = bool(readiness.get("collision_ready", False))
    if not collision_ready and not args.inspection_only:
        raise SystemExit(
            "semantic scene is not collision-ready; use --inspection-only for diagnostics"
        )
    T_robot_scene, calibration = load_accepted_robot_scene_transform(
        args.robot_scene_calibration
    )
    T_robot_appliance, evidence = appliance_pose_from_scene(
        scene,
        args.semantic_name,
        T_robot_scene,
        minimum_confidence=args.minimum_confidence,
    )
    if (args.tag_observation is None) != (args.tag_id is None):
        parser.error("--tag-observation and --tag-id must be supplied together")
    if args.tag_observation:
        tag_payload = _load(args.tag_observation)
        enrollment = enroll_local_tag(
            T_robot_appliance,
            matrix4(tag_payload.get("T_robot_tag"), "T_robot_tag"),
            tag_id=args.tag_id,
            appliance_semantic_name=args.semantic_name,
        )
    else:
        enrollment = {
            "schema": "piper_robot.appliance_frame_enrollment/v1",
            "accepted": True,
            "appliance_semantic_name": args.semantic_name,
            "local_tag": None,
            "T_robot_appliance_at_enrollment": T_robot_appliance.tolist(),
        }
    enrollment.update(
        {
            "accepted": bool(collision_ready),
            "motion_authority": bool(collision_ready),
            "pose_source": "sam_rgbd_semantic_volume",
            "tag_is_optional": True,
            "tag_placement_is_portable_assumption": False,
            "evidence": evidence,
            "readiness": readiness,
            "provenance": {
                "scene": str(args.scene.resolve()),
                "scene_sha256": _sha256(args.scene),
                "robot_scene_calibration": str(
                    args.robot_scene_calibration.resolve()
                ),
                "robot_scene_calibration_sha256": _sha256(
                    args.robot_scene_calibration
                ),
                "calibration_schema": calibration.get("schema"),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(enrollment, indent=2) + "\n")
    print(json.dumps(enrollment, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
