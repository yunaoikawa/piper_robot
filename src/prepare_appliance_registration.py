#!/usr/bin/env python3
"""Prepare a bounded reference-to-live appliance registration for motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.appliance_frame import (
    appliance_pose_from_local_tag,
    matrix4,
    registration_gate,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _accepted_enrollment(payload: dict, label: str) -> None:
    if not bool(payload.get("accepted", False)) or not bool(
        payload.get("motion_authority", False)
    ):
        raise ValueError(f"{label} enrollment is not motion-authoritative")


def _size(payload: dict) -> np.ndarray | None:
    value = payload.get("evidence", {}).get("size_xyz_m")
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    return result if result.shape == (3,) and np.all(result > 0.0) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-enrollment", type=Path, required=True)
    parser.add_argument("--current-enrollment", type=Path, required=True)
    parser.add_argument("--current-tag-observation", type=Path)
    parser.add_argument("--maximum-translation-m", type=float, default=0.35)
    parser.add_argument("--maximum-yaw-deg", type=float, default=45.0)
    parser.add_argument("--maximum-tilt-deg", type=float, default=8.0)
    parser.add_argument("--maximum-relative-size-error", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = _load(args.reference_enrollment)
    current = _load(args.current_enrollment)
    _accepted_enrollment(reference, "reference")
    _accepted_enrollment(current, "current")
    if reference.get("appliance_semantic_name") != current.get(
        "appliance_semantic_name"
    ):
        raise ValueError("reference and current appliance semantics differ")
    T_reference = matrix4(
        reference["T_robot_appliance_at_enrollment"],
        "reference T_robot_appliance",
    )
    pose_source = "fresh_sam_rgbd_enrollment"
    if args.current_tag_observation:
        if current.get("local_tag") is None:
            raise ValueError("current lab has no local tag enrollment")
        observation = _load(args.current_tag_observation)
        observed_id = int(observation["tag_id"])
        expected_id = int(current["local_tag"]["id"])
        if observed_id != expected_id:
            raise ValueError(
                f"observed local tag {observed_id} does not match {expected_id}"
            )
        T_current = appliance_pose_from_local_tag(
            observation["T_robot_tag"], current
        )
        pose_source = "lab_local_tag_tracking"
    else:
        T_current = matrix4(
            current["T_robot_appliance_at_enrollment"],
            "current T_robot_appliance",
        )

    gate = registration_gate(
        T_reference,
        T_current,
        maximum_translation_m=args.maximum_translation_m,
        maximum_yaw_deg=args.maximum_yaw_deg,
        maximum_tilt_deg=args.maximum_tilt_deg,
    )
    reference_size = _size(reference)
    current_size = _size(current)
    relative_size_error = None
    size_accepted = True
    if reference_size is not None and current_size is not None:
        relative_size_error = float(
            np.max(np.abs(current_size - reference_size) / reference_size)
        )
        size_accepted = bool(
            relative_size_error <= float(args.maximum_relative_size_error)
        )
    gate.update(
        {
            "schema": "piper_robot.appliance_registration/v1",
            "accepted": bool(gate["accepted"] and size_accepted),
            "pose_source": pose_source,
            "appliance_semantic_name": reference["appliance_semantic_name"],
            "relative_size_error": relative_size_error,
            "maximum_relative_size_error": float(
                args.maximum_relative_size_error
            ),
            "size_accepted": size_accepted,
            "reference_enrollment": str(args.reference_enrollment.resolve()),
            "current_enrollment": str(args.current_enrollment.resolve()),
            "current_tag_observation": (
                None
                if args.current_tag_observation is None
                else str(args.current_tag_observation.resolve())
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))
    return 0 if gate["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
