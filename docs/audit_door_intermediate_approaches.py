#!/usr/bin/env python3
"""Audit intermediate Door observations; do not promote probes to configurations."""
import json
from pathlib import Path

from evaluate_door_approach_distance import (
    ROOT, MODEL, REFERENCE, ProductionRightFK, checked_observation, pose_difference,
)
from evaluate_door_first_approach import COMPILED, measure_motion, read_json, source


def evaluate():
    fk = ProductionRightFK(ROOT / MODEL)
    goal, reference_audit = checked_observation(REFERENCE, fk)
    model = read_json(COMPILED)["visual_servo"]
    rows = []
    # All preserved matching preclose/orientation/visual-correction records,
    # not a selection based on whether the metric improves.
    for path in sorted((ROOT / "data/runs/pasteur").glob("incubator_door_*/*.json")):
        if not any(s in path.parent.name for s in (
            "retry3_preclose", "retry4_preclose", "retry5_restore", "retry6",
        )):
            continue
        if not any(s in path.name for s in ("preclose", "orientation", "visual_alignment")):
            continue
        motion = json.loads(path.read_text())
        if not isinstance(motion.get("after"), dict) or "right_ee_wxyz_xyz" not in motion["after"]:
            continue
        directory = str((path.parent / "after").relative_to(ROOT))
        observation, audit = checked_observation(directory, fk)
        if observation["timestamp_s"] != motion["after"]["timestamp_s"]:
            raise ValueError(f"Motion/frame timestamp mismatch: {path}")
        role = {
            "demo_preclose.json": "repeated demonstrated preclose",
            "restore_aligned_preclose.json": "restoration of previously adjusted pose",
            "orientation_probe.json": "within-development orientation probe",
            "aligned_yaw_preclose.json": "within-development yaw-aligned approach",
            "visual_alignment.json": "within-run visual correction",
        }[path.name]
        row = {
            "motion": source(str(path.relative_to(ROOT))), "role": role,
            "timestamp_s": observation["timestamp_s"],
            **pose_difference(observation["right_ee_wxyz_xyz"], goal["right_ee_wxyz_xyz"]),
            **audit,
            "new_agent_configuration_established": False,
        }
        try:
            measured = measure_motion(str(path.relative_to(ROOT)), model["feature"], model["goal_feature_mean"])
            row["image_measurement"] = measured
        except RuntimeError as exc:
            # An image-detector failure must not discard an available EE pose.
            row["image_measurement"] = {"status": "unavailable", "reason": str(exc)}
        rows.append(row)
    return {
        "definition": "Intermediate observations, not additional validated first-approach configurations.",
        "distance_assumption": "Fixed closed door and robot base; EE-origin separation, not jaw-to-handle distance.",
        "reference": reference_audit,
        "observations": sorted(rows, key=lambda row: row["timestamp_s"]),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
