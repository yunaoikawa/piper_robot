#!/usr/bin/env python3
"""Re-evaluate selected Door preclose images without contacting robot hardware.

Stdout is a reproducible JSON evidence snapshot. This measures a rigid-parent
image-feature proxy, not handle pose, physical distance, or grasp success.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rollout.incubator_door_visual import extract_feature  # noqa: E402

RUNS = "data/runs/pasteur/"
COMPILED = "data/reference/pasteur/incubator/compiled_door_open_v1.json"
AUTO = RUNS + "incubator_auto_open_20260808_demo_retry2/"
CONFIGURATIONS = (
    {"stage": "D0", "label": "Generic replay",
     "missing_reason": "No identified initial preclose for the generic baseline."},
    {"stage": "D1", "label": "Relative demo",
     "motion": RUNS + "incubator_door_20260808T043120Z_demo_preclose/demo_preclose.json",
     "boundary_evidence": "Selected D1 run: absolute demo-preclose after hover, before demo-contact at 04:33:01 UTC. Earlier development probes are not part of this selected-run measure."},
    {"stage": "D2", "label": "Checkpointed pull",
     "motion": RUNS + "incubator_door_20260808T044056Z_retry_preclose_registration/demo_preclose.json",
     "boundary_evidence": "Selected retry-2 run: demo-preclose after retry-hover, before retry2-demo-contact at 04:44:02 UTC. Not an independent reset trial."},
    {"stage": "D3", "label": "Metric alignment",
     "missing_reason": "Yaw probes, restored preclose poses, and manual orientation selection precede the final visual-alignment sequence; a comparable uncorrected first approach is not established.",
     "rejected_surrogate": RUNS + "incubator_door_20260808_retry6_yaw_aligned_visual1/before/right.png"},
    {"stage": "D4", "label": "Autonomous endpoints",
     "motion": AUTO + "03_aligned-yaw-preclose/motion/aligned_yaw_preclose.json",
     "boundary_evidence": "Journal attempt=1, visual-check step=0, after aligned-yaw-preclose and before any visual-align-step. Uses the successful D3 preclose anchor plus live plane-yaw correction."},
    {"stage": "D5", "label": "Evidence hardening",
     "missing_reason": "No new physical opening after evaluator hardening."},
)


def read_json(path):
    return json.loads((ROOT / path).read_text())


def source(path):
    return {"source_path": path,
            "source_sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}


def feature_errors(feature, goal):
    feature, goal = np.asarray(feature, float), np.asarray(goal, float)
    if feature.shape != (3,) or goal.shape != (3,) or not np.isfinite([feature, goal]).all():
        raise ValueError("Expected finite [u/W, v/H, log(area/(W*H))] features")
    return {"uv_error": float(np.linalg.norm(feature[:2] - goal[:2])),
            "absolute_log_area_error": float(abs(feature[2] - goal[2]))}


def measure_motion(path, settings, goal):
    motion = read_json(path)
    frame_dir = str(Path(path).parent / "after")
    image_path = frame_dir + "/right.png"
    observation = read_json(frame_dir + "/observation.json")
    if observation["timestamp_s"] != motion["after"]["timestamp_s"]:
        raise ValueError(f"Motion/frame timestamp mismatch: {path}")
    image = cv2.imread(str(ROOT / image_path))
    if image is None:
        raise ValueError(f"Missing image: {image_path}")
    feature, detection = extract_feature(image, settings)
    return {"status": "measured", **feature_errors(feature, goal),
            "motion_stage": motion["stage"],
            "timestamp_s": observation["timestamp_s"],
            "image_shape_hw": list(image.shape[:2]), "detection": detection,
            "sources": [source(path), source(image_path), source(frame_dir + "/observation.json")]}


def evaluate():
    compiled = read_json(COMPILED)
    model = compiled["visual_servo"]
    settings, goal = model["feature"], model["goal_feature_mean"]
    rows = []
    for spec in CONFIGURATIONS:
        row = dict(spec)
        if "motion" in spec:
            row.update(measure_motion(spec["motion"], settings, goal))
        else:
            row["status"] = "not_identified"
        rows.append(row)
    journal = read_json(AUTO + "journal.json")
    checks = [e for e in journal["events"] if e["event"] == "visual-check" and e["attempt"] == 1]
    if [e["step"] for e in checks] != [0, 1]:
        raise ValueError("Unexpected first-attempt visual check sequence")
    for key, journal_key in (("uv_error", "uv_error"), ("absolute_log_area_error", "log_area_error")):
        if not np.isclose(rows[4][key], checks[0]["report"][journal_key], rtol=0, atol=1e-12):
            raise ValueError("Re-evaluation disagrees with initial journal check")
    corrected = measure_motion(AUTO + "04_visual-align-step/motion/visual_alignment.json", settings, goal)
    if not np.isclose(corrected["uv_error"], checks[1]["report"]["uv_error"], rtol=0, atol=1e-12):
        raise ValueError("Re-evaluation disagrees with corrected journal check")
    return {
        "schema": "door_first_approach_retrospective/v1",
        "definition": "Feature residual at the initial preclose of each selected run, before subsequent image-based correction; not the first move of the entire development history.",
        "evaluator": {
            "position_formula": "sqrt((u/W-goal_u)^2 + (v/H-goal_v)^2)",
            "scale_formula": "abs(log(A/(W*H)) - goal_log_area)",
            "units": "dimensionless; smaller is closer to the demonstration feature",
            "settings": settings, "goal_feature_uv_log_area": goal,
            "goal_feature_std": model["goal_feature_std"],
            "reference_definition": "Mean red-label feature at the close frame of 12 verified successful teleoperation demonstrations, NOT the handle center or a preclose-specific pose.",
            "sources": [source(COMPILED), source("src/compile_incubator_door_demos.py"), source("rollout/incubator_door_visual.py")],
        },
        "limitations": [
            "One selected run per measured configuration; no controlled ablation, common start pose, or uncertainty estimate.",
            "D4 reuses the successful D3 anchor. This measures initialization on the same scene, not generalization to unseen placements.",
            "Label centroid is an indirect parent-feature proxy; translation, rotation, viewpoint, occlusion, and segmentation can all change it.",
            "The goal is a close-frame mean whereas observations precede contact; do not interpret the residual as true grasp-pose error.",
            "Log-area is reported separately, not combined into a weighted performance score or converted to depth.",
            "Initial image detections were visually checked on D1, D2, D4: each box selects the red EYELA label, not the shadow or jaws.",
            "Unknown configurations remain missing; no interpolation or smoothing across them.",
        ],
        "configurations": rows,
        "within_run_correction": {"stage": "D4", "role": "Diagnostic only; NOT another configuration or an initial approach", "journal": source(AUTO + "journal.json"), "after_one_correction": corrected},
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
