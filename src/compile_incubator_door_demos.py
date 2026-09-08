#!/usr/bin/env python3
"""Compile visually verified incubator-door successes into one reference."""

from __future__ import annotations

import argparse
import cv2
import h5py
import json
import mink
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.incubator_door_demo import compile_demonstrations
from rollout.incubator_door_visual import extract_feature, fit_ridge


VERIFIED_SUCCESS_STEMS = (
    "door_open_20260703_164850",
    "door_open_20260703_163756",
    "door_open_20260703_173136",
    "door_open_20260703_165810",
    "door_open_20260703_170229",
    "door_open_20260703_175546",
    "door_open_20260703_180246",
    "door_open_20260703_163437",
    "door_open_20260703_162442",
    "door_open_20260708_151315",
    "door_open_20260703_164136",
    "door_open_20260703_175931",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("src/configs/pasteur_incubator_door_demo.json"),
    )
    args = parser.parse_args()
    paths = [args.input_dir / f"{stem}.hdf5" for stem in VERIFIED_SUCCESS_STEMS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"verified door recordings are missing: {missing}")
    result = compile_demonstrations(paths)
    profile = json.loads(args.profile.read_text())
    settings = profile["visual_feature"]
    feature_rows = []
    local_targets = []
    groups = []
    goals = []
    for group, path in enumerate(paths):
        with h5py.File(path, "r") as recording:
            positions = np.asarray(recording["right_ee_pos"][:], dtype=float)
            quaternions = np.asarray(recording["right_ee_quat"][:], dtype=float)
            gripper = np.asarray(recording["right_gripper"][:], dtype=float)
        close = int(np.flatnonzero(gripper < 0.8)[0])
        capture = cv2.VideoCapture(str(path.with_name(f"{path.stem}_right.mp4")))
        frames = []
        for _ in range(close + 1):
            ok, image = capture.read()
            if not ok:
                break
            frames.append(image)
        capture.release()
        if len(frames) <= close:
            raise RuntimeError(f"right video ended before close frame: {path}")
        goal, _ = extract_feature(frames[close], settings)
        goals.append(goal)
        rotation = mink.SE3(
            np.r_[quaternions[close], positions[close]]
        ).as_matrix()[:3, :3]
        first = max(0, close - int(settings["training_window_frames"]))
        for frame in range(first, close + 1):
            feature, _ = extract_feature(frames[frame], settings)
            feature_rows.append(np.r_[goal - feature, 1.0])
            local_targets.append(rotation.T @ (positions[close] - positions[frame]))
            groups.append(group)
    feature_rows = np.asarray(feature_rows, dtype=float)
    local_targets = np.asarray(local_targets, dtype=float)
    groups = np.asarray(groups, dtype=int)
    errors = []
    for group in np.unique(groups):
        fit = groups != group
        held_out = groups == group
        coefficients = fit_ridge(
            feature_rows[fit], local_targets[fit], ridge=settings["ridge"]
        )
        errors.extend(
            np.linalg.norm(
                feature_rows[held_out] @ coefficients - local_targets[held_out],
                axis=1,
            )
        )
    coefficients = fit_ridge(
        feature_rows, local_targets, ridge=settings["ridge"]
    )
    result["visual_servo"] = {
        "feature": settings,
        "goal_feature_mean": np.mean(goals, axis=0).tolist(),
        "goal_feature_std": np.std(goals, axis=0).tolist(),
        "coefficients": coefficients.tolist(),
        "training_sample_count": int(len(feature_rows)),
        "held_out_position_error_m": {
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
            "maximum": float(np.max(errors)),
        },
        "execution_policy": "lateral_only_then_fresh_observation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
