#!/usr/bin/env python3
"""Offline audit of agent intervention episodes; never connects to the robot."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def summarize(episode_dir: Path):
    manifest = json.loads((episode_dir / "manifest.json").read_text())
    h5_path = next(episode_dir.glob("*.hdf5"))
    with h5py.File(h5_path, "r") as source:
        bias = source["xyz_bias_left_right"][()]
        revisions = source["intervention_revision"][()]
        commanded = source["commanded_target_quat16"][()]
        measured = source["right_ee_pos"][()]
        timestamps = source["active_timestamps"][()]
    events_path = episode_dir / "interventions.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    nudges = [event for event in events if event.get("event") == "bias_nudge"]
    valid_command = np.isfinite(commanded[:, 11:14]).all(axis=1)
    tracking_error = np.linalg.norm(commanded[valid_command, 11:14] - measured[valid_command], axis=1)
    return {
        "episode": episode_dir.name,
        "task": manifest["task"],
        "outcome": manifest["outcome"],
        "training_eligible": manifest["training_eligible"],
        "samples": int(len(timestamps)),
        "duration_active_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "corrections": len(nudges),
        "maximum_revision": int(np.max(revisions)) if len(revisions) else 0,
        "final_right_bias_m": bias[-1, 3:6].tolist() if len(bias) else None,
        "tracking_error_median_m": float(np.median(tracking_error)) if len(tracking_error) else None,
        "tracking_error_p95_m": float(np.quantile(tracking_error, .95)) if len(tracking_error) else None,
        "dropped_samples": manifest.get("dropped_samples", 0),
        "deadline_misses": manifest.get("deadline_misses", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dirs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    reports = [summarize(path) for path in args.episode_dirs]
    if args.json:
        print(json.dumps(reports, indent=2))
        return
    for report in reports:
        print(
            f"{report['episode']}: {report['outcome']} samples={report['samples']} "
            f"nudges={report['corrections']} final_bias={report['final_right_bias_m']} "
            f"p95_tracking={report['tracking_error_p95_m']}"
        )


if __name__ == "__main__":
    main()
