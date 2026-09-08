#!/usr/bin/env python3
"""Extract a reusable demo-relative grasp template from a recorded episode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.visual_servo import ManipulationTemplate


def longest_closed_run(gripper, threshold=0.5):
    closed = np.asarray(gripper, dtype=float).reshape(-1) < threshold
    best = None
    start = None
    for index, value in enumerate(np.r_[closed, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (index - start, -start, start)
            if best is None or candidate > best:
                best = candidate
            start = None
    if best is None:
        raise ValueError("demo has no closed-gripper run")
    return {"frame": int(best[2]), "length": int(best[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo")
    ap.add_argument("--output", required=True)
    ap.add_argument("--arm", choices=("left", "right"), default="right")
    ap.add_argument(
        "--reference-object-position", type=float, nargs=3, required=True,
        metavar=("X", "Y", "Z"),
    )
    ap.add_argument("--goal-frame", type=int)
    ap.add_argument("--pregrasp-offset", type=float, nargs=3, default=[0, 0, 0.010])
    ap.add_argument("--fine-feature-goal", type=float, nargs=2)
    ap.add_argument("--active-view-ee-pose", type=float, nargs=7)
    ap.add_argument("--empty-close-ratio", type=float, default=0.01)
    args = ap.parse_args()

    with h5py.File(args.demo, "r") as episode:
        pos = np.asarray(episode[f"{args.arm}_ee_pos"], dtype=float)
        quat = np.asarray(episode[f"{args.arm}_ee_quat"], dtype=float)
        grip = np.asarray(episode[f"{args.arm}_gripper"], dtype=float)
    run = longest_closed_run(grip)
    frame = run["frame"] if args.goal_frame is None else args.goal_frame
    if not 0 <= frame < len(pos):
        raise SystemExit(f"goal frame {frame} outside 0..{len(pos) - 1}")
    template = ManipulationTemplate(
        reference_object_position_m=args.reference_object_position,
        goal_ee_pose=np.r_[quat[frame], pos[frame]],
        pregrasp_offset_m=args.pregrasp_offset,
        tracked_translation_axes=[True, True, False],
        fine_feature_goal=args.fine_feature_goal,
        empty_close_ratio=args.empty_close_ratio,
        active_view_ee_pose=args.active_view_ee_pose,
    )
    template.save(args.output)
    print(
        f"saved {args.output}: arm={args.arm} frame={frame} "
        f"longest_closed_run={run}"
    )


if __name__ == "__main__":
    main()
