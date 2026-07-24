#!/usr/bin/env python3
"""Locate a lid between two taught endpoints and compute its right-arm target."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.rpc import RPCClient
from rollout.endpoint_interpolation import EndpointCalibration
from rollout.sam_segmentation import detect_blue_cross_center
from src.calibrate_lid_endpoints import capture_left


def execute_pregrasp(rpc, target_pose, max_step_m=0.025):
    current = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    delta = np.asarray(target_pose[4:7]) - current[4:7]
    steps = max(1, int(np.ceil(np.linalg.norm(delta) / max_step_m)))
    for index in range(1, steps + 1):
        fraction = index / steps
        waypoint = np.asarray(target_pose, dtype=float).copy()
        waypoint[4:7] = current[4:7] + fraction * delta
        rpc.set_right_ee_target(
            mink.SE3(waypoint), gripper_target=1.0, preview_time=0.8
        )
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="src/configs/pasteur_lid_endpoints.json"
    )
    parser.add_argument("--output-dir", default="/tmp/lid_endpoint_run")
    parser.add_argument("--feature-px", nargs=2, type=float)
    parser.add_argument("--reject-outside", action="store_true")
    parser.add_argument("--execute-pregrasp", action="store_true")
    parser.add_argument("--descend-close", action="store_true")
    args = parser.parse_args()

    calibration = EndpointCalibration.load(args.config)
    image, timestamp = capture_left(10.0)
    feature = (
        np.asarray(args.feature_px, dtype=float)
        if args.feature_px is not None
        else detect_blue_cross_center(image)
    )
    if feature is None:
        raise SystemExit("blue cross not detected; holding both arms")
    result = calibration.interpolate(feature, reject_outside=args.reject_outside)

    rpc = RPCClient("localhost", 8081, timeout_ms=10000)
    rpc.init()
    current_observer = np.asarray(rpc.get_left_ee_pose().parameters(), dtype=float)
    observer_shift = float(
        np.linalg.norm(current_observer[4:7] - calibration.observer_pose[4:7])
    )
    if observer_shift > 0.005:
        raise SystemExit(
            f"left observer moved {observer_shift * 1000:.1f}mm since calibration"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    left_px = tuple(np.rint(calibration.left.feature_px).astype(int))
    right_px = tuple(np.rint(calibration.right.feature_px).astype(int))
    current_px = tuple(np.rint(feature).astype(int))
    cv2.line(overlay, left_px, right_px, (255, 0, 255), 3)
    cv2.circle(overlay, left_px, 10, (255, 0, 255), 2)
    cv2.circle(overlay, right_px, 10, (255, 0, 255), 2)
    cv2.drawMarker(
        overlay, current_px, (0, 255, 255), cv2.MARKER_CROSS, 36, 4
    )
    text = (
        f"t={result.fraction:.3f} cross-track="
        f"{result.cross_track_error_px:.1f}px"
    )
    cv2.putText(
        overlay,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        overlay,
        text,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    raw_path = output_dir / "current_raw.png"
    overlay_path = output_dir / "current_overlay.png"
    cv2.imwrite(str(raw_path), image)
    cv2.imwrite(str(overlay_path), overlay)

    report = {
        "timestamp": timestamp,
        "feature_px": np.asarray(feature).round(2).tolist(),
        "fraction": result.fraction,
        "unclamped_fraction": result.unclamped_fraction,
        "cross_track_error_px": result.cross_track_error_px,
        "target_pose": result.target_pose.round(6).tolist(),
        "observer_shift_mm": observer_shift * 1000.0,
        "raw_image": str(raw_path),
        "overlay_image": str(overlay_path),
        "executed": bool(args.execute_pregrasp),
    }
    print(json.dumps(report), flush=True)

    if args.execute_pregrasp:
        execute_pregrasp(rpc, result.target_pose)
    if args.descend_close:
        if not args.execute_pregrasp:
            raise SystemExit("--descend-close requires --execute-pregrasp")
        current = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
        contact = current.copy()
        contact[6] -= calibration.contact_drop_m
        rpc.set_right_ee_target(
            mink.SE3(contact), gripper_target=1.0, preview_time=0.8
        )
        time.sleep(1.0)
        rpc.set_right_ee_target(
            rpc.get_right_ee_pose(), gripper_target=0.0, preview_time=0.2
        )


if __name__ == "__main__":
    main()
