#!/usr/bin/env python3
"""Record one left/right lid endpoint without moving either arm."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.camera_id import load_camera_map
from robot.rpc import RPCClient
from rollout.camera import USBWristCameraFeedManager
from rollout.sam_segmentation import detect_blue_cross_center


def capture_left(timeout_s: float):
    stop = threading.Event()
    camera = USBWristCameraFeedManager(
        stop,
        device_index=load_camera_map().get("left", 1),
        label="left endpoint observer",
    )
    camera.start()
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            frame, timestamp, _ = camera.get_latest_frame()
            if frame is not None and frame.size and float(np.percentile(frame, 99)) > 5:
                return (
                    cv2.cvtColor(np.rot90(frame, k=3), cv2.COLOR_RGB2BGR),
                    float(timestamp),
                )
            time.sleep(0.05)
    finally:
        stop.set()
        camera.stop()
    raise RuntimeError("left camera did not produce a usable frame")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--config", default="src/configs/pasteur_lid_endpoints.json"
    )
    parser.add_argument("--output-dir", default="/tmp/lid_endpoint_calibration")
    parser.add_argument("--feature-px", nargs=2, type=float)
    parser.add_argument("--contact-drop-m", type=float, default=0.010)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    image, timestamp = capture_left(args.timeout_s)
    feature = (
        np.asarray(args.feature_px, dtype=float)
        if args.feature_px is not None
        else detect_blue_cross_center(image)
    )
    if feature is None:
        raise SystemExit("blue cross not detected; use --feature-px after inspecting image")

    rpc = RPCClient("localhost", 8081, timeout_ms=10000)
    rpc.init()
    right_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    observer_pose = np.asarray(rpc.get_left_ee_pose().parameters(), dtype=float)

    config_path = Path(args.config)
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {
            "version": 1,
            "feature_source": "blue_cross",
            "contact_drop_m": args.contact_drop_m,
            "max_cross_track_px": 25.0,
            "endpoints": {},
        }
    config["observer_pose_wxyz_xyz"] = observer_pose.tolist()
    config["contact_drop_m"] = float(args.contact_drop_m)
    config.setdefault("endpoints", {})[args.side] = {
        "feature_px": np.asarray(feature, dtype=float).tolist(),
        "pregrasp_pose_wxyz_xyz": right_pose.tolist(),
        "timestamp": timestamp,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{args.side}_raw.png"
    overlay_path = output_dir / f"{args.side}_overlay.png"
    overlay = image.copy()
    center = tuple(np.rint(feature).astype(int))
    cv2.drawMarker(overlay, center, (0, 255, 255), cv2.MARKER_CROSS, 36, 4)
    cv2.putText(
        overlay,
        f"{args.side.upper()} endpoint {center}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        overlay,
        f"{args.side.upper()} endpoint {center}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    cv2.imwrite(str(raw_path), image)
    cv2.imwrite(str(overlay_path), overlay)
    print(
        json.dumps(
            {
                "side": args.side,
                "feature_px": np.asarray(feature).round(2).tolist(),
                "right_pregrasp_pose": right_pose.round(6).tolist(),
                "observer_pose": observer_pose.round(6).tolist(),
                "raw_image": str(raw_path),
                "overlay_image": str(overlay_path),
                "config": str(config_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
