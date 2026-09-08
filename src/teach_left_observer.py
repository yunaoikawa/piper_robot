#!/usr/bin/env python3
"""Capture and optionally move the left wrist camera for observer teaching.

Each invocation performs at most one bounded Cartesian move and writes the
latest image plus pose to an output directory.  It never homes either arm.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.rpc import RPCClient
from rollout.camera import USBWristCameraFeedManager
from robot.camera_id import load_camera_map


def capture_left(stop_event: threading.Event, timeout_s: float):
    camera = USBWristCameraFeedManager(
        stop_event, device_index=load_camera_map().get("left", 1), label="left observer"
    )
    camera.start()
    deadline = time.time() + timeout_s
    frame = None
    timestamp = None
    try:
        while time.time() < deadline:
            frame, timestamp, _ = camera.get_latest_frame()
            if frame is not None and frame.size and float(np.percentile(frame, 99)) > 5:
                return camera, frame, timestamp
            time.sleep(0.05)
    except Exception:
        camera.stop()
        raise
    camera.stop()
    raise RuntimeError("left observer camera did not produce a usable frame")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/left_observer_teaching")
    parser.add_argument("--dx", type=float, default=0.0, help="Cartesian X step in metres")
    parser.add_argument("--dy", type=float, default=0.0, help="Cartesian Y step in metres")
    parser.add_argument("--dz", type=float, default=0.0, help="Cartesian Z step in metres")
    parser.add_argument("--accept", action="store_true", help="save the current pose as observer pose")
    parser.add_argument("--pose-config", default="src/configs/pasteur_left_observer.json")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    delta = np.asarray([args.dx, args.dy, args.dz], dtype=float)
    if np.linalg.norm(delta) > 0.045:
        raise SystemExit("single observer move is limited to 45 mm")

    rpc = RPCClient("localhost", 8081, timeout_ms=10000)
    rpc.init()
    current = rpc.get_left_ee_pose()
    current_params = np.asarray(current.parameters(), dtype=float)
    if np.linalg.norm(delta) > 0:
        target = current_params.copy()
        target[4:7] += delta
        rpc.set_left_ee_target(mink.SE3(target), gripper_target=1.0, preview_time=0.8)
        time.sleep(1.5)

    stop_event = threading.Event()
    camera = None
    try:
        camera, frame, timestamp = capture_left(stop_event, args.timeout_s)
        bgr = cv2.cvtColor(np.rot90(frame, k=3), cv2.COLOR_RGB2BGR)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(output_dir.glob("step_*.png")))
        image_path = output_dir / f"step_{index:03d}.png"
        pose = np.asarray(rpc.get_left_ee_pose().parameters(), dtype=float)
        cv2.imwrite(str(image_path), bgr)
        record = {"image": str(image_path), "timestamp": float(timestamp), "ee_pose": pose.tolist()}
        (output_dir / f"step_{index:03d}.json").write_text(json.dumps(record, indent=2) + "\n")
        if args.accept:
            config_path = Path(args.pose_config)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"ee_pose_wxyz_xyz": pose.tolist()}, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        stop_event.set()
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    main()
