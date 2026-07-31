#!/usr/bin/env python3
"""Tap-select and prepare the minimal SAM-free right-arm lid grasp.

This command intentionally performs no motion yet.  It produces an immutable,
auditable target plan.  Hardware execution remains unavailable until both
clean endpoint images are operator-confirmed and the static MuJoCo audit is
attached to that exact plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.endpoint_interpolation import EndpointCalibration
from robot.camera_id import load_camera_map
from rollout.camera import USBWristCameraFeedManager
from rollout.tapped_lid_target import (
    associate_blue_component,
    register_fixed_head,
    validate_tap_frame,
)
from src.fast_lid_grasp_ui import TapSelectionStore, make_server


def capture_head(timeout_s: float):
    """Capture one fresh head RGB frame without importing a SAM module."""
    stop = threading.Event()
    camera = USBWristCameraFeedManager(
        stop,
        device_index=load_camera_map().get("head", 0),
        label="fast lid fixed head observer",
    )
    camera.start()
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            frame, timestamp, _ = camera.get_latest_frame()
            if frame is not None and frame.size and float(np.percentile(frame, 99)) > 5:
                return cv2.cvtColor(np.rot90(frame, k=3), cv2.COLOR_RGB2BGR), float(timestamp)
            time.sleep(0.05)
    finally:
        stop.set()
        camera.stop()
    raise RuntimeError("head camera did not produce a usable frame")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/configs/pasteur_lid_endpoints.json")
    parser.add_argument("--head-image", help="offline image; otherwise capture live head")
    parser.add_argument("--head-timestamp", type=float)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--output-dir", default="data/runs/pasteur/fast_lid_grasp_latest")
    parser.add_argument("--calibrate-endpoint", choices=("left", "right"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        raise SystemExit(
            "hardware motion is fail-closed: finish clean left/right endpoint taps "
            "and attach an accepted static MuJoCo audit first"
        )

    if args.head_image:
        image = cv2.imread(args.head_image)
        if image is None:
            raise SystemExit(f"cannot read {args.head_image}")
        timestamp = float(args.head_timestamp if args.head_timestamp is not None else time.time())
    else:
        image, timestamp = capture_head(10.0)
    store = TapSelectionStore(image, timestamp)
    server = make_server(store, args.host, args.port)
    server.timeout = 0.25
    print(json.dumps({"url": f"http://100.127.18.64:{args.port}/", "frame_sha256": store.frame_hash}), flush=True)
    deadline = time.monotonic() + args.timeout_s
    while not store.event.is_set() and time.monotonic() < deadline:
        server.handle_request()
    server.server_close()
    if store.selection is None:
        raise SystemExit("tap selection timed out; no motion was sent")

    validate_tap_frame(
        image,
        store.selection,
        frame_timestamp=timestamp,
        now=timestamp,
        maximum_age_s=max(2.0, args.timeout_s + 1.0),
    )
    target = associate_blue_component(image, store.selection.uv)
    calibration = EndpointCalibration.load(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registration = None
    reference_path = json.loads(Path(args.config).read_text()).get(
        "head_reference_image"
    )
    if reference_path:
        reference = cv2.imread(reference_path)
        if reference is None:
            raise SystemExit(f"cannot read fixed-head reference {reference_path}")
        registration = register_fixed_head(reference, image)
        if not registration.accepted:
            raise SystemExit(
                "head camera no longer matches endpoint calibration: "
                f"matches={registration.matches}, "
                f"inliers={registration.inlier_fraction:.1%}, "
                "median_residual="
                f"{registration.median_residual_diagonal_fraction:.4f} diagonal"
            )

    if args.calibrate_endpoint:
        side = args.calibrate_endpoint
        sample = getattr(calibration, side)
        value = json.loads(Path(args.config).read_text())
        endpoint_image = (output_dir / f"head_{side}.png").resolve()
        if not cv2.imwrite(str(endpoint_image), image):
            raise SystemExit(f"failed to write {endpoint_image}")
        value["version"] = 2
        value["image_shape_hw"] = list(image.shape[:2])
        value["max_cross_track_fraction"] = calibration.max_cross_track_fraction
        value["endpoints"][side]["feature_px"] = list(target.center_px)
        value["endpoints"][side]["feature_status"] = "confirmed"
        value["endpoints"][side]["feature_timestamp"] = timestamp
        value.setdefault("head_reference_image", str(endpoint_image))
        # The already measured EE pose is deliberately retained.
        assert value["endpoints"][side]["pregrasp_pose_wxyz_xyz"] == sample.pregrasp_pose.tolist()
        Path(args.config).write_text(json.dumps(value, indent=2) + "\n")
        result = {"mode": "endpoint_calibration", "side": side, "commands_sent": False}
    else:
        result_interp = calibration.interpolate(target.center_px, image_shape_hw=image.shape[:2], reject_outside=True)
        result = {
            "mode": "plan_only",
            "fraction": result_interp.fraction,
            "cross_track_error_px": result_interp.cross_track_error_px,
            "pregrasp_pose_wxyz_xyz": result_interp.target_pose.tolist(),
            "commands_sent": False,
            "next_gate": "static_mujoco_audit",
        }
    result.update({
        "tap": store.selection.to_dict(),
        "marker_center_px": list(target.center_px),
        "marker_center_uv": list(target.center_uv),
        "sam_used": False,
        "left_arm_commands": 0,
        "head_registration": (
            None
            if registration is None
            else {
                "accepted": registration.accepted,
                "matches": registration.matches,
                "inlier_fraction": registration.inlier_fraction,
                "median_residual_diagonal_fraction": registration.median_residual_diagonal_fraction,
            }
        ),
    })
    (output_dir / "head.png").write_bytes(cv2.imencode(".png", image)[1].tobytes())
    (output_dir / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
