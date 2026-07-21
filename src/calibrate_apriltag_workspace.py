#!/usr/bin/env python3
"""Discover AprilTags and build/update a planar replay tag profile."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from record3d import Record3DStream
from robot.camera_id import load_camera_map
from rollout.apriltag_retarget import (
    classify_roles, detect_tags, estimate_tag_camera_pose, render_tags)


def capture_head(timeout=10.0):
    devices = Record3DStream.get_connected_devices()
    index = load_camera_map().get("head", 0)
    if index >= len(devices):
        raise RuntimeError(f"head camera index {index} unavailable; found {len(devices)} devices")
    stream = Record3DStream()
    frame_count = 0
    def mark_ready():
        nonlocal frame_count
        frame_count += 1
    stream.on_new_frame = mark_ready
    stream.connect(devices[index])
    deadline = time.time() + timeout
    # The first Record3D frames are often blurred or underexposed. Detection
    # should use a settled frame, particularly for the 30 mm lid tag.
    while frame_count < 12 and time.time() < deadline:
        time.sleep(0.02)
    if frame_count == 0:
        raise RuntimeError("timed out waiting for head camera")
    rgb = np.asarray(stream.get_rgb_frame())
    intrinsics = stream.get_intrinsic_mat()
    # record3d's native destructor/disconnect can segfault on this host. Keep
    # the session alive until the one-shot process exits (see os._exit below).
    image = cv2.cvtColor(np.rot90(rgb, k=3), cv2.COLOR_RGB2BGR)
    # Clockwise image rotation: u'=raw_h-1-v, v'=u.
    camera_matrix = np.array([
        [intrinsics.fy, 0.0, rgb.shape[0] - 1.0 - intrinsics.ty],
        [0.0, intrinsics.fx, intrinsics.tx],
        [0.0, 0.0, 1.0],
    ])
    return image, intrinsics, camera_matrix


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--capture-head", action="store_true")
    ap.add_argument("--output", required=True, help="profile JSON to create/update")
    ap.add_argument("--annotated", default="/tmp/apriltag_discovery.png")
    ap.add_argument("--lid-id", type=int, help="override automatic 30mm lid-tag classification")
    ap.add_argument("--fixed", nargs=3, action="append", metavar=("ID", "ROBOT_X", "ROBOT_Y"),
                    help="register a 60mm fixed tag center; repeat >=3 times")
    ap.add_argument("--reference-lid", nargs=3, type=float, metavar=("X", "Y", "YAW_RAD"))
    ap.add_argument("--reference-wrist-corners", nargs=8, type=float)
    args = ap.parse_args()

    if args.capture_head and os.environ.get("PASTEUR_TAG_CAPTURE_WORKER") != "1":
        environment = os.environ.copy()
        environment["PASTEUR_TAG_CAPTURE_WORKER"] = "1"
        for attempt in range(1, 4):
            result = subprocess.run([sys.executable, *sys.argv], env=environment)
            if result.returncode == 0:
                return
            print(f"Record3D capture worker failed ({attempt}/3, code={result.returncode}); retrying",
                  file=sys.stderr, flush=True)
            time.sleep(1.0)
        raise SystemExit("head capture failed after 3 isolated attempts")

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise SystemExit(f"cannot read {args.image}")
        intrinsics = None
        camera_matrix = None
    else:
        image, intrinsics, camera_matrix = capture_head()
    detections = detect_tags(image)
    roles = classify_roles(detections, args.lid_id)
    lid_id = next(tag_id for tag_id, role in roles.items() if role == "lid")
    annotated = render_tags(image, detections, roles)
    cv2.imwrite(args.annotated, annotated)

    path = Path(args.output)
    cfg = json.loads(path.read_text()) if path.exists() else {}
    cfg.update({
        "version": 1,
        "family": detections[0].family,
        "lid_id": lid_id,
        "tag_sizes_m": {"lid": 0.03, "fixed": 0.06},
        "discovered_ids": {str(tag_id): role for tag_id, role in roles.items()},
        "phases": {"approach_ramp_end": 60, "pregrasp": 81, "grip": 82,
                   "retarget_hold_end": 140, "retarget_blend_end": 190,
                   "release": 227},
        "max_translation_m": 0.10,
        "max_yaw_deg": 20.0,
    })
    if camera_matrix is not None:
        pnp = {}
        for tag in detections:
            size = 0.03 if roles[tag.tag_id] == "lid" else 0.06
            if roles[tag.tag_id] == "ignored":
                continue
            try:
                rvec, tvec, rms = estimate_tag_camera_pose(
                    tag, camera_matrix, size)
                pnp[str(tag.tag_id)] = {
                    "size_m": size, "rvec": rvec.tolist(),
                    "tvec_m": tvec.tolist(), "reprojection_rms_px": rms,
                }
            except ValueError as exc:
                pnp[str(tag.tag_id)] = {"size_m": size, "error": str(exc)}
        cfg["head_tag_pnp"] = pnp
    if intrinsics is not None:
        cfg["head_intrinsics"] = {
            key: float(getattr(intrinsics, key))
            for key in ("fx", "fy", "tx", "ty") if hasattr(intrinsics, key)
        }
        cfg["head_camera_matrix_rotated"] = camera_matrix.tolist()
    if args.fixed:
        cfg.setdefault("fixed_robot_xy", {})
        for tag_id, x, y in args.fixed:
            cfg["fixed_robot_xy"][str(int(tag_id))] = [float(x), float(y)]
    if args.reference_lid:
        cfg["reference_lid_pose"] = list(args.reference_lid)
    if args.reference_wrist_corners:
        cfg["reference_wrist_corners"] = np.asarray(
            args.reference_wrist_corners).reshape(4, 2).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    fixed_count = sum(role == "fixed" for role in roles.values())
    print(json.dumps({"profile": str(path), "annotated": args.annotated,
                      "family": detections[0].family, "roles": roles,
                      "fixed_visible": fixed_count,
                      "ready_for_planar_fit": fixed_count >= 3}, indent=2))
    if args.capture_head:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
