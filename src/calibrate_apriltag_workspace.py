#!/usr/bin/env python3
"""Discover AprilTags and build/update a planar replay tag profile."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from record3d import Record3DStream
from robot.camera_id import load_camera_map
from rollout.apriltag_retarget import (
    classify_roles, detect_tags, estimate_tag_camera_pose, render_tags)


def capture_camera(label="head", timeout=10.0):
    devices = Record3DStream.get_connected_devices()
    index = load_camera_map().get(label, 0)
    if index >= len(devices):
        raise RuntimeError(f"{label} camera index {index} unavailable; found {len(devices)} devices")
    stream = Record3DStream()
    settled = threading.Event()
    frame_holder = [None]
    depth_holder = [None]
    best_sharpness = [-1.0]
    frame_count = 0
    def mark_ready():
        nonlocal frame_count
        try:
            candidate = np.array(stream.get_rgb_frame(), copy=True)
            frame_count += 1
            if frame_count >= 5:
                gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
                sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sharpness > best_sharpness[0]:
                    best_sharpness[0] = sharpness
                    frame_holder[0] = candidate
                    depth_holder[0] = np.array(stream.get_depth_frame(), copy=True)
            if frame_count >= 20:
                settled.set()
        except Exception:
            pass
    stream.on_new_frame = mark_ready
    stream.on_stream_stopped = lambda: None
    stream.connect(devices[index])
    # The first Record3D frames are often blurred or underexposed. Detection
    # should use a settled frame, particularly for the 30 mm lid tag.
    settled.wait(timeout=timeout)
    if frame_holder[0] is None:
        raise RuntimeError(f"timed out waiting for {label} camera")
    rgb = frame_holder[0]
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
    depth = (None if depth_holder[0] is None
             else np.rot90(depth_holder[0], k=3))
    return image, intrinsics, camera_matrix, depth


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--image")
    source.add_argument("--capture-head", action="store_true")
    source.add_argument("--capture-camera", choices=("head", "right", "left"))
    ap.add_argument("--output", required=True, help="profile JSON to create/update")
    ap.add_argument("--annotated", default="/tmp/apriltag_discovery.png")
    ap.add_argument("--raw", help="unmodified capture (default: ANNOTATED_raw.png)")
    ap.add_argument("--depth", help="rotated float depth NPY (default: RAW_depth.npy)")
    ap.add_argument("--lid-id", type=int, help="override automatic 30mm lid-tag classification")
    ap.add_argument("--fixed", nargs=3, action="append", metavar=("ID", "ROBOT_X", "ROBOT_Y"),
                    help="register a 60mm fixed tag center; repeat >=3 times")
    ap.add_argument("--reference-lid", nargs=3, type=float, metavar=("X", "Y", "YAW_RAD"))
    ap.add_argument("--reference-wrist-corners", nargs=8, type=float)
    args = ap.parse_args()

    path = Path(args.output)
    cfg = json.loads(path.read_text()) if path.exists() else {}
    # Once the operator has confirmed the lid ID, never reclassify it from
    # apparent image size: a distant 60 mm workspace tag can look smaller than
    # the 30 mm lid tag. An explicit CLI override still takes precedence.
    configured_lid_id = cfg.get("lid_id")
    lid_id_hint = (args.lid_id if args.lid_id is not None
                   else int(configured_lid_id) if configured_lid_id is not None
                   else None)

    capturing = args.capture_head or args.capture_camera is not None
    if capturing and os.environ.get("PASTEUR_TAG_CAPTURE_WORKER") != "1":
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
        depth = None
    else:
        camera_label = args.capture_camera or "head"
        image, intrinsics, camera_matrix, depth = capture_camera(camera_label)
    raw_path = (Path(args.raw) if args.raw else
                Path(args.annotated).with_name(Path(args.annotated).stem + "_raw.png"))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(raw_path), image)
    depth_path = None
    depth_preview = None
    depth_stats = None
    if depth is not None and depth.size:
        depth_path = (Path(args.depth) if args.depth else
                      raw_path.with_name(raw_path.stem + "_depth.npy"))
        np.save(depth_path, depth)
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size:
            lo, hi = np.percentile(valid, [2, 98])
            normalized = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
            preview = cv2.applyColorMap(
                np.uint8(255 * (1.0 - normalized)), cv2.COLORMAP_TURBO)
            preview[~np.isfinite(depth) | (depth <= 0)] = 0
            depth_preview = str(depth_path.with_suffix(".png"))
            cv2.imwrite(depth_preview, preview)
            depth_stats = {"valid_fraction": float(valid.size / depth.size),
                           "min": float(valid.min()), "median": float(np.median(valid)),
                           "max": float(valid.max())}
    detections = detect_tags(image)
    detected_ids = {tag.tag_id for tag in detections}
    if lid_id_hint is None or lid_id_hint in detected_ids:
        roles = classify_roles(detections, lid_id_hint)
        lid_id = next(tag_id for tag_id, role in roles.items() if role == "lid")
    else:
        # Active-view calibration may expose the fixed workspace tags while the
        # arm temporarily occludes the lid. Preserve already confirmed roles and
        # infer only similarly sized new fixed tags; do not relabel a new lid.
        saved_roles = {int(key): value
                       for key, value in cfg.get("discovered_ids", {}).items()}
        fixed_perimeters = [tag.perimeter for tag in detections
                            if saved_roles.get(tag.tag_id) == "fixed"]
        fixed_scale = float(np.median(fixed_perimeters)) if fixed_perimeters else None
        roles = {}
        for tag in detections:
            if tag.tag_id in saved_roles:
                roles[tag.tag_id] = saved_roles[tag.tag_id]
            elif fixed_scale is not None and tag.perimeter >= 0.70 * fixed_scale:
                roles[tag.tag_id] = "fixed"
            else:
                roles[tag.tag_id] = "ignored"
        lid_id = int(lid_id_hint)
    annotated = render_tags(image, detections, roles)
    cv2.imwrite(args.annotated, annotated)

    cfg.update({
        "version": 1,
        "family": detections[0].family,
        "lid_id": lid_id,
        "tag_sizes_m": {"lid": 0.03, "fixed": 0.06},
        "discovered_ids": {
            **cfg.get("discovered_ids", {}),
            **{str(tag_id): role for tag_id, role in roles.items()},
        },
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
        cfg[f"{camera_label}_tag_pnp"] = pnp
    if intrinsics is not None:
        camera_label = args.capture_camera or "head"
        cfg[f"{camera_label}_intrinsics"] = {
            key: float(getattr(intrinsics, key))
            for key in ("fx", "fy", "tx", "ty") if hasattr(intrinsics, key)
        }
        cfg[f"{camera_label}_camera_matrix_rotated"] = camera_matrix.tolist()
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
    lid_visible = lid_id in detected_ids
    print(json.dumps({"profile": str(path), "raw": str(raw_path),
                      "depth": None if depth_path is None else str(depth_path),
                      "depth_preview": depth_preview, "depth_stats": depth_stats,
                      "annotated": args.annotated,
                      "family": detections[0].family, "roles": roles,
                      "fixed_visible": fixed_count,
                      "lid_visible": lid_visible,
                      "ready_for_planar_fit": fixed_count >= 3}, indent=2))
    if capturing:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
