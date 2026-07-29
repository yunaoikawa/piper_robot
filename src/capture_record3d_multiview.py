#!/usr/bin/env python3
"""Capture manually repositioned, stopped Record3D RGB-D views.

The phone stays connected for the complete capture so every saved pose shares
one Record3D session frame.  The script never sends a robot command.  It asks
the operator to move the head RGB-D camera and press Enter before each burst.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid

import cv2
import numpy as np
from record3d import Record3DStream

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.camera_id import load_camera_map
from rollout.multiview_scene import camera_pose_stability
from src.capture_record3d_bundle import (
    CAMERA_MAP_PATH,
    REPO_ROOT,
    decode_misc,
    file_record,
    git_provenance,
    intrinsics_dict,
    pose_dict,
    robot_state_snapshot,
    utc_iso,
    write_json,
)


def _quality(rgb: np.ndarray, depth: np.ndarray, confidence: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    valid = np.isfinite(depth) & (depth > 0.0)
    return {
        "rgb_mean_gray": float(np.mean(gray)),
        "rgb_near_black_fraction": float(np.mean(gray < 8)),
        "rgb_sharpness_laplacian_var": float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ),
        "depth_valid_fraction": float(np.mean(valid)),
        "depth_median_m": float(np.median(depth[valid])) if np.any(valid) else None,
        "confidence_high_fraction": float(np.mean(confidence == 2)),
    }


def _write_view(
    session_dir: Path,
    view_name: str,
    frames: list[dict],
) -> dict:
    raw_root = session_dir / "raw" / "head" / view_name
    raw_root.mkdir(parents=True, exist_ok=False)
    records = []
    for sequence, bundle in enumerate(frames):
        frame_dir = raw_root / f"{sequence:06d}"
        frame_dir.mkdir()
        rgb_path = frame_dir / "rgb.png"
        depth_path = frame_dir / "depth.npy"
        confidence_path = frame_dir / "confidence.npy"
        meta_path = frame_dir / "meta.json"
        if not cv2.imwrite(
            str(rgb_path),
            cv2.cvtColor(bundle["rgb"], cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError(f"failed to write {rgb_path}")
        np.save(depth_path, bundle["depth"], allow_pickle=False)
        np.save(confidence_path, bundle["confidence"], allow_pickle=False)
        metadata = {
            key: value
            for key, value in bundle.items()
            if key not in {"rgb", "depth", "confidence"}
        }
        write_json(meta_path, metadata)
        record = {
            **metadata,
            "sequence": sequence,
            "files": {
                "rgb_png": file_record(rgb_path, session_dir),
                "depth_npy": file_record(depth_path, session_dir),
                "confidence_npy": file_record(confidence_path, session_dir),
                "meta_json": file_record(meta_path, session_dir),
            },
        }
        records.append(record)

    derived = session_dir / "derived" / view_name
    derived.mkdir(parents=True, exist_ok=False)
    representative = records[len(records) // 2]
    source = session_dir / representative["files"]["rgb_png"]["path"]
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    preview = derived / "rgb_landscape.png"
    if image is None or not cv2.imwrite(
        str(preview), cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    ):
        raise RuntimeError(f"failed to write {preview}")
    index_path = derived / "frames.jsonl"
    index_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    stability = camera_pose_stability(
        [item["camera_pose"] for item in records]
    )
    return {
        "name": view_name,
        "frame_count": len(records),
        "representative_sequence": int(representative["sequence"]),
        "pose_stability": stability,
        "frames_index": file_record(index_path, session_dir),
        "preview": file_record(preview, session_dir),
        "frames": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--view",
        action="append",
        dest="views",
        help="stopped-view name; defaults to center, left, right",
    )
    parser.add_argument("--frames-per-view", type=int, default=7)
    parser.add_argument("--warmup-frames", type=int, default=15)
    parser.add_argument("--view-timeout-s", type=float, default=20.0)
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "data" / "captures" / "pasteur"),
    )
    parser.add_argument(
        "--condition",
        default="manual_head_multiview",
    )
    parser.add_argument(
        "--robot-state",
        action="store_true",
        help="save read-only robot state before and after capture",
    )
    parser.add_argument("--robot-host", default="localhost")
    parser.add_argument("--robot-port", type=int, default=8081)
    args = parser.parse_args()
    views = args.views or ["center", "left", "right"]
    if (
        args.frames_per_view < 3
        or args.warmup_frames < 0
        or len(set(views)) != len(views)
        or any(not item.strip() or "/" in item for item in views)
    ):
        parser.error("views must be unique safe names and frames-per-view >= 3")

    created_ns = time.time_ns()
    session_id = (
        datetime.fromtimestamp(created_ns / 1e9, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        + f"_head_{args.condition}_{uuid.uuid4().hex[:8]}"
    )
    session_dir = (
        Path(args.output_root)
        / datetime.fromtimestamp(created_ns / 1e9, tz=timezone.utc).date().isoformat()
        / session_id
    )
    session_dir.mkdir(parents=True, exist_ok=False)
    calibration_dir = session_dir / "calibration"
    calibration_dir.mkdir()
    if CAMERA_MAP_PATH.exists():
        shutil.copy2(CAMERA_MAP_PATH, calibration_dir / CAMERA_MAP_PATH.name)
    partial_manifest = session_dir / "manifest.partial.json"
    manifest = {
        "schema": "piper_robot.rgbd_multiview_capture/v1",
        "schema_version": 1,
        "session_id": session_id,
        "status": "collecting",
        "purpose": args.condition,
        "created_at_utc": utc_iso(created_ns),
        "camera_label": "head",
        "view_order": views,
        "frames_per_view": args.frames_per_view,
        "pose_frame": "single_record3d_session_local",
        "capture_mode": "manual_camera_reposition_only",
        "commands_sent": False,
        "repository": git_provenance(),
    }
    write_json(partial_manifest, manifest)

    robot_before = (
        robot_state_snapshot(args.robot_host, args.robot_port)
        if args.robot_state
        else None
    )
    devices = Record3DStream.get_connected_devices()
    camera_index = int(load_camera_map().get("head", 0))
    if camera_index >= len(devices):
        raise RuntimeError(
            f"head Record3D camera unavailable at index {camera_index}; "
            f"found {len(devices)} device(s)"
        )
    device = devices[camera_index]
    stream = Record3DStream()
    lock = threading.Lock()
    ready = threading.Event()
    view_done = threading.Event()
    active_view: str | None = None
    active_frames: deque[dict] = deque()
    callback_count = 0
    errors: list[str] = []

    def on_new_frame() -> None:
        nonlocal callback_count, active_view
        with lock:
            callback_count += 1
            if callback_count >= args.warmup_frames:
                ready.set()
            if active_view is None:
                return
            try:
                start_realtime_ns = time.time_ns()
                start_monotonic_ns = time.monotonic_ns()
                rgb = np.array(stream.get_rgb_frame(), copy=True)
                depth = np.array(stream.get_depth_frame(), copy=True)
                confidence = np.array(stream.get_confidence_frame(), copy=True)
                coefficients = stream.get_intrinsic_mat()
                pose = stream.get_camera_pose()
                misc = np.array(stream.get_misc_data(), copy=True)
                active_frames.append(
                    {
                        "view_name": active_view,
                        "host_callback_realtime_ns": start_realtime_ns,
                        "host_callback_monotonic_ns": start_monotonic_ns,
                        "rgb_shape": list(rgb.shape),
                        "depth_shape": list(depth.shape),
                        "confidence_shape": list(confidence.shape),
                        "depth_unit": "m",
                        "intrinsics": intrinsics_dict(coefficients, rgb.shape),
                        "camera_pose": pose_dict(pose),
                        "record3d_misc": decode_misc(misc),
                        "quality": _quality(rgb, depth, confidence),
                        "rgb": rgb,
                        "depth": depth,
                        "confidence": confidence,
                    }
                )
                if len(active_frames) >= args.frames_per_view:
                    active_view = None
                    view_done.set()
            except Exception as exc:
                errors.append(f"callback: {type(exc).__name__}: {exc}")
                active_view = None
                view_done.set()

    stream.on_new_frame = on_new_frame
    stream.on_stream_stopped = lambda: errors.append(
        "Record3D stream stopped unexpectedly"
    )
    stream.connect(device)
    if not ready.wait(timeout=args.view_timeout_s):
        raise RuntimeError("Record3D warmup timed out")
    device_type = int(stream.get_device_type())
    if device_type != 1:
        raise RuntimeError(
            "head camera is not reporting Record3D LiDAR depth "
            f"(device_type={device_type})"
        )

    saved_views = []
    for index, name in enumerate(views, 1):
        print(
            f"\n[{index}/{len(views)}] head cameraを{name!r}位置へ動かし、"
            "完全に静止させてから Enter を押してください。",
            flush=True,
        )
        input()
        with lock:
            active_frames.clear()
            active_view = name
            view_done.clear()
        if not view_done.wait(timeout=args.view_timeout_s):
            with lock:
                active_view = None
            raise RuntimeError(f"{name}: frame burst timed out")
        with lock:
            captured = list(active_frames)
        if errors:
            raise RuntimeError(errors[-1])
        view_record = _write_view(session_dir, name, captured)
        saved_views.append(view_record)
        stability = view_record["pose_stability"]
        print(
            f"{name}: 保存完了 "
            f"(移動 {stability['maximum_translation_m']*1000:.1f} mm, "
            f"回転 {stability['maximum_rotation_deg']:.2f} deg)",
            flush=True,
        )
        if not stability["accepted"]:
            raise RuntimeError(
                f"{name}: camera moved during burst; repeat the capture"
            )

    robot_after = (
        robot_state_snapshot(args.robot_host, args.robot_port)
        if args.robot_state
        else None
    )
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": utc_iso(),
            "device": {
                "index": camera_index,
                "udid": getattr(device, "udid", None),
                "product_id": getattr(device, "product_id", None),
                "record3d_device_type": device_type,
                "record3d_device_type_name": "LiDAR",
            },
            "views": saved_views,
            "robot_state": {
                "before": robot_before,
                "after": robot_after,
                "commands_sent": False,
            },
            "errors": errors,
        }
    )
    write_json(partial_manifest, manifest)
    final_manifest = session_dir / "manifest.json"
    os.replace(partial_manifest, final_manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "session_dir": str(session_dir),
                "manifest": str(final_manifest),
                "views": views,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    exit_code = 2
    try:
        exit_code = main()
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
    finally:
        # Record3D 1.4 may crash during native stream destruction on this host.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
