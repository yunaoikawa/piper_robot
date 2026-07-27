#!/usr/bin/env python3
"""Capture a lossless, camera-only Record3D bundle.

The capture path never sends a robot command.  When ``--robot-state`` is used,
it only brackets the camera session with read-only RPC state snapshots.
Record3D access is intentionally isolated in this short-lived process because
the native binding can crash while tearing down a stream on this host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from record3d import Record3DStream

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.camera_id import load_camera_map


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERA_MAP_PATH = REPO_ROOT / "robot" / "camera_map.json"


def utc_iso(timestamp_ns: int | None = None) -> str:
    if timestamp_ns is None:
        timestamp_ns = time.time_ns()
    return datetime.fromtimestamp(
        timestamp_ns / 1e9, tz=timezone.utc
    ).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def git_provenance() -> dict[str, Any]:
    def run(*args: str, strip_output: bool = True) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() if strip_output else result.stdout

    # Keep the leading XY status columns.  Stripping the whole output removes
    # the first path's leading space when it is modified only in the worktree.
    status = run("status", "--porcelain=v1", strip_output=False) or ""
    dirty_paths = []
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        dirty_paths.append(path)
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "origin": run("remote", "get-url", "origin"),
        "dirty_paths": dirty_paths,
    }


def intrinsics_dict(coefficients: Any, rgb_shape: tuple[int, ...]) -> dict[str, Any]:
    fx = float(coefficients.fx)
    fy = float(coefficients.fy)
    tx = float(coefficients.tx)
    ty = float(coefficients.ty)
    raw = [[fx, 0.0, tx], [0.0, fy, ty], [0.0, 0.0, 1.0]]
    raw_height = int(rgb_shape[0])
    rotated_clockwise = [
        [fy, 0.0, raw_height - 1.0 - ty],
        [0.0, fx, tx],
        [0.0, 0.0, 1.0],
    ]
    return {
        "coefficients": {"fx": fx, "fy": fy, "tx": tx, "ty": ty},
        "K_raw_rgb": raw,
        "K_rgb_rotated_clockwise": rotated_clockwise,
    }


def pose_dict(pose: Any) -> dict[str, Any]:
    return {
        "translation_xyz_m": [
            float(pose.tx),
            float(pose.ty),
            float(pose.tz),
        ],
        "quaternion_xyzw": [
            float(pose.qx),
            float(pose.qy),
            float(pose.qz),
            float(pose.qw),
        ],
        "frame": "record3d_session_local",
    }


def decode_misc(raw_misc: np.ndarray) -> dict[str, Any]:
    raw_bytes = np.asarray(raw_misc, dtype=np.uint8).tobytes()
    result: dict[str, Any] = {
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "bytes": len(raw_bytes),
    }
    try:
        text = raw_bytes.decode("utf-8")
        result["utf8"] = text
        result["json"] = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        result["hex"] = raw_bytes.hex()
    return result


def file_record(path: Path, session_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(session_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def se3_dict(value: Any) -> dict[str, Any]:
    return {
        "translation_xyz_m": np.asarray(
            value.translation(), dtype=float
        ).tolist(),
        "quaternion_wxyz": np.asarray(
            value.rotation().wxyz, dtype=float
        ).tolist(),
    }


def robot_state_snapshot(host: str, port: int) -> dict[str, Any]:
    """Read robot state without invoking init, home, or any command method."""
    from robot.rpc import RPCClient

    started_realtime_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    client = RPCClient(host, port, timeout_ms=4000)
    result: dict[str, Any] = {
        "host_realtime_ns_start": started_realtime_ns,
        "host_monotonic_ns_start": started_monotonic_ns,
        "source": f"read_only_rpc://{host}:{port}",
    }
    errors: dict[str, str] = {}

    def read(name: str, method: Any, convert: Any) -> None:
        try:
            result[name] = convert(method())
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"

    try:
        read(
            "left_joint_positions_rad",
            client.get_left_joint_positions,
            lambda value: np.asarray(value, dtype=float).tolist(),
        )
        read(
            "right_joint_positions_rad",
            client.get_right_joint_positions,
            lambda value: np.asarray(value, dtype=float).tolist(),
        )
        read(
            "left_joint_torque",
            client.get_left_joint_torque,
            lambda value: np.asarray(value, dtype=float).tolist(),
        )
        read(
            "right_joint_torque",
            client.get_right_joint_torque,
            lambda value: np.asarray(value, dtype=float).tolist(),
        )
        read("left_ee_pose", client.get_left_ee_pose, se3_dict)
        read("right_ee_pose", client.get_right_ee_pose, se3_dict)
        # Some gripper serial links are unavailable while cone_e still provides
        # valid CAN joint state.  Keep these optional and last so they cannot
        # prevent a camera capture.
        read(
            "left_gripper_open_ratio",
            client.get_left_gripper_exact,
            float,
        )
        read(
            "right_gripper_open_ratio",
            client.get_right_gripper_exact,
            float,
        )
    finally:
        result["errors"] = errors
        result["host_realtime_ns_end"] = time.time_ns()
        result["host_monotonic_ns_end"] = time.monotonic_ns()
        client.socket.close(linger=0)
        client.context.term()
    return result


def write_preview(
    session_dir: Path, selected_frame: dict[str, Any]
) -> dict[str, Any]:
    derived = session_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    rgb_file = session_dir / selected_frame["files"]["rgb_png"]["path"]
    depth_file = session_dir / selected_frame["files"]["depth_npy"]["path"]

    rgb_bgr_raw = cv2.imread(str(rgb_file), cv2.IMREAD_COLOR)
    rgb_landscape = cv2.rotate(rgb_bgr_raw, cv2.ROTATE_90_CLOCKWISE)
    rgb_preview = derived / "head_rgb_landscape.png"
    if not cv2.imwrite(str(rgb_preview), rgb_landscape):
        raise RuntimeError(f"failed to write {rgb_preview}")

    depth = np.load(depth_file)
    valid_mask = np.isfinite(depth) & (depth > 0)
    depth_color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(valid_mask):
        lo, hi = np.percentile(depth[valid_mask], [2, 98])
        normalized_valid = np.clip(
            (depth[valid_mask] - lo) / max(float(hi - lo), 1e-6), 0, 1
        )
        depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
        depth_u8[valid_mask] = np.uint8(255 * (1.0 - normalized_valid))
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        depth_color[~valid_mask] = 0
    depth_color = cv2.rotate(depth_color, cv2.ROTATE_90_CLOCKWISE)
    depth_preview = derived / "head_depth_landscape.png"
    if not cv2.imwrite(str(depth_preview), depth_color):
        raise RuntimeError(f"failed to write {depth_preview}")

    return {
        "selected_sequence": selected_frame["sequence"],
        "rgb": file_record(rgb_preview, session_dir),
        "depth": file_record(depth_preview, session_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="head", choices=("head", "left", "right"))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "data" / "captures" / "pasteur"),
    )
    parser.add_argument(
        "--condition", default="overhead_before_front_scan"
    )
    parser.add_argument(
        "--robot-state",
        action="store_true",
        help="bracket capture with read-only qpos/torque/EE RPC snapshots",
    )
    parser.add_argument("--robot-host", default="localhost")
    parser.add_argument("--robot-port", type=int, default=8081)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup_frames < 0:
        parser.error("frames must be positive and warmup-frames non-negative")

    created_ns = time.time_ns()
    session_id = (
        datetime.fromtimestamp(created_ns / 1e9, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        + f"_{args.camera}_{args.condition}_{uuid.uuid4().hex[:8]}"
    )
    session_dir = (
        Path(args.output_root)
        / datetime.fromtimestamp(created_ns / 1e9, tz=timezone.utc).date().isoformat()
        / session_id
    )
    session_dir.mkdir(parents=True, exist_ok=False)
    raw_root = session_dir / "raw" / args.camera
    raw_root.mkdir(parents=True)
    calibration_dir = session_dir / "calibration"
    calibration_dir.mkdir()
    if CAMERA_MAP_PATH.exists():
        shutil.copy2(CAMERA_MAP_PATH, calibration_dir / CAMERA_MAP_PATH.name)

    partial_manifest = session_dir / "manifest.partial.json"
    clock_start = {
        "host_realtime_ns": time.time_ns(),
        "host_monotonic_ns": time.monotonic_ns(),
    }
    manifest: dict[str, Any] = {
        "schema": "piper_robot.rgbd_capture_session",
        "schema_version": 1,
        "session_id": session_id,
        "status": "collecting",
        "purpose": args.condition,
        "created_at_utc": utc_iso(created_ns),
        "camera_label": args.camera,
        "requested_frames": args.frames,
        "warmup_frames": args.warmup_frames,
        "orientation": {
            "raw": "Record3D native portrait",
            "derived_preview": "90 degrees clockwise",
        },
        "clock": {
            "host_clock_mapping_start": clock_start,
            "record3d_misc_timestamp_semantics": (
                "preserved verbatim; not independently verified as sensor time"
            ),
        },
        "repository": git_provenance(),
        "capture_script": {
            "path": Path(__file__).relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(Path(__file__)),
        },
    }
    write_json(partial_manifest, manifest)

    robot_before = None
    robot_after = None
    if args.robot_state:
        robot_before = robot_state_snapshot(args.robot_host, args.robot_port)

    devices = Record3DStream.get_connected_devices()
    camera_map = load_camera_map()
    camera_index = int(camera_map.get(args.camera, 0))
    if camera_index >= len(devices):
        raise RuntimeError(
            f"{args.camera} index {camera_index} unavailable; found {len(devices)}"
        )
    device = devices[camera_index]
    stream = Record3DStream()
    frame_queue: queue.Queue[dict[str, Any]] = queue.Queue(
        maxsize=max(8, args.frames * 2)
    )
    capture_done = threading.Event()
    writer_done = threading.Event()
    callback_lock = threading.Lock()
    errors: list[str] = []
    frame_records: list[dict[str, Any]] = []
    callback_count = 0
    accepted_count = 0

    frames_jsonl = session_dir / "frames.jsonl"

    def writer() -> None:
        try:
            with frames_jsonl.open("a", encoding="utf-8") as index_stream:
                while not writer_done.is_set() or not frame_queue.empty():
                    try:
                        bundle = frame_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    sequence = int(bundle["sequence"])
                    frame_dir = raw_root / f"{sequence:06d}"
                    frame_dir.mkdir()
                    rgb_path = frame_dir / "rgb.png"
                    depth_path = frame_dir / "depth.npy"
                    confidence_path = frame_dir / "confidence.npy"
                    meta_path = frame_dir / "meta.json"

                    rgb = bundle.pop("rgb")
                    depth = bundle.pop("depth")
                    confidence = bundle.pop("confidence")
                    if not cv2.imwrite(
                        str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    ):
                        raise RuntimeError(f"failed to write {rgb_path}")
                    np.save(depth_path, depth, allow_pickle=False)
                    np.save(confidence_path, confidence, allow_pickle=False)
                    write_json(meta_path, bundle)

                    files = {
                        "rgb_png": file_record(rgb_path, session_dir),
                        "depth_npy": file_record(depth_path, session_dir),
                        "confidence_npy": file_record(
                            confidence_path, session_dir
                        ),
                        "meta_json": file_record(meta_path, session_dir),
                    }
                    record = {
                        "sequence": sequence,
                        "condition": args.condition,
                        "host_callback_realtime_ns": bundle[
                            "host_callback_realtime_ns"
                        ],
                        "host_callback_monotonic_ns": bundle[
                            "host_callback_monotonic_ns"
                        ],
                        "rgb_shape": bundle["rgb_shape"],
                        "depth_shape": bundle["depth_shape"],
                        "confidence_shape": bundle["confidence_shape"],
                        "quality": bundle["quality"],
                        "camera_pose": bundle["camera_pose"],
                        "intrinsics": bundle["intrinsics"],
                        "record3d_misc": bundle["record3d_misc"],
                        "files": files,
                    }
                    index_stream.write(
                        json.dumps(record, sort_keys=True) + "\n"
                    )
                    index_stream.flush()
                    os.fsync(index_stream.fileno())
                    frame_records.append(record)
                    frame_queue.task_done()
        except Exception as exc:  # leave a diagnosable partial session
            errors.append(f"writer: {type(exc).__name__}: {exc}")
            capture_done.set()

    writer_thread = threading.Thread(target=writer, daemon=True)
    writer_thread.start()

    def on_new_frame() -> None:
        nonlocal callback_count, accepted_count
        if capture_done.is_set() or not callback_lock.acquire(blocking=False):
            return
        try:
            callback_count += 1
            if callback_count <= args.warmup_frames:
                return
            start_realtime_ns = time.time_ns()
            start_monotonic_ns = time.monotonic_ns()
            rgb = np.array(stream.get_rgb_frame(), copy=True)
            depth = np.array(stream.get_depth_frame(), copy=True)
            confidence = np.array(stream.get_confidence_frame(), copy=True)
            coefficients = stream.get_intrinsic_mat()
            camera_pose = stream.get_camera_pose()
            misc = np.array(stream.get_misc_data(), copy=True)
            end_realtime_ns = time.time_ns()
            end_monotonic_ns = time.monotonic_ns()

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            valid_depth = np.isfinite(depth) & (depth > 0)
            quality = {
                "rgb_mean_gray": float(np.mean(gray)),
                "rgb_near_black_fraction": float(np.mean(gray < 8)),
                "rgb_sharpness_laplacian_var": float(
                    cv2.Laplacian(gray, cv2.CV_64F).var()
                ),
                "depth_valid_fraction": float(np.mean(valid_depth)),
                "depth_median_m": (
                    float(np.median(depth[valid_depth]))
                    if np.any(valid_depth)
                    else None
                ),
                "confidence_high_fraction": (
                    float(np.mean(confidence == 2))
                    if confidence.size
                    else None
                ),
            }
            bundle = {
                "sequence": accepted_count,
                "host_callback_realtime_ns": start_realtime_ns,
                "host_callback_monotonic_ns": start_monotonic_ns,
                "host_copy_end_realtime_ns": end_realtime_ns,
                "host_copy_end_monotonic_ns": end_monotonic_ns,
                "rgb_shape": list(rgb.shape),
                "rgb_dtype": str(rgb.dtype),
                "rgb_color_space": "RGB",
                "depth_shape": list(depth.shape),
                "depth_dtype": str(depth.dtype),
                "depth_unit": "m",
                "depth_invalid": "non-finite or <= 0",
                "confidence_shape": list(confidence.shape),
                "confidence_dtype": str(confidence.dtype),
                "confidence_values": {
                    "0": "low",
                    "1": "medium",
                    "2": "high",
                },
                "intrinsics": intrinsics_dict(coefficients, rgb.shape),
                "camera_pose": pose_dict(camera_pose),
                "record3d_misc": decode_misc(misc),
                "quality": quality,
                "rgb": rgb,
                "depth": depth,
                "confidence": confidence,
            }
            frame_queue.put_nowait(bundle)
            accepted_count += 1
            if accepted_count >= args.frames:
                capture_done.set()
        except queue.Full:
            errors.append("callback: writer queue full")
            capture_done.set()
        except Exception as exc:
            errors.append(f"callback: {type(exc).__name__}: {exc}")
            capture_done.set()
        finally:
            callback_lock.release()

    stream.on_new_frame = on_new_frame
    stream.on_stream_stopped = lambda: errors.append(
        "Record3D stream stopped unexpectedly"
    )
    stream.connect(device)
    # The native binding initializes this field to zero, which is also the
    # enum value for TrueDepth.  It is not meaningful until a frame arrives.
    device_type_at_connect = int(stream.get_device_type())
    completed = capture_done.wait(timeout=args.timeout_s)
    if not completed:
        errors.append(
            f"capture timeout after {args.timeout_s}s "
            f"({accepted_count}/{args.frames} frames)"
        )
    writer_done.set()
    writer_thread.join(timeout=20.0)
    if writer_thread.is_alive():
        errors.append("writer did not finish within 20s")
    device_type = (
        int(stream.get_device_type()) if callback_count > 0 else None
    )

    if args.robot_state:
        robot_after = robot_state_snapshot(args.robot_host, args.robot_port)

    quality_summary: dict[str, Any] = {}
    if frame_records:
        for key in (
            "rgb_mean_gray",
            "rgb_near_black_fraction",
            "rgb_sharpness_laplacian_var",
            "depth_valid_fraction",
            "depth_median_m",
            "confidence_high_fraction",
        ):
            values = [
                record["quality"][key]
                for record in frame_records
                if record["quality"][key] is not None
            ]
            if values:
                quality_summary[key] = {
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

    robot_stability = None
    if robot_before is not None and robot_after is not None:
        required = ("left_joint_positions_rad", "right_joint_positions_rad")
        if all(key in robot_before and key in robot_after for key in required):
            left_delta = np.abs(
                np.asarray(robot_after["left_joint_positions_rad"])
                - np.asarray(robot_before["left_joint_positions_rad"])
            )
            right_delta = np.abs(
                np.asarray(robot_after["right_joint_positions_rad"])
                - np.asarray(robot_before["right_joint_positions_rad"])
            )
            robot_stability = {
                "left_max_abs_joint_delta_rad": float(np.max(left_delta)),
                "right_max_abs_joint_delta_rad": float(np.max(right_delta)),
                "stationary_threshold_rad": 0.005,
                "stationary": bool(
                    np.max(left_delta) <= 0.005
                    and np.max(right_delta) <= 0.005
                ),
            }
        else:
            robot_stability = {
                "stationary": None,
                "reason": "joint position unavailable in one or both snapshots",
            }

    preview = None
    if frame_records:
        preview = write_preview(
            session_dir, frame_records[len(frame_records) // 2]
        )

    manifest.update(
        {
            "status": (
                "complete"
                if not errors and len(frame_records) == args.frames
                else "rejected"
            ),
            "completed_at_utc": utc_iso(),
            "actual_frames": len(frame_records),
            "callbacks_seen": callback_count,
            "device": {
                "index": camera_index,
                "udid": getattr(device, "udid", None),
                "product_id": getattr(device, "product_id", None),
                "record3d_device_type": device_type,
                "record3d_device_type_name": {
                    0: "TrueDepth",
                    1: "LiDAR",
                }.get(device_type, "unavailable"),
                "record3d_device_type_source": (
                    "after_first_callback"
                    if callback_count > 0
                    else "unavailable_no_callbacks"
                ),
                "record3d_device_type_at_connect_uninitialized": (
                    device_type_at_connect
                ),
            },
            "clock": {
                **manifest["clock"],
                "host_clock_mapping_end": {
                    "host_realtime_ns": time.time_ns(),
                    "host_monotonic_ns": time.monotonic_ns(),
                },
            },
            "frame_index": file_record(frames_jsonl, session_dir)
            if frames_jsonl.exists()
            else None,
            "quality_summary": quality_summary,
            "robot_state": {
                "before": robot_before,
                "after": robot_after,
                "stability": robot_stability,
                "commands_sent": False,
            },
            "derived_preview": preview,
            "errors": errors,
        }
    )
    write_json(partial_manifest, manifest)
    final_manifest = session_dir / "manifest.json"
    os.replace(partial_manifest, final_manifest)

    result = {
        "session_dir": str(session_dir),
        "manifest": str(final_manifest),
        "status": manifest["status"],
        "frames": len(frame_records),
        "preview": (
            str(session_dir / preview["rgb"]["path"]) if preview else None
        ),
        "errors": errors,
    }
    print(json.dumps(result, indent=2), flush=True)
    return 0 if manifest["status"] == "complete" else 2


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
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    finally:
        # Avoid the native Record3D destructor/disconnect crash seen on this host.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)
