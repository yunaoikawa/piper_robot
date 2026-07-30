#!/usr/bin/env python3
"""Import a Record3D EXR+JPG video as a stopped-view RGB-D capture.

The importer samples views by camera-path distance, not by frame number.  Near
each target it selects a sharp three-frame burst whose Record3D pose is stable.
This keeps moving-camera blur out of SAM while preserving metric baseline for
multiview fusion.  It never connects to or commands the robot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.multiview_scene import camera_pose_stability
from src.capture_record3d_bundle import (
    file_record,
    git_provenance,
    utc_iso,
    write_json,
)


def _openexr_depth(path: Path) -> np.ndarray:
    try:
        import OpenEXR
    except ImportError as exc:
        raise RuntimeError(
            "Record3D EXR import requires `python -m pip install OpenEXR`"
        ) from exc
    channels = OpenEXR.File(str(path)).channels()
    if "R" not in channels:
        raise ValueError(f"{path}: Record3D depth channel R is missing")
    depth = np.asarray(channels["R"].pixels, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"{path}: expected one-channel depth, got {depth.shape}")
    return depth


def _pose_record(raw_pose) -> dict:
    values = np.asarray(raw_pose, dtype=float).reshape(7)
    return {
        "quaternion_xyzw": values[:4].tolist(),
        "translation_xyz_m": values[4:].tolist(),
        "frame": "record3d_session_local",
    }


def _camera_matrix(coefficients) -> list[list[float]]:
    fx, fy, cx, cy = np.asarray(coefficients, dtype=float).reshape(4)
    if min(fx, fy) <= 0.0:
        raise ValueError("Record3D focal lengths must be positive")
    return [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]]


def _path_distance(poses) -> np.ndarray:
    translations = np.asarray(poses, dtype=float)[:, 4:7]
    increments = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(increments)))


def path_fraction_targets(poses, view_count: int) -> list[int]:
    """Return indices spaced over travelled distance, excluding endpoints."""

    if view_count < 2:
        raise ValueError("at least two views are required")
    distance = _path_distance(poses)
    if distance[-1] < 0.04:
        raise ValueError("Record3D camera path baseline is below 4 cm")
    fractions = np.linspace(0.05, 0.95, view_count)
    return [
        int(np.argmin(np.abs(distance - fraction * distance[-1])))
        for fraction in fractions
    ]


def _sharpness(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    scale = min(1.0, 640.0 / image.shape[1])
    if scale < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def select_view_centers(
    poses,
    rgb_dir: Path,
    *,
    view_count: int,
    frames_per_view: int,
    candidate_radius: int,
) -> list[dict]:
    """Choose diverse, sharp, stable bursts around path-fraction targets."""

    if frames_per_view < 3 or frames_per_view % 2 != 1:
        raise ValueError("frames-per-view must be an odd integer >= 3")
    half = frames_per_view // 2
    pose_records = [_pose_record(value) for value in poses]
    distance = _path_distance(poses)
    selected = []
    used = set()
    for target in path_fraction_targets(poses, view_count):
        lower = max(half, target - candidate_radius)
        upper = min(len(poses) - half - 1, target + candidate_radius)
        candidates = []
        for center in range(lower, upper + 1):
            indices = list(range(center - half, center + half + 1))
            stability = camera_pose_stability(
                [pose_records[index] for index in indices]
            )
            sharpness = _sharpness(rgb_dir / f"{center}.jpg")
            candidates.append((center, indices, stability, sharpness))
        if not candidates:
            raise ValueError(f"no candidate frames around target {target}")
        sharp_scale = max(
            float(np.percentile([item[3] for item in candidates], 90.0)),
            1e-6,
        )

        def rank(item) -> tuple:
            center, _, stability, sharpness = item
            path_error = abs(distance[center] - distance[target])
            motion = (
                stability["maximum_translation_m"] / 0.003
                + stability["maximum_rotation_deg"] / 1.0
            )
            score = (
                sharpness / sharp_scale
                - 0.40 * path_error / max(distance[-1] * 0.05, 1e-6)
                - 0.30 * motion
            )
            return (stability["accepted"], score, -abs(center - target))

        ordered = sorted(candidates, key=rank, reverse=True)
        choice = next(
            (item for item in ordered if not set(item[1]) & used),
            ordered[0],
        )
        center, indices, stability, sharpness = choice
        used.update(indices)
        selected.append(
            {
                "target_index": int(target),
                "center_index": int(center),
                "frame_indices": indices,
                "path_fraction": float(distance[center] / distance[-1]),
                "sharpness": float(sharpness),
                "pose_stability": stability,
            }
        )
    return selected


def _write_selected_view(
    export_dir: Path,
    capture_dir: Path,
    metadata: dict,
    selection: dict,
    view_name: str,
    *,
    minimum_valid_depth_fraction: float,
) -> dict:
    raw_root = capture_dir / "raw" / "head" / view_name
    raw_root.mkdir(parents=True)
    records = []
    for sequence, source_index in enumerate(selection["frame_indices"]):
        frame_dir = raw_root / f"{sequence:06d}"
        frame_dir.mkdir()
        source_rgb = export_dir / "rgb" / f"{source_index}.jpg"
        source_depth = export_dir / "depth" / f"{source_index}.exr"
        rgb_path = frame_dir / "rgb.jpg"
        depth_path = frame_dir / "depth.npy"
        confidence_path = frame_dir / "confidence.npy"
        meta_path = frame_dir / "meta.json"
        shutil.copy2(source_rgb, rgb_path)
        depth = _openexr_depth(source_depth)
        valid = np.isfinite(depth) & (depth > 0.0)
        valid_fraction = float(np.mean(valid))
        if valid_fraction < minimum_valid_depth_fraction:
            raise ValueError(
                f"{source_depth}: valid depth fraction {valid_fraction:.3f} "
                f"< {minimum_valid_depth_fraction:.3f}"
            )
        confidence = np.zeros(depth.shape, dtype=np.uint8)
        confidence[valid] = 1
        np.save(depth_path, depth, allow_pickle=False)
        np.save(confidence_path, confidence, allow_pickle=False)
        coefficients = metadata["perFrameIntrinsicCoeffs"][source_index]
        source_timestamp = metadata["frameTimestamps"][source_index]
        record = {
            "sequence": sequence,
            "source_frame_index": int(source_index),
            "source_timestamp_s": float(source_timestamp),
            "rgb_shape": [int(metadata["h"]), int(metadata["w"]), 3],
            "depth_shape": list(depth.shape),
            "confidence_shape": list(depth.shape),
            "depth_unit": "m",
            "intrinsics": {
                "K_raw_rgb": _camera_matrix(coefficients),
                "coefficients": {
                    key: float(value)
                    for key, value in zip(("fx", "fy", "tx", "ty"), coefficients)
                },
            },
            "camera_pose": _pose_record(metadata["poses"][source_index]),
            "quality": {
                "depth_valid_fraction": valid_fraction,
                "depth_median_m": float(np.median(depth[valid])),
                "confidence_source": "valid_exr_synthesized_medium",
            },
        }
        write_json(meta_path, record)
        record["files"] = {
            "rgb_png": file_record(rgb_path, capture_dir),
            "depth_npy": file_record(depth_path, capture_dir),
            "confidence_npy": file_record(confidence_path, capture_dir),
            "meta_json": file_record(meta_path, capture_dir),
        }
        records.append(record)

    derived = capture_dir / "derived" / view_name
    derived.mkdir(parents=True)
    representative = records[len(records) // 2]
    representative_rgb = (
        capture_dir / representative["files"]["rgb_png"]["path"]
    )
    preview = derived / "rgb_landscape.jpg"
    shutil.copy2(representative_rgb, preview)
    index = derived / "frames.jsonl"
    index.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "name": view_name,
        "frame_count": len(records),
        "representative_sequence": int(representative["sequence"]),
        "selection": selection,
        "pose_stability": selection["pose_stability"],
        "frames_index": file_record(index, capture_dir),
        "preview": file_record(preview, capture_dir),
        "frames": records,
    }


def import_video(args) -> dict:
    export_dir = Path(args.input).resolve()
    metadata_path = export_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    frame_count = len(metadata.get("poses", ()))
    required_lengths = {
        "poses": frame_count,
        "perFrameIntrinsicCoeffs": frame_count,
        "frameTimestamps": frame_count,
    }
    if (
        metadata.get("cameraType") != 1
        or frame_count < args.view_count * args.frames_per_view
        or any(len(metadata.get(key, ())) != count for key, count in required_lengths.items())
    ):
        raise ValueError("input is not a complete Record3D LiDAR EXR+JPG export")
    if not all(
        (export_dir / "rgb" / f"{index}.jpg").is_file()
        and (export_dir / "depth" / f"{index}.exr").is_file()
        for index in range(frame_count)
    ):
        raise ValueError("Record3D RGB/depth frame sequence is incomplete")

    selections = select_view_centers(
        metadata["poses"],
        export_dir / "rgb",
        view_count=args.view_count,
        frames_per_view=args.frames_per_view,
        candidate_radius=args.candidate_radius,
    )
    created_ns = time.time_ns()
    session_id = (
        datetime.fromtimestamp(created_ns / 1e9, tz=timezone.utc).strftime(
            "%Y%m%dT%H%M%S.%fZ"
        )
        + f"_head_record3d_exr_{uuid.uuid4().hex[:8]}"
    )
    capture_dir = (
        Path(args.output_root).resolve()
        / datetime.fromtimestamp(created_ns / 1e9, tz=timezone.utc).date().isoformat()
        / session_id
    )
    capture_dir.mkdir(parents=True)
    views = []
    for index, selection in enumerate(selections):
        views.append(
            _write_selected_view(
                export_dir,
                capture_dir,
                metadata,
                selection,
                f"path_{index:02d}",
                minimum_valid_depth_fraction=args.minimum_valid_depth_fraction,
            )
        )
    manifest = {
        "schema": "piper_robot.rgbd_multiview_capture/v1",
        "schema_version": 1,
        "session_id": session_id,
        "status": "complete",
        "purpose": "record3d_exr_video_keyframes",
        "created_at_utc": utc_iso(created_ns),
        "completed_at_utc": utc_iso(),
        "camera_label": "head",
        "view_order": [view["name"] for view in views],
        "frames_per_view": args.frames_per_view,
        "pose_frame": "single_record3d_session_local",
        "capture_mode": "imported_record3d_exr_jpg_video",
        "commands_sent": False,
        "device": {
            "record3d_device_type": 1,
            "record3d_device_type_name": "LiDAR",
        },
        "source": {
            "export_dir": str(export_dir),
            "metadata": file_record(metadata_path, export_dir),
            "frame_count": frame_count,
            "fps": metadata.get("fps"),
            "selection_method": (
                "camera_path_fraction_plus_pose_stability_and_rgb_sharpness"
            ),
        },
        "repository": git_provenance(),
        "views": views,
        "robot_state": {
            "before": None,
            "after": None,
            "commands_sent": False,
        },
        "limitations": [
            "Record3D EXR export does not include confidence maps; finite depth is medium confidence",
            "robot joint state was not sampled during the phone recording",
        ],
    }
    write_json(capture_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "capture_dir": str(capture_dir),
                "manifest": str(capture_dir / "manifest.json"),
                "selected_source_frames": [
                    selection["center_index"] for selection in selections
                ],
            },
            indent=2,
        )
    )
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output-root",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "captures"
            / "pasteur"
        ),
    )
    parser.add_argument("--view-count", type=int, default=5)
    parser.add_argument("--frames-per-view", type=int, default=3)
    parser.add_argument("--candidate-radius", type=int, default=24)
    parser.add_argument(
        "--minimum-valid-depth-fraction",
        type=float,
        default=0.20,
    )
    args = parser.parse_args(argv)
    if (
        args.view_count < 2
        or args.frames_per_view < 3
        or args.frames_per_view % 2 != 1
        or args.candidate_radius < 0
        or not 0.0 < args.minimum_valid_depth_fraction <= 1.0
    ):
        parser.error("invalid view selection or depth validation arguments")
    import_video(args)


if __name__ == "__main__":
    main()
