#!/usr/bin/env python3
"""Connect a fixed head calibration to a saved multiview semantic scene."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.fixed_head_scene_registration import (
    refine_against_reference_rgbd,
    register_rgbd_features,
    validate_against_static_scene,
)
from rollout.semantic_scene_pipeline import sha256_file
from src.capture_record3d_multiview import _robot_qpos
from src.reconstruct_multiview_scene import _temporal_view


SCHEMA = "piper_robot.robot_scene_registration/v1"
CALIBRATION_SCHEMA = "piper_robot.camera_robot_calibration/v1"
MULTIVIEW_SCHEMA = "piper_robot.multiview_semantic_scene/v1"


def _validated_transform(value, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if np.linalg.det(rotation) < 0.999:
        raise ValueError(f"{name} rotation is reflected")
    return transform


def _view_by_name(manifest: dict, name: str) -> dict:
    for view in manifest.get("views", ()):
        if view.get("name") == name:
            return view
    raise KeyError(name)


def _representative_qpos(view: dict) -> list[float]:
    state = view.get("robot_state", {})
    stability = state.get("stability", {})
    value = stability.get("representative_qpos_rad")
    if stability.get("accepted") is not True or value is None:
        before = _robot_qpos(state.get("before"))
        after = _robot_qpos(state.get("after"))
        if before is None or after is None:
            raise ValueError("fixed capture lacks stable synchronized qpos")
        value = ((before + after) / 2.0).tolist()
    result = np.asarray(value, dtype=float)
    if result.shape != (12,) or not np.all(np.isfinite(result)):
        raise ValueError("representative qpos must contain 12 finite values")
    return result.tolist()


def _union_masks(paths, shape) -> np.ndarray | None:
    combined = np.zeros(shape[:2], dtype=np.uint8)
    count = 0
    for path in paths:
        mask = cv2.imread(str(Path(path).resolve()), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        if mask.shape != combined.shape:
            mask = cv2.resize(
                mask,
                combined.shape[::-1],
                interpolation=cv2.INTER_NEAREST,
            )
        combined[mask > 0] = 1
        count += 1
    return combined if count else None


def build(args) -> dict:
    fixed_capture = Path(args.fixed_capture).resolve()
    fixed_manifest_path = fixed_capture / "manifest.json"
    fixed_manifest = json.loads(fixed_manifest_path.read_text())
    if fixed_manifest.get("schema") != "piper_robot.rgbd_multiview_capture/v1":
        raise ValueError("fixed capture schema is unsupported")
    if fixed_manifest.get("operator_action") != "move-robot":
        raise ValueError("fixed capture must keep the camera fixed and move robot")
    if fixed_manifest.get("commands_sent") is not False:
        raise ValueError("fixed capture command provenance is unsafe")
    fixed_views = list(fixed_manifest.get("views", ()))
    if not fixed_views:
        raise ValueError("fixed capture has no completed views")
    fixed_view = (
        _view_by_name(fixed_manifest, args.fixed_view)
        if args.fixed_view
        else fixed_views[0]
    )
    fixed = _temporal_view(
        fixed_capture, fixed_view, minimum_confidence=args.minimum_confidence
    )

    calibration_path = Path(args.camera_calibration).resolve()
    calibration = json.loads(calibration_path.read_text())
    if calibration.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("camera calibration schema is unsupported")
    if calibration.get("accepted") is not True:
        raise ValueError("camera calibration is not accepted")
    robot_from_fixed = _validated_transform(
        calibration.get("T_robot_camera"), "T_robot_camera"
    )
    fixed_udid = fixed_manifest.get("device", {}).get("udid")
    calibration_udid = (
        calibration.get("record3d_udid")
        or calibration.get("source", {}).get("record3d_udid")
    )
    if fixed_udid != calibration_udid:
        raise ValueError("fixed capture camera UDID differs from calibration")

    multiview_path = Path(args.multiview_report).resolve()
    multiview = json.loads(multiview_path.read_text())
    if multiview.get("schema") != MULTIVIEW_SCHEMA:
        raise ValueError("multiview report schema is unsupported")
    reference_capture = Path(multiview["capture"]).resolve()
    reference_manifest = json.loads(
        (reference_capture / "manifest.json").read_text()
    )
    registration_by_name = {
        item["name"]: item
        for item in multiview.get("registration", {}).get("views", ())
        if item.get("accepted") is True
    }
    candidates = []
    failures = {}
    for reference_view in reference_manifest.get("views", ()):
        name = reference_view.get("name")
        registration = registration_by_name.get(name)
        if registration is None:
            continue
        try:
            reference = _temporal_view(
                reference_capture,
                reference_view,
                minimum_confidence=args.minimum_confidence,
            )
            feature = register_rgbd_features(
                fixed["rgb_bgr"],
                fixed["depth_m"],
                fixed["camera_matrix"],
                reference["rgb_bgr"],
                reference["depth_m"],
                reference["camera_matrix"],
            )
            if not feature.accepted:
                failures[name] = "metric feature gate rejected"
                continue
            level_from_reference = _validated_transform(
                registration["reference_from_camera"],
                f"{name} reference_from_camera",
            )
            level_from_fixed_seed = (
                level_from_reference @ feature.target_from_source
            )
            candidates.append(
                (
                    feature.inliers,
                    -feature.median_residual_m,
                    name,
                    feature,
                    level_from_fixed_seed,
                    reference,
                )
            )
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    if not candidates:
        raise RuntimeError(
            "no saved view passed fixed-camera metric registration: "
            + json.dumps(failures, sort_keys=True)
        )
    _, _, selected_name, feature, level_from_fixed_seed, reference = max(
        candidates, key=lambda item: item[:2]
    )

    source_mask_paths = (
        calibration.get("source", {})
        .get("mask_sources", {})
        .get(fixed_view["name"], ())
    )
    source_dynamic_mask = _union_masks(
        source_mask_paths, fixed["rgb_bgr"].shape
    )
    dynamic_names = {
        "robot",
        "petri_dish",
        "petri_lid",
        "culture_media_bottle",
    }
    target_mask_paths = [
        item["mask_path"]
        for item in multiview.get("semantics", {})
        .get("views", {})
        .get(selected_name, ())
        if item.get("semantic_name") in dynamic_names
    ]
    target_dynamic_mask = _union_masks(
        target_mask_paths, reference["rgb_bgr"].shape
    )
    target_from_fixed, reference_refinement = refine_against_reference_rgbd(
        fixed["depth_m"],
        fixed["camera_matrix"],
        reference["depth_m"],
        reference["camera_matrix"],
        feature.target_from_source,
        source_dynamic_mask=source_dynamic_mask,
        target_dynamic_mask=target_dynamic_mask,
    )
    selected_registration = registration_by_name[selected_name]
    level_from_fixed = (
        _validated_transform(
            selected_registration["reference_from_camera"],
            f"{selected_name} reference_from_camera",
        )
        @ target_from_fixed
    )

    archive = np.load(
        multiview_path.parent / "scene_mesh_multiview.npz"
    )
    vertices = np.asarray(archive["vertices_xyz_m"], dtype=float)
    labels = np.asarray(archive["semantic_labels"], dtype=int)
    semantic_ids = multiview.get("semantics", {}).get("label_ids", {})
    excluded = {
        int(semantic_ids[name])
        for name in ("robot", "petri_dish", "petri_lid", "culture_media_bottle")
        if name in semantic_ids
    }
    static = vertices[
        np.all(np.isfinite(vertices), axis=1)
        & ~np.isin(labels, list(excluded))
    ]
    scene_validation = validate_against_static_scene(
        fixed["depth_m"],
        fixed["camera_matrix"],
        level_from_fixed,
        static,
        source_dynamic_mask=source_dynamic_mask,
    )
    accepted = bool(
        feature.accepted
        and reference_refinement["accepted"]
        and scene_validation["accepted"]
    )
    robot_from_level = robot_from_fixed @ np.linalg.inv(level_from_fixed)
    qpos = _representative_qpos(fixed_view)
    result = {
        "schema": SCHEMA,
        "registration_id": (
            f"robot-scene-{calibration['calibration_id']}-"
            f"{multiview_path.parent.name}"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "accepted": accepted,
        "commands_sent": False,
        "transform_convention": {
            "T_level_fixed_camera": "p_level = T_level_fixed_camera @ p_fixed_camera",
            "T_robot_level": "p_robot = T_robot_level @ p_level",
        },
        "T_level_fixed_camera": level_from_fixed.tolist(),
        "T_robot_level": robot_from_level.tolist(),
        "selected_reference_view": selected_name,
        "feature_registration": {
            "accepted": feature.accepted,
            "matches": feature.matches,
            "metric_matches": feature.metric_matches,
            "inliers": feature.inliers,
            "inlier_fraction": feature.inlier_fraction,
            "median_residual_m": feature.median_residual_m,
            "p90_residual_m": feature.p90_residual_m,
        },
        "reference_rgbd_refinement": reference_refinement,
        "static_scene_validation": scene_validation,
        "robot_state": {
            "representative_qpos_rad": qpos,
            "source_view": fixed_view["name"],
            "ordering": "left_joint1_to_6_then_right_joint1_to_6",
        },
        "sources": {
            "fixed_capture_manifest": {
                "path": str(fixed_manifest_path),
                "sha256": sha256_file(fixed_manifest_path),
                "record3d_udid": fixed_udid,
            },
            "camera_calibration": {
                "path": str(calibration_path),
                "sha256": sha256_file(calibration_path),
                "calibration_id": calibration["calibration_id"],
            },
            "multiview_report": {
                "path": str(multiview_path),
                "sha256": sha256_file(multiview_path),
            },
        },
        "failures_by_reference_view": failures,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-capture", required=True)
    parser.add_argument("--fixed-view")
    parser.add_argument("--camera-calibration", required=True)
    parser.add_argument("--multiview-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-confidence", type=int, default=1)
    args = parser.parse_args(argv)
    result = build(args)
    print(json.dumps(result, indent=2), flush=True)
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
