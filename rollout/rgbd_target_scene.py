"""Refresh one completed target object in a reviewed MuJoCo scene.

The semantic adapter owns the image mask.  RGB-D and a fixed metric tag own
the target surface pose.  The object catalog owns completed collision size.
This keeps task-specific pixels and stale reconstructed object poses out of
the motion model.
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from rollout.scene_registration import (
    bridge_camera_from_fixed_tag,
    tag_pose_camera,
    transform_points,
)
from src.refine_scene_robot_alignment import _reference_tag_consensus


def selected_head_rgbd(capture: str | Path):
    capture = Path(capture).resolve()
    manifest = json.loads((capture / "manifest.json").read_text())
    sequence = int(manifest["derived_preview"]["selected_sequence"])
    frame = capture / "raw" / "head" / f"{sequence:06d}"
    meta = json.loads((frame / "meta.json").read_text())
    image = cv2.imread(
        str(capture / manifest["derived_preview"]["rgb"]["path"]),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise FileNotFoundError("selected head RGB preview is missing")
    depth = cv2.rotate(
        np.load(frame / "depth.npy"), cv2.ROTATE_90_CLOCKWISE
    )
    confidence = cv2.rotate(
        np.load(frame / "confidence.npy"), cv2.ROTATE_90_CLOCKWISE
    )
    return image, depth, confidence, meta, manifest


def target_surface_in_scene(
    *,
    image_bgr: np.ndarray,
    depth_landscape_m: np.ndarray,
    confidence_landscape: np.ndarray,
    meta: dict,
    mask_rgb: np.ndarray,
    reference_report: str | Path,
    reference_capture: str | Path,
    tag_id: int,
    tag_size_m: float,
) -> tuple[np.ndarray, dict]:
    """Estimate a robust visible target-surface point in scene coordinates."""

    mask = cv2.resize(
        np.asarray(mask_rgb, dtype=np.uint8) * 255,
        (depth_landscape_m.shape[1], depth_landscape_m.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    valid = (
        (mask >= 64)
        & np.isfinite(depth_landscape_m)
        & (depth_landscape_m > 0)
        & (np.asarray(confidence_landscape) >= 1)
    )
    if int(np.count_nonzero(valid)) < 8:
        raise ValueError("target mask has insufficient valid RGB-D support")
    matrix_rgb = np.asarray(
        meta["intrinsics"]["K_rgb_rotated_clockwise"], dtype=float
    )
    image_height, image_width = image_bgr.shape[:2]
    depth_height, depth_width = depth_landscape_m.shape
    matrix_depth = matrix_rgb.copy()
    matrix_depth[0] *= depth_width / image_width
    matrix_depth[1] *= depth_height / image_height
    rows, columns = np.nonzero(valid)
    depths = np.asarray(depth_landscape_m[rows, columns], dtype=float)
    points_camera = np.column_stack(
        (
            (columns - matrix_depth[0, 2]) * depths / matrix_depth[0, 0],
            (rows - matrix_depth[1, 2]) * depths / matrix_depth[1, 1],
            depths,
        )
    )
    reference, reference_records = _reference_tag_consensus(
        Path(reference_report).resolve(),
        Path(reference_capture).resolve(),
        tag_id=int(tag_id),
        tag_size_m=float(tag_size_m),
    )
    camera_from_tag, tag_rms = tag_pose_camera(
        image_bgr,
        matrix_rgb,
        tag_id=int(tag_id),
        tag_size_m=float(tag_size_m),
    )
    scene_from_camera = bridge_camera_from_fixed_tag(
        reference.transform, camera_from_tag
    )
    points_scene = transform_points(points_camera, scene_from_camera)
    center = np.median(points_scene, axis=0)
    spread = np.percentile(
        np.linalg.norm(points_scene - center, axis=1), 90
    )
    return center, {
        "valid_depth_pixels": int(len(points_scene)),
        "surface_scene_xyz_m": center.tolist(),
        "surface_spread_p90_m": float(spread),
        "tag_reprojection_rms_px": float(tag_rms),
        "reference_tag_translation_spread_m": float(
            reference.translation_spread_m
        ),
        "reference_tag_rotation_spread_deg": float(reference.rotation_spread_deg),
        "reference_tag_records": reference_records,
        "scene_from_camera": scene_from_camera.tolist(),
    }


def write_completed_vertical_object_scene(
    *,
    source_model: str | Path,
    output_model: str | Path,
    body_name: str,
    visible_top_scene_xyz_m,
    diameter_m: float,
    height_m: float,
    top_feature_diameter_m: float | None = None,
    top_feature_height_m: float = 0.0,
    unobserved_dynamic_bodies: tuple[str, ...] = (),
) -> dict:
    """Complete a vertical object downward from its observed top surface."""

    source_model = Path(source_model).resolve()
    output_model = Path(output_model).resolve()
    tree = ET.parse(source_model)
    root = tree.getroot()
    for include in root.findall(".//include"):
        included = Path(include.get("file", ""))
        if included and not included.is_absolute():
            include.set("file", str((source_model.parent / included).resolve()))
    compiler = root.find("compiler")
    if compiler is not None:
        for attribute in ("meshdir", "texturedir"):
            configured = compiler.get(attribute)
            if configured and not Path(configured).is_absolute():
                compiler.set(
                    attribute,
                    str((source_model.parent / configured).resolve()),
                )
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"planning model lacks body {body_name!r}")
    geom = body.find("geom")
    if geom is None:
        raise ValueError(f"planning body {body_name!r} lacks geometry")
    top = np.asarray(visible_top_scene_xyz_m, dtype=float)
    top_height = float(top_feature_height_m)
    if top_height < 0.0 or top_height >= float(height_m):
        raise ValueError("top feature height must lie within completed height")
    if top_height > 0.0 and top_feature_diameter_m is None:
        raise ValueError("top feature diameter is required when its height is positive")
    body_height = float(height_m) - top_height
    body.set("pos", " ".join(f"{value:.8f}" for value in top))
    body.set("euler", "0 0 0")
    geom.set("type", "cylinder")
    geom.set("size", f"{float(diameter_m) / 2.0:.8f} {body_height / 2.0:.8f}")
    geom.set("pos", f"0 0 {-top_height - body_height / 2.0:.8f}")
    geom.set("name", f"{body_name}-completed-collision")
    if top_height > 0.0:
        top_geom = ET.SubElement(body, "geom")
        top_geom.set("name", f"{body_name}-top-feature-collision")
        top_geom.set("type", "cylinder")
        top_geom.set(
            "size",
            f"{float(top_feature_diameter_m) / 2.0:.8f} {top_height / 2.0:.8f}",
        )
        top_geom.set("pos", f"0 0 {-top_height / 2.0:.8f}")
        top_geom.set("rgba", geom.get("rgba", "1 0.55 0.1 0.84"))
        top_geom.set("contype", "1")
        top_geom.set("conaffinity", "1")
    disabled = []
    for stale_name in unobserved_dynamic_bodies:
        stale = root.find(f".//body[@name='{stale_name}']")
        if stale is None:
            continue
        for stale_geom in stale.findall(".//geom"):
            stale_geom.set("contype", "0")
            stale_geom.set("conaffinity", "0")
        disabled.append(stale_name)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_model, encoding="unicode")
    return {
        "source_model": str(source_model),
        "output_model": str(output_model),
        "body_name": body_name,
        "completion": "catalog_vertical_cylinder_down_from_observed_top",
        "visible_top_scene_xyz_m": top.tolist(),
        "body_center_scene_xyz_m": (
            top + np.asarray([0.0, 0.0, -top_height - body_height / 2.0])
        ).tolist(),
        "diameter_m": float(diameter_m),
        "height_m": float(height_m),
        "top_feature_diameter_m": (
            None if top_feature_diameter_m is None else float(top_feature_diameter_m)
        ),
        "top_feature_height_m": top_height,
        "unobserved_dynamic_collision_disabled": disabled,
        "motion_scope": "free_space_alignment_to_observed_target_only",
    }
