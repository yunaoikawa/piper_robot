#!/usr/bin/env python3
"""Localize a SAM target from a wrist RGB-D support-plane intersection.

Transparent target pixels often have no depth.  The target centre ray is
therefore intersected with a robust plane fitted to valid depth in a
scale-relative ring around the SAM mask.  A previously accepted wrist
hand-eye calibration and a freshly fitted controller/CAD bridge then map the
point into the current semantic MuJoCo scene.

This is observation-only.  It never imports the robot RPC client and never
uses a demonstrated arm pose, trajectory, or gripper width.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rollout.wrist_rgbd_target import (
    TargetObservation,
    _read_frame_index,
    fit_controller_cad_bridge,
    median_rgbd,
)


def support_plane_target_point(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    target_mask: np.ndarray,
    *,
    ring_radius_fraction: float = 0.125,
    residual_threshold_m: float = 0.003,
    maximum_samples: int = 8000,
    ransac_iterations: int = 500,
    seed: int = 7,
) -> tuple[np.ndarray, dict]:
    depth = np.asarray(depth_m, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float)
    mask = np.asarray(target_mask, dtype=bool)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth and target mask must have the same 2D shape")
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera matrix must be finite 3x3")
    ys_target, xs_target = np.nonzero(mask)
    if len(xs_target) < 100:
        raise ValueError("SAM target mask is too small")
    radius = max(
        3,
        int(round(min(depth.shape) * float(ring_radius_fraction))),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    ring = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    ring &= ~mask
    ring &= np.isfinite(depth) & (depth > 0.05)
    ys, xs = np.nonzero(ring)
    values = depth[ys, xs]
    if len(values) < 300:
        raise ValueError("insufficient valid support depth around SAM target")
    generator = np.random.default_rng(seed)
    if len(values) > int(maximum_samples):
        selected = generator.choice(
            len(values), int(maximum_samples), replace=False
        )
        ys, xs, values = ys[selected], xs[selected], values[selected]
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    points = np.c_[
        (xs - cx) * values / fx,
        (ys - cy) * values / fy,
        values,
    ]
    best = None
    threshold = float(residual_threshold_m)
    for _ in range(int(ransac_iterations)):
        first, second, third = points[
            generator.choice(len(points), 3, replace=False)
        ]
        normal = np.cross(second - first, third - first)
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            continue
        normal /= length
        offset = -float(normal @ first)
        residual = np.abs(points @ normal + offset)
        count = int(np.count_nonzero(residual < threshold))
        if best is None or count > best[0]:
            best = (count, residual)
    if best is None:
        raise ValueError("support-plane RANSAC found no model")
    inliers = best[1] < threshold
    inlier_points = points[inliers]
    center = np.mean(inlier_points, axis=0)
    _, _, axes = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = axes[-1]
    offset = -float(normal @ center)
    if normal[2] > 0:
        normal = -normal
        offset = -offset
    residual = np.abs(inlier_points @ normal + offset)
    inlier_ratio = float(len(inlier_points) / len(points))
    if inlier_ratio < 0.70:
        raise ValueError(
            f"support plane inlier ratio is too low: {inlier_ratio:.3f}"
        )
    center_px = np.asarray(
        [float(np.mean(xs_target)), float(np.mean(ys_target))]
    )
    ray = np.asarray(
        [
            (center_px[0] - cx) / fx,
            (center_px[1] - cy) / fy,
            1.0,
        ]
    )
    denominator = float(normal @ ray)
    if abs(denominator) < 1e-6:
        raise ValueError("target ray is parallel to the support plane")
    distance = -offset / denominator
    if not np.isfinite(distance) or distance <= 0.05:
        raise ValueError("support-plane intersection is behind the camera")
    point = distance * ray
    return point, {
        "target_center_px": center_px.tolist(),
        "ring_radius_fraction": float(ring_radius_fraction),
        "ring_radius_px": radius,
        "support_depth_samples": int(len(points)),
        "support_plane_inliers": int(len(inlier_points)),
        "support_plane_inlier_ratio": inlier_ratio,
        "support_plane_median_residual_m": float(np.median(residual)),
        "support_plane_normal_camera": normal.tolist(),
        "support_plane_offset_camera_m": float(offset),
        "target_point_camera_m": point.tolist(),
        "target_mask_depth_valid_fraction": float(
            np.count_nonzero(
                mask & np.isfinite(depth) & (depth > 0.05)
            )
            / np.count_nonzero(mask)
        ),
    }


def _controller_from_ee(state: dict) -> np.ndarray:
    quaternion = np.asarray(state["quaternion_wxyz"], dtype=float)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(
        np.r_[quaternion[1:], quaternion[0]]
    ).as_matrix()
    transform[:3, 3] = np.asarray(
        state["translation_xyz_m"], dtype=float
    )
    return transform


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--target-mask", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--bridge-config", required=True)
    parser.add_argument("--scene-model", required=True)
    parser.add_argument("--model-branch", default="right")
    parser.add_argument("--input-scene", required=True)
    parser.add_argument("--semantic-name", default="petri_lid")
    parser.add_argument("--output-scene", required=True)
    parser.add_argument("--diagnostics", required=True)
    args = parser.parse_args(argv)

    capture = Path(args.capture).resolve()
    image, depth, _ = median_rgbd(capture)
    records = _read_frame_index(capture)
    orientation = json.loads(
        (capture / "manifest.json").read_text()
    )["orientation"]["derived_preview"]
    if orientation != "90 degrees clockwise":
        raise ValueError(f"unsupported capture orientation: {orientation}")
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
    matrix = np.asarray(
        records[0]["intrinsics"]["K_rgb_rotated_clockwise"],
        dtype=float,
    )
    mask = cv2.imread(
        str(Path(args.target_mask).resolve()), cv2.IMREAD_GRAYSCALE
    )
    if mask is None:
        raise ValueError("could not read target mask")
    target_camera, plane = support_plane_target_point(
        depth, matrix, mask > 0
    )

    calibration = json.loads(
        Path(args.calibration_report).resolve().read_text()
    )
    if calibration.get("accepted") is not True:
        raise ValueError("wrist calibration report was not accepted")
    bridge_profile = json.loads(
        Path(args.bridge_config).resolve().read_text()
    )["kinematic_bridge"]
    observations = [
        TargetObservation(**item)
        for item in calibration["capture_observations"]
    ]
    bridge = fit_controller_cad_bridge(
        observations,
        model_path=args.scene_model,
        model_branch=args.model_branch,
        profile=bridge_profile,
    )
    if bridge.get("accepted") is not True:
        raise ValueError(f"controller/CAD bridge rejected: {bridge}")

    manifest = json.loads((capture / "manifest.json").read_text())
    ee_state = manifest["robot_state"]["after"]["right_ee_pose"]
    controller_from_camera = (
        _controller_from_ee(ee_state)
        @ np.asarray(calibration["fit"]["ee_from_camera"], dtype=float)
    )
    target_controller = (
        controller_from_camera @ np.r_[target_camera, 1.0]
    )[:3]
    scene_from_controller = np.asarray(
        bridge["scene_from_cad_base"], dtype=float
    ) @ np.linalg.inv(
        np.asarray(bridge["controller_from_cad_base"], dtype=float)
    )
    target_scene_measured = (
        scene_from_controller @ np.r_[target_controller, 1.0]
    )[:3]

    input_scene_path = Path(args.input_scene).resolve()
    scene = json.loads(input_scene_path.read_text())
    matches = [
        item
        for item in scene.get("objects", [])
        if item.get("semantic_name") == args.semantic_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {args.semantic_name!r}, found {len(matches)}"
        )
    result = copy.deepcopy(scene)
    target = next(
        item
        for item in result["objects"]
        if item.get("semantic_name") == args.semantic_name
    )
    pose = np.asarray(target["pose_scene"], dtype=float)
    previous = pose[:3, 3].copy()
    pose[:2, 3] = target_scene_measured[:2]
    target["pose_scene"] = pose.tolist()
    target["status"] = "live_sam_wrist_rgbd_support_plane_localized"
    target.setdefault("perception", {}).update(
        {
            "sam_mask": str(Path(args.target_mask).resolve()),
            "support_plane_intersection": plane,
            "hand_eye_accepted": True,
            "kinematic_bridge_accepted": True,
            "successful_pose_or_trajectory_used": False,
        }
    )
    result["operator_confirmed"] = False
    source = result.setdefault("source", {})
    source["live_wrist_rgbd_localization"] = {
        "created_at_s": time.time(),
        "capture": str(capture),
        "calibration_report": str(
            Path(args.calibration_report).resolve()
        ),
        "scene_model": str(Path(args.scene_model).resolve()),
        "model_branch": args.model_branch,
        "previous_target_scene_xyz_m": previous.tolist(),
        "measured_target_scene_xyz_m": target_scene_measured.tolist(),
        "support_constrained_target_scene_xyz_m": pose[:3, 3].tolist(),
        "target_controller_xyz_m": target_controller.tolist(),
        "successful_pose_or_trajectory_used": False,
    }

    output_scene = Path(args.output_scene).resolve()
    diagnostics = Path(args.diagnostics).resolve()
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    output_scene.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    diagnostic = {
        "schema": "piper_robot.sam_wrist_rgbd_localization/v1",
        "commands_sent": False,
        "capture": str(capture),
        "target_mask": str(Path(args.target_mask).resolve()),
        "plane": plane,
        "target_controller_xyz_m": target_controller.tolist(),
        "target_scene_measured_xyz_m": target_scene_measured.tolist(),
        "target_scene_support_constrained_xyz_m": pose[:3, 3].tolist(),
        "previous_target_scene_xyz_m": previous.tolist(),
        "scene_xy_correction_m": (
            pose[:2, 3] - previous[:2]
        ).tolist(),
        "hand_eye_accepted": True,
        "kinematic_bridge": bridge,
        "successful_pose_or_trajectory_used": False,
    }
    diagnostics.write_text(
        json.dumps(
            diagnostic, indent=2, ensure_ascii=False, allow_nan=False
        )
        + "\n"
    )
    print(json.dumps(diagnostic, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
