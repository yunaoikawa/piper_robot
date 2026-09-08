"""Register a newly fixed RGB-D head camera to a saved semantic scene.

Feature matches provide a wide-baseline metric seed from synchronized RGB-D.
Conservative point-cloud registration then refines only one rigid camera
transform.  Robot links and scene geometry are never deformed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from rollout.multiview_scene import (
    depth_points_and_normals,
    refine_pose_point_to_plane,
)
from rollout.scene_3d import backproject


@dataclass(frozen=True)
class RGBDRegistration:
    target_from_source: np.ndarray
    matches: int
    metric_matches: int
    inliers: int
    inlier_fraction: float
    median_residual_m: float
    p90_residual_m: float
    accepted: bool


def rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(source) < 3 or len(source) != len(target):
        raise ValueError("rigid fit requires at least three paired points")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vh = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vh[-1] *= -1.0
        rotation = vh.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    return points @ transform[:3, :3].T + transform[:3, 3]


def ransac_rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    threshold_m: float = 0.020,
    iterations: int = 1000,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(source) < 6 or len(source) != len(target):
        raise ValueError("RANSAC requires six or more metric correspondences")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(int(iterations)):
        indices = rng.choice(len(source), 3, replace=False)
        if (
            np.linalg.matrix_rank(source[indices] - source[indices].mean(0)) < 2
            or np.linalg.matrix_rank(target[indices] - target[indices].mean(0)) < 2
        ):
            continue
        candidate = rigid_transform(source[indices], target[indices])
        residual = np.linalg.norm(
            transform_points(source, candidate) - target, axis=1
        )
        inliers = residual <= float(threshold_m)
        score = (
            int(np.count_nonzero(inliers)),
            -float(np.median(residual[inliers]))
            if np.any(inliers)
            else -float("inf"),
        )
        if best is None or score > best[0]:
            best = (score, candidate, inliers)
    if best is None or np.count_nonzero(best[2]) < 3:
        raise ValueError("rigid RANSAC found no valid hypothesis")
    refined = rigid_transform(source[best[2]], target[best[2]])
    residual = np.linalg.norm(transform_points(source, refined) - target, axis=1)
    inliers = residual <= float(threshold_m)
    if np.count_nonzero(inliers) >= 3:
        refined = rigid_transform(source[inliers], target[inliers])
        residual = np.linalg.norm(
            transform_points(source, refined) - target, axis=1
        )
        inliers = residual <= float(threshold_m)
    return refined, inliers, residual


def _depth_xyz_at_rgb_points(
    points_uv: np.ndarray,
    *,
    rgb_shape: tuple[int, ...],
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_uv = np.asarray(points_uv, dtype=float).reshape(-1, 2)
    depth = np.asarray(depth_m, dtype=float)
    height, width = depth.shape
    rgb_height, rgb_width = rgb_shape[:2]
    depth_uv = np.c_[
        points_uv[:, 0] * width / rgb_width,
        points_uv[:, 1] * height / rgb_height,
    ]
    pixels = np.rint(depth_uv).astype(int)
    valid = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    z = np.full(len(points_uv), np.nan)
    valid_indices = np.flatnonzero(valid)
    z[valid_indices] = depth[pixels[valid_indices, 1], pixels[valid_indices, 0]]
    valid &= np.isfinite(z) & (z > 0.20) & (z < 2.50)
    matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    xyz = np.full((len(points_uv), 3), np.nan)
    xyz[valid, 2] = z[valid]
    xyz[valid, 0] = (
        (depth_uv[valid, 0] - matrix[0, 2]) * z[valid] / matrix[0, 0]
    )
    xyz[valid, 1] = (
        (depth_uv[valid, 1] - matrix[1, 2]) * z[valid] / matrix[1, 1]
    )
    return xyz, valid


def register_rgbd_features(
    source_rgb_bgr: np.ndarray,
    source_depth_m: np.ndarray,
    source_camera_matrix: np.ndarray,
    target_rgb_bgr: np.ndarray,
    target_depth_m: np.ndarray,
    target_camera_matrix: np.ndarray,
    *,
    maximum_features: int = 6000,
    ratio: float = 0.72,
    ransac_threshold_m: float = 0.020,
    minimum_inliers: int = 24,
    minimum_inlier_fraction: float = 0.25,
    maximum_median_residual_m: float = 0.012,
    maximum_p90_residual_m: float = 0.025,
) -> RGBDRegistration:
    """Estimate ``target_from_source`` from ORB matches with metric depth."""

    source = np.asarray(source_rgb_bgr)
    target = np.asarray(target_rgb_bgr)
    detector = cv2.ORB_create(
        nfeatures=int(maximum_features), fastThreshold=8
    )
    source_keypoints, source_descriptors = detector.detectAndCompute(
        cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), None
    )
    target_keypoints, target_descriptors = detector.detectAndCompute(
        cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), None
    )
    if source_descriptors is None or target_descriptors is None:
        raise ValueError("ORB found insufficient features")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        source_descriptors, target_descriptors, k=2
    )
    matches = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < float(ratio) * pair[1].distance
    ]
    if len(matches) < 12:
        raise ValueError("too few ORB ratio-test matches")
    source_uv = np.asarray(
        [source_keypoints[item.queryIdx].pt for item in matches], dtype=float
    )
    target_uv = np.asarray(
        [target_keypoints[item.trainIdx].pt for item in matches], dtype=float
    )
    source_xyz, source_valid = _depth_xyz_at_rgb_points(
        source_uv,
        rgb_shape=source.shape,
        depth_m=source_depth_m,
        camera_matrix=source_camera_matrix,
    )
    target_xyz, target_valid = _depth_xyz_at_rgb_points(
        target_uv,
        rgb_shape=target.shape,
        depth_m=target_depth_m,
        camera_matrix=target_camera_matrix,
    )
    valid = source_valid & target_valid
    if np.count_nonzero(valid) < 6:
        raise ValueError("too few feature matches have metric depth")
    transform, inliers, residual = ransac_rigid_transform(
        source_xyz[valid],
        target_xyz[valid],
        threshold_m=ransac_threshold_m,
    )
    inlier_residual = residual[inliers]
    count = int(np.count_nonzero(inliers))
    fraction = float(count / np.count_nonzero(valid))
    median = float(np.median(inlier_residual))
    p90 = float(np.percentile(inlier_residual, 90))
    accepted = bool(
        count >= int(minimum_inliers)
        and fraction >= float(minimum_inlier_fraction)
        and median <= float(maximum_median_residual_m)
        and p90 <= float(maximum_p90_residual_m)
    )
    return RGBDRegistration(
        target_from_source=transform,
        matches=len(matches),
        metric_matches=int(np.count_nonzero(valid)),
        inliers=count,
        inlier_fraction=fraction,
        median_residual_m=median,
        p90_residual_m=p90,
        accepted=accepted,
    )


def refine_against_reference_rgbd(
    source_depth_m: np.ndarray,
    source_camera_matrix: np.ndarray,
    target_depth_m: np.ndarray,
    target_camera_matrix: np.ndarray,
    target_from_source_seed: np.ndarray,
    *,
    source_dynamic_mask: np.ndarray | None = None,
    target_dynamic_mask: np.ndarray | None = None,
    stride: int = 3,
    maximum_seed_correction_m: float = 0.030,
    maximum_seed_correction_deg: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Refine one camera transform using only non-dynamic RGB-D surfaces."""

    def static_mask(depth, dynamic):
        if dynamic is None:
            return None
        dynamic = np.asarray(dynamic, dtype=np.uint8)
        if dynamic.shape != np.asarray(depth).shape:
            dynamic = cv2.resize(
                dynamic,
                np.asarray(depth).shape[::-1],
                interpolation=cv2.INTER_NEAREST,
            )
        return dynamic == 0

    source_points, _ = depth_points_and_normals(
        source_depth_m,
        source_camera_matrix,
        valid_mask=static_mask(source_depth_m, source_dynamic_mask),
        stride=stride,
    )
    target_points, target_normals = depth_points_and_normals(
        target_depth_m,
        target_camera_matrix,
        valid_mask=static_mask(target_depth_m, target_dynamic_mask),
        stride=stride,
    )
    result = refine_pose_point_to_plane(
        source_points,
        target_from_source_seed,
        target_points,
        target_normals,
        maximum_correspondence_m=0.050,
        maximum_iterations=15,
        acceptance_median_m=0.010,
        acceptance_p90_m=0.025,
        acceptance_overlap=0.30,
    )
    correction = (
        np.asarray(result.reference_from_camera, dtype=float)
        @ np.linalg.inv(np.asarray(target_from_source_seed, dtype=float))
    )
    translation = float(np.linalg.norm(correction[:3, 3]))
    cosine = np.clip((np.trace(correction[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    accepted = bool(
        result.accepted
        and translation <= maximum_seed_correction_m
        and rotation_deg <= maximum_seed_correction_deg
    )
    return np.asarray(result.reference_from_camera, dtype=float), {
        "accepted": accepted,
        "median_residual_m": result.median_residual_m,
        "p90_residual_m": result.p90_residual_m,
        "overlap_fraction": result.overlap_fraction,
        "iterations": result.iterations,
        "seed_correction_translation_m": translation,
        "seed_correction_rotation_deg": rotation_deg,
        "maximum_seed_correction_m": maximum_seed_correction_m,
        "maximum_seed_correction_deg": maximum_seed_correction_deg,
        "dynamic_masks_applied": bool(
            source_dynamic_mask is not None and target_dynamic_mask is not None
        ),
    }


def validate_against_static_scene(
    source_depth_m: np.ndarray,
    source_camera_matrix: np.ndarray,
    level_from_source: np.ndarray,
    static_scene_points_level: np.ndarray,
    *,
    source_dynamic_mask: np.ndarray | None = None,
    stride: int = 4,
    maximum_correspondence_m: float = 0.040,
    minimum_inlier_fraction: float = 0.55,
    maximum_median_inlier_m: float = 0.020,
    maximum_p90_inlier_m: float = 0.035,
) -> dict:
    """Validate scene overlap without allowing ICP to move the camera."""

    depth = np.asarray(source_depth_m, dtype=float)
    valid = np.isfinite(depth) & (depth >= 0.20) & (depth <= 2.50)
    if source_dynamic_mask is not None:
        dynamic = np.asarray(source_dynamic_mask, dtype=np.uint8)
        if dynamic.shape != depth.shape:
            dynamic = cv2.resize(
                dynamic,
                depth.shape[::-1],
                interpolation=cv2.INTER_NEAREST,
            )
        valid &= dynamic == 0
    sample = np.zeros(depth.shape, dtype=bool)
    sample[:: int(stride), :: int(stride)] = True
    source = transform_points(
        backproject(depth, source_camera_matrix)[valid & sample],
        level_from_source,
    )
    target = np.asarray(static_scene_points_level, dtype=float).reshape(-1, 3)
    source = source[np.all(np.isfinite(source), axis=1)]
    target = target[np.all(np.isfinite(target), axis=1)]
    if len(source) < 80 or len(target) < 80:
        raise ValueError("too few static points for scene validation")
    distances, _ = cKDTree(target).query(source)
    inliers = distances <= float(maximum_correspondence_m)
    if np.count_nonzero(inliers) < 50:
        median = p90 = float("inf")
    else:
        median = float(np.median(distances[inliers]))
        p90 = float(np.percentile(distances[inliers], 90))
    fraction = float(np.mean(inliers))
    accepted = bool(
        fraction >= minimum_inlier_fraction
        and median <= maximum_median_inlier_m
        and p90 <= maximum_p90_inlier_m
    )
    return {
        "accepted": accepted,
        "points": int(len(source)),
        "inlier_fraction": fraction,
        "median_inlier_m": median,
        "p90_inlier_m": p90,
        "maximum_correspondence_m": maximum_correspondence_m,
        "minimum_inlier_fraction": minimum_inlier_fraction,
        "maximum_median_inlier_m": maximum_median_inlier_m,
        "maximum_p90_inlier_m": maximum_p90_inlier_m,
        "dynamic_mask_applied": source_dynamic_mask is not None,
    }
