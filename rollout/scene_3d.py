"""AprilTag-free geometry for Record3D manipulation scenes.

The transparent object mask is used only to select a camera ray.  Its depth is
obtained from the locally fitted support surface, because LiDAR commonly
returns the bench below a transparent Petri-dish lid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PlaneModel:
    normal: np.ndarray
    offset: float
    inlier_fraction: float
    rms_m: float


@dataclass(frozen=True)
class TargetEstimate:
    pixel_xy: np.ndarray
    point_camera_xyz_m: np.ndarray
    plane: PlaneModel
    support_sample_count: int
    confidence: float


@dataclass(frozen=True)
class RegistrationResult:
    live_to_reference: np.ndarray
    rmse_m: float
    inlier_fraction: float
    iterations: int
    accepted: bool


def temporal_median_depth(
    frames,
    *,
    rotate_clockwise: bool = False,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
) -> np.ndarray:
    """Combine fresh depth frames while rejecting invalid sensor samples."""

    prepared = []
    for frame in frames:
        depth = np.asarray(frame, dtype=float)
        if depth.ndim != 2:
            raise ValueError("depth frames must be two-dimensional")
        if rotate_clockwise:
            depth = np.rot90(depth, k=3)
        depth = depth.copy()
        invalid = (
            ~np.isfinite(depth)
            | (depth <= min_depth_m)
            | (depth >= max_depth_m)
        )
        depth[invalid] = np.nan
        prepared.append(depth)
    if not prepared:
        raise ValueError("at least one depth frame is required")
    if any(frame.shape != prepared[0].shape for frame in prepared):
        raise ValueError("depth frames have different shapes")
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(prepared, axis=0), axis=0)


def scaled_camera_matrix(matrix, source_shape, target_shape) -> np.ndarray:
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    out = np.asarray(matrix, dtype=float).reshape(3, 3).copy()
    out[0, :] *= target_width / source_width
    out[1, :] *= target_height / source_height
    out[2, :] = (0.0, 0.0, 1.0)
    return out


def backproject(depth_m, camera_matrix) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    z = depth
    x = (xx - matrix[0, 2]) * z / matrix[0, 0]
    y = (yy - matrix[1, 2]) * z / matrix[1, 1]
    return np.stack((x, y, z), axis=-1)


def fit_plane_ransac(
    points,
    *,
    threshold_m: float = 0.006,
    iterations: int = 180,
    seed: int = 0,
) -> PlaneModel:
    """Fit a plane with deterministic RANSAC followed by an SVD refinement."""

    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if len(xyz) < 30:
        raise ValueError("too few finite points for support-plane fit")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        sample = xyz[rng.choice(len(xyz), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        residual = np.abs(xyz @ normal + offset)
        inliers = residual <= threshold_m
        score = (int(np.count_nonzero(inliers)), -float(np.median(residual)))
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or np.count_nonzero(best[1]) < 20:
        raise ValueError("support-plane RANSAC failed")
    inlier_points = xyz[best[1]]
    center = inlier_points.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_points - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] > 0:
        normal = -normal
    offset = -float(normal @ center)
    residual = np.abs(inlier_points @ normal + offset)
    return PlaneModel(
        normal=normal,
        offset=offset,
        inlier_fraction=float(len(inlier_points) / len(xyz)),
        rms_m=float(np.sqrt(np.mean(residual**2))),
    )


def ray_plane_intersection(pixel_xy, camera_matrix, plane: PlaneModel):
    matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    pixel = np.asarray(pixel_xy, dtype=float).reshape(2)
    ray = np.linalg.inv(matrix) @ np.array([pixel[0], pixel[1], 1.0])
    denominator = float(plane.normal @ ray)
    if abs(denominator) < 1e-6:
        raise ValueError("camera ray is parallel to support plane")
    distance = -plane.offset / denominator
    if not 0.05 < distance < 5.0:
        raise ValueError("support-plane intersection is behind/outside camera")
    return ray * distance


def estimate_target_on_support_plane(
    depth_m,
    camera_matrix,
    target_mask,
    target_pixel_xy,
    *,
    ring_margin_px: int = 70,
    plane_threshold_m: float = 0.006,
) -> TargetEstimate:
    """Estimate target XYZ by fitting the bench immediately around its mask."""

    depth = np.asarray(depth_m, dtype=float)
    mask = np.asarray(target_mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError("depth and target mask shapes differ")
    if np.count_nonzero(mask) < 20:
        raise ValueError("target mask is too small")
    kernel_size = max(3, 2 * int(ring_margin_px) + 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    outer = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inner = cv2.dilate(mask.astype(np.uint8), inner_kernel).astype(bool)
    ring = outer & ~inner & np.isfinite(depth)
    ring &= (depth > 0.05) & (depth < 5.0)
    if np.count_nonzero(ring) < 60:
        raise ValueError("insufficient support surface around target")

    # Suppress foreground/background clutter before RANSAC.  The local median
    # is normally the bench even when the mask itself contains transparent-ray
    # returns from that same surface.
    values = depth[ring]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    band = max(0.035, 4.0 * 1.4826 * mad)
    ring &= np.abs(depth - median) <= band
    points = backproject(depth, camera_matrix)[ring]
    plane = fit_plane_ransac(
        points, threshold_m=plane_threshold_m, iterations=220
    )
    point = ray_plane_intersection(target_pixel_xy, camera_matrix, plane)
    coverage = min(1.0, len(points) / 1200.0)
    residual_score = float(np.exp(-plane.rms_m / 0.006))
    confidence = coverage * plane.inlier_fraction * residual_score
    return TargetEstimate(
        pixel_xy=np.asarray(target_pixel_xy, dtype=float),
        point_camera_xyz_m=point,
        plane=plane,
        support_sample_count=int(len(points)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def voxel_downsample(points, voxel_m: float = 0.008) -> np.ndarray:
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if not len(xyz):
        return xyz
    keys = np.floor(xyz / float(voxel_m)).astype(np.int64)
    _, unique = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(unique)]


def _nearest(reference, query, chunk_size=256):
    indices = []
    distances = []
    for start in range(0, len(query), chunk_size):
        chunk = query[start : start + chunk_size]
        squared = np.sum(
            (chunk[:, None, :] - reference[None, :, :]) ** 2, axis=2
        )
        index = np.argmin(squared, axis=1)
        indices.append(index)
        distances.append(np.sqrt(squared[np.arange(len(chunk)), index]))
    return np.concatenate(indices), np.concatenate(distances)


def _rigid_transform(source, target):
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if np.linalg.det(rotation) < 0:
        vh[-1] *= -1
        rotation = vh.T @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def register_point_clouds(
    live_points,
    reference_points,
    *,
    max_correspondence_m: float = 0.035,
    max_iterations: int = 20,
    acceptance_rmse_m: float = 0.012,
    acceptance_inlier_fraction: float = 0.55,
) -> RegistrationResult:
    """Small, dependency-free point-to-point ICP for camera-shift recovery."""

    live = voxel_downsample(live_points)
    reference = voxel_downsample(reference_points)
    if len(live) < 50 or len(reference) < 50:
        raise ValueError("too few points for scene registration")
    # Bound the quadratic matcher on dense Record3D clouds.
    live = live[:6000]
    reference = reference[:6000]
    transform = np.eye(4)
    previous_rmse = np.inf
    inlier_fraction = 0.0
    completed = 0
    for iteration in range(max_iterations):
        moved = live @ transform[:3, :3].T + transform[:3, 3]
        indices, distance = _nearest(reference, moved)
        inliers = distance <= max_correspondence_m
        if np.count_nonzero(inliers) < 30:
            break
        update = _rigid_transform(moved[inliers], reference[indices[inliers]])
        transform = update @ transform
        rmse = float(np.sqrt(np.mean(distance[inliers] ** 2)))
        inlier_fraction = float(np.count_nonzero(inliers) / len(live))
        completed = iteration + 1
        if abs(previous_rmse - rmse) < 1e-5:
            previous_rmse = rmse
            break
        previous_rmse = rmse
    accepted = (
        np.isfinite(previous_rmse)
        and previous_rmse <= acceptance_rmse_m
        and inlier_fraction >= acceptance_inlier_fraction
    )
    return RegistrationResult(
        live_to_reference=transform,
        rmse_m=float(previous_rmse),
        inlier_fraction=inlier_fraction,
        iterations=completed,
        accepted=bool(accepted),
    )


def nearest_scene_distance(point_xyz, scene_points) -> float:
    point = np.asarray(point_xyz, dtype=float).reshape(1, 3)
    scene = np.asarray(scene_points, dtype=float).reshape(-1, 3)
    scene = scene[np.all(np.isfinite(scene), axis=1)]
    if not len(scene):
        return float("inf")
    return float(np.sqrt(np.min(np.sum((scene - point) ** 2, axis=1))))
