"""Tag-free multiview RGB-D registration and conservative scene fusion.

The module operates entirely on saved observations.  Record3D poses provide
the relative-pose seed, SAM labels select static registration surfaces, and a
small point-to-plane refinement removes residual tracking drift.  It never
sends a robot command and it never promotes a visually plausible map to robot
motion authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree

from rollout.scene_3d import backproject
from rollout.scene_semantics import (
    LABEL_BACKGROUND,
    LABEL_FREE,
    LABEL_UNKNOWN,
)
from rollout.scene_volume import SceneVolume, TriangleMesh, VoxelGrid


@dataclass(frozen=True)
class MultiviewFrame:
    name: str
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    confidence: np.ndarray
    camera_matrix: np.ndarray
    reference_from_camera: np.ndarray
    semantic_labels: np.ndarray


@dataclass(frozen=True)
class PoseRefinement:
    reference_from_camera: np.ndarray
    median_residual_m: float
    p90_residual_m: float
    overlap_fraction: float
    iterations: int
    accepted: bool
    reasons: tuple[str, ...]


def quaternion_xyzw_to_rotation(quaternion) -> np.ndarray:
    """Return a normalized right-handed rotation matrix."""

    x, y, z, w = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm((x, y, z, w)))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("camera quaternion is invalid")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def record3d_pose_matrix(pose: dict) -> np.ndarray:
    """Decode Record3D's camera pose as session-from-camera."""

    translation = np.asarray(pose["translation_xyz_m"], dtype=float)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("camera translation must contain three finite values")
    result = np.eye(4)
    result[:3, :3] = quaternion_xyzw_to_rotation(pose["quaternion_xyzw"])
    result[:3, 3] = translation
    return result


def normalize_record3d_poses(poses: list[dict]) -> list[np.ndarray]:
    """Express all camera poses in the first camera frame."""

    if not poses:
        raise ValueError("at least one Record3D pose is required")
    session_from_camera = [record3d_pose_matrix(item) for item in poses]
    reference_from_session = np.linalg.inv(session_from_camera[0])
    return [reference_from_session @ item for item in session_from_camera]


def gravity_level_transform(first_record3d_pose: dict) -> np.ndarray:
    """Map the first-camera frame into a gravity-levelled right-handed frame.

    ARKit/Record3D session Y is gravity-up.  The first camera X axis supplies
    the horizontal heading, so this removes tilt without inventing a mirror.
    """

    session_from_camera = record3d_pose_matrix(first_record3d_pose)
    reference_from_session = np.linalg.inv(session_from_camera)
    up = reference_from_session[:3, :3] @ np.array([0.0, 1.0, 0.0])
    up /= np.linalg.norm(up)
    right_hint = np.array([1.0, 0.0, 0.0])
    right = right_hint - up * float(right_hint @ up)
    if np.linalg.norm(right) < 1e-6:
        right_hint = np.array([0.0, 1.0, 0.0])
        right = right_hint - up * float(right_hint @ up)
    right /= np.linalg.norm(right)
    forward = np.cross(up, right)
    forward /= np.linalg.norm(forward)
    level_from_reference = np.eye(4)
    level_from_reference[:3, :3] = np.stack((right, forward, up), axis=0)
    if np.linalg.det(level_from_reference[:3, :3]) < 0.999:
        raise ValueError("gravity-level rotation is not right-handed")
    return level_from_reference


def camera_pose_stability(poses: list[dict]) -> dict:
    """Measure motion within a stopped-view burst."""

    matrices = [record3d_pose_matrix(item) for item in poses]
    if not matrices:
        raise ValueError("pose burst is empty")
    center = matrices[len(matrices) // 2]
    inverse_center = np.linalg.inv(center)
    translations = []
    rotations = []
    for matrix in matrices:
        delta = inverse_center @ matrix
        translations.append(float(np.linalg.norm(delta[:3, 3])))
        cosine = np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rotations.append(float(np.degrees(np.arccos(cosine))))
    return {
        "maximum_translation_m": max(translations),
        "maximum_rotation_deg": max(rotations),
        "accepted": bool(
            max(translations) <= 0.003 and max(rotations) <= 1.0
        ),
    }


def _rotation_vector_matrix(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        skew = np.array(
            [
                [0.0, -vector[2], vector[1]],
                [vector[2], 0.0, -vector[0]],
                [-vector[1], vector[0], 0.0],
            ]
        )
        return np.eye(3) + skew
    axis = vector / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (
        skew @ skew
    )


def depth_points_and_normals(
    depth_m,
    camera_matrix,
    *,
    valid_mask=None,
    stride: int = 4,
    min_depth_m: float = 0.20,
    max_depth_m: float = 2.00,
    maximum_depth_step_m: float = 0.04,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a strided organized cloud with image-neighbour normals."""

    depth = np.asarray(depth_m, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError("valid mask and depth shapes differ")
        valid &= mask
    points = backproject(depth, matrix)
    horizontal = points[:, 2:] - points[:, :-2]
    vertical = points[2:, :] - points[:-2, :]
    normal_image = np.full_like(points, np.nan)
    normals = np.cross(
        horizontal[1:-1],
        vertical[:, 1:-1],
    )
    norm = np.linalg.norm(normals, axis=2)
    neighbor_valid = (
        valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & (np.abs(depth[1:-1, 2:] - depth[1:-1, :-2]) <= maximum_depth_step_m)
        & (np.abs(depth[2:, 1:-1] - depth[:-2, 1:-1]) <= maximum_depth_step_m)
        & (norm > 1e-9)
    )
    normalized = np.full_like(normals, np.nan)
    normalized[neighbor_valid] = (
        normals[neighbor_valid] / norm[neighbor_valid, None]
    )
    normal_image[1:-1, 1:-1] = normalized
    sample = np.zeros(depth.shape, dtype=bool)
    sample[1:-1:stride, 1:-1:stride] = True
    sample &= valid & np.all(np.isfinite(normal_image), axis=2)
    return points[sample], normal_image[sample]


def transform_points_normals(
    points, normals, transform
) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (
        np.asarray(points, dtype=float) @ rotation.T + translation,
        np.asarray(normals, dtype=float) @ rotation.T,
    )


def refine_pose_point_to_plane(
    source_points_camera,
    initial_reference_from_camera,
    target_points_reference,
    target_normals_reference,
    *,
    maximum_correspondence_m: float = 0.06,
    maximum_iterations: int = 15,
    acceptance_median_m: float = 0.010,
    acceptance_p90_m: float = 0.025,
    acceptance_overlap: float = 0.30,
) -> PoseRefinement:
    """Refine one shared camera pose without deforming scene geometry."""

    source = np.asarray(source_points_camera, dtype=float).reshape(-1, 3)
    target = np.asarray(target_points_reference, dtype=float).reshape(-1, 3)
    target_normals = np.asarray(target_normals_reference, dtype=float).reshape(
        -1, 3
    )
    if len(source) < 80 or len(target) < 80 or len(target) != len(target_normals):
        raise ValueError("too few points for multiview pose refinement")
    if len(source) > 12000:
        source = source[np.linspace(0, len(source) - 1, 12000).astype(int)]
    if len(target) > 30000:
        indices = np.linspace(0, len(target) - 1, 30000).astype(int)
        target = target[indices]
        target_normals = target_normals[indices]
    tree = cKDTree(target)
    transform = np.asarray(initial_reference_from_camera, dtype=float).reshape(
        4, 4
    ).copy()
    completed = 0

    for iteration in range(maximum_iterations):
        moved = source @ transform[:3, :3].T + transform[:3, 3]
        distance, indices = tree.query(moved, k=1)
        inliers = np.isfinite(distance) & (distance <= maximum_correspondence_m)
        if np.count_nonzero(inliers) < 60:
            break
        point = moved[inliers]
        match = target[indices[inliers]]
        normal = target_normals[indices[inliers]]
        residual = np.sum((point - match) * normal, axis=1)
        trim = np.abs(residual) <= np.percentile(np.abs(residual), 85.0)
        point = point[trim]
        normal = normal[trim]
        residual = residual[trim]
        if len(residual) < 50:
            break
        scale = max(float(np.median(np.abs(residual))) * 2.5, 0.002)
        weight = 1.0 / np.sqrt(1.0 + (residual / scale) ** 2)
        system = np.column_stack((np.cross(point, normal), normal))
        weighted_system = system * weight[:, None]
        update, *_ = np.linalg.lstsq(
            weighted_system,
            -residual * weight,
            rcond=None,
        )
        rotation_update = update[:3]
        translation_update = update[3:]
        rotation_norm = float(np.linalg.norm(rotation_update))
        translation_norm = float(np.linalg.norm(translation_update))
        if rotation_norm > np.deg2rad(3.0):
            rotation_update *= np.deg2rad(3.0) / rotation_norm
        if translation_norm > 0.02:
            translation_update *= 0.02 / translation_norm
        delta = np.eye(4)
        delta[:3, :3] = _rotation_vector_matrix(rotation_update)
        delta[:3, 3] = translation_update
        transform = delta @ transform
        completed = iteration + 1
        if (
            np.linalg.norm(rotation_update) < 2e-5
            and np.linalg.norm(translation_update) < 2e-5
        ):
            break

    moved = source @ transform[:3, :3].T + transform[:3, 3]
    distance, indices = tree.query(moved, k=1)
    inliers = np.isfinite(distance) & (distance <= maximum_correspondence_m)
    overlap = float(np.mean(inliers))
    if np.count_nonzero(inliers):
        plane_residual = np.abs(
            np.sum(
                (moved[inliers] - target[indices[inliers]])
                * target_normals[indices[inliers]],
                axis=1,
            )
        )
        median = float(np.median(plane_residual))
        p90 = float(np.percentile(plane_residual, 90.0))
    else:
        median = float("inf")
        p90 = float("inf")
    reasons = []
    if overlap < acceptance_overlap:
        reasons.append(
            f"overlap {overlap:.3f} < {acceptance_overlap:.3f}"
        )
    if median > acceptance_median_m:
        reasons.append(
            f"median residual {median:.4f}m > {acceptance_median_m:.4f}m"
        )
    if p90 > acceptance_p90_m:
        reasons.append(
            f"p90 residual {p90:.4f}m > {acceptance_p90_m:.4f}m"
        )
    return PoseRefinement(
        reference_from_camera=transform,
        median_residual_m=median,
        p90_residual_m=p90,
        overlap_fraction=overlap,
        iterations=completed,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def automatic_world_grid(
    point_sets,
    *,
    voxel_size_m: float = 0.005,
    truncation_m: float = 0.020,
    percentile: float = 0.5,
    margin_m: float = 0.03,
    maximum_voxels: int = 24_000_000,
) -> VoxelGrid:
    finite_sets = []
    for points in point_sets:
        values = np.asarray(points, dtype=float).reshape(-1, 3)
        values = values[np.all(np.isfinite(values), axis=1)]
        if len(values):
            finite_sets.append(values)
    if not finite_sets:
        raise ValueError("no finite multiview points")
    points = np.vstack(finite_sets)
    lower = np.percentile(points, percentile, axis=0) - margin_m
    upper = (
        np.percentile(points, 100.0 - percentile, axis=0)
        + margin_m
        + truncation_m
    )
    origin = np.floor(lower / voxel_size_m) * voxel_size_m
    shape_xyz = np.ceil((upper - origin) / voxel_size_m).astype(int)
    if np.any(shape_xyz < 2):
        raise ValueError("multiview grid is degenerate")
    voxel_count = int(np.prod(shape_xyz, dtype=np.int64))
    if voxel_count > int(maximum_voxels):
        raise ValueError(
            f"multiview grid has {voxel_count} voxels, limit is "
            f"{int(maximum_voxels)}; crop the capture or increase voxel size"
        )
    return VoxelGrid(
        origin_xyz_m=origin,
        voxel_size_m=float(voxel_size_m),
        shape_zyx=tuple(int(value) for value in shape_xyz[::-1]),
    )


def integrate_multiview_projective_depth(
    frames: list[MultiviewFrame],
    grid: VoxelGrid,
    *,
    truncation_m: float = 0.020,
    min_depth_m: float = 0.20,
    max_depth_m: float = 2.00,
    minimum_confidence: int = 1,
    dynamic_label_ids=(),
) -> SceneVolume:
    """Fuse arbitrary camera poses while preserving unobserved space."""

    if not frames:
        raise ValueError("at least one multiview frame is required")
    nz, ny, nx = grid.shape_zyx
    x, y, z = grid.centers()
    xx, yy = np.meshgrid(x, y)
    tsdf_sum = np.zeros((nz, ny, nx), dtype=np.float32)
    weight_sum = np.zeros((nz, ny, nx), dtype=np.float32)
    semantic_weight = np.zeros((nz, ny, nx), dtype=np.float32)
    semantics = np.full((nz, ny, nx), LABEL_UNKNOWN, dtype=np.uint8)

    for frame in frames:
        depth = np.asarray(frame.depth_m, dtype=float)
        confidence = np.asarray(frame.confidence)
        labels = np.asarray(frame.semantic_labels, dtype=np.uint8)
        if confidence.shape != depth.shape or labels.shape != depth.shape:
            raise ValueError(f"{frame.name}: depth/confidence/label shapes differ")
        valid_depth = (
            np.isfinite(depth)
            & (depth >= min_depth_m)
            & (depth <= max_depth_m)
            & (confidence >= minimum_confidence)
        )
        matrix = np.asarray(frame.camera_matrix, dtype=float).reshape(3, 3)
        reference_from_camera = np.asarray(
            frame.reference_from_camera, dtype=float
        ).reshape(4, 4)
        camera_from_reference = np.linalg.inv(reference_from_camera)
        rotation = camera_from_reference[:3, :3]
        translation = camera_from_reference[:3, 3]
        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
        for iz, z_value in enumerate(z):
            world = np.column_stack(
                (
                    xx.reshape(-1),
                    yy.reshape(-1),
                    np.full(xx.size, z_value),
                )
            )
            camera = world @ rotation.T + translation
            camera_z = camera[:, 2]
            projected = camera_z > 1e-6
            u = np.zeros(len(camera), dtype=np.int32)
            v = np.zeros(len(camera), dtype=np.int32)
            u[projected] = np.rint(
                fx * camera[projected, 0] / camera_z[projected] + cx
            ).astype(np.int32)
            v[projected] = np.rint(
                fy * camera[projected, 1] / camera_z[projected] + cy
            ).astype(np.int32)
            inside = (
                projected
                & (u >= 0)
                & (u < depth.shape[1])
                & (v >= 0)
                & (v < depth.shape[0])
            )
            sampled_depth = np.full(len(camera), np.nan, dtype=float)
            sampled_confidence = np.zeros(len(camera), dtype=float)
            sampled_labels = np.full(
                len(camera), LABEL_BACKGROUND, dtype=np.uint8
            )
            sampled_valid = np.zeros(len(camera), dtype=bool)
            sampled_depth[inside] = depth[v[inside], u[inside]]
            sampled_confidence[inside] = confidence[v[inside], u[inside]]
            sampled_labels[inside] = labels[v[inside], u[inside]]
            sampled_valid[inside] = valid_depth[v[inside], u[inside]]
            signed = sampled_depth - camera_z
            observed = sampled_valid & (signed >= -truncation_m)
            if not np.any(observed):
                continue
            ray_norm = np.linalg.norm(camera, axis=1)
            view_cosine = np.zeros(len(camera), dtype=float)
            view_cosine[projected] = np.clip(
                camera_z[projected] / ray_norm[projected], 0.20, 1.0
            )
            weights = (
                np.clip(sampled_confidence / 2.0, 0.0, 1.0)
                * view_cosine**2
            )
            weights[~observed] = 0.0
            normalized = np.clip(signed / truncation_m, -1.0, 1.0)
            layer_tsdf = tsdf_sum[iz].reshape(-1)
            layer_weight = weight_sum[iz].reshape(-1)
            layer_tsdf[observed] += (
                normalized[observed] * weights[observed]
            ).astype(np.float32)
            layer_weight[observed] += weights[observed].astype(np.float32)
            surface = observed & (np.abs(signed) <= truncation_m)
            semantic_update = surface & (
                weights >= semantic_weight[iz].reshape(-1)
            )
            semantics[iz].reshape(-1)[semantic_update] = sampled_labels[
                semantic_update
            ]
            semantic_weight[iz].reshape(-1)[semantic_update] = weights[
                semantic_update
            ]

    observed = weight_sum > 0.0
    tsdf = np.ones_like(tsdf_sum)
    tsdf[observed] = tsdf_sum[observed] / weight_sum[observed]
    free = observed & (tsdf > 0.0)
    occupied = observed & ~free
    semantics[free] = LABEL_FREE
    semantics[occupied & (semantics == LABEL_UNKNOWN)] = LABEL_BACKGROUND
    dynamic = np.isin(semantics, tuple(int(v) for v in dynamic_label_ids))
    static_occupied = occupied & ~dynamic
    esdf = np.full(tsdf.shape, np.nan, dtype=np.float32)
    if np.any(static_occupied):
        positive = distance_transform_edt(
            ~static_occupied, sampling=grid.voxel_size_m
        )
        esdf[free] = positive[free]
    if np.any(free):
        negative = distance_transform_edt(
            ~free, sampling=grid.voxel_size_m
        )
        esdf[static_occupied] = -negative[static_occupied]
    return SceneVolume(grid, tsdf, observed, esdf, semantics)


def merge_triangle_meshes(meshes: list[TriangleMesh]) -> TriangleMesh:
    if not meshes:
        raise ValueError("no measured meshes to merge")
    vertices = []
    faces = []
    colors = []
    labels = []
    offset = 0
    for mesh in meshes:
        vertices.append(np.asarray(mesh.vertices_xyz_m))
        faces.append(np.asarray(mesh.faces, dtype=np.int32) + offset)
        colors.append(np.asarray(mesh.colors_rgb, dtype=np.uint8))
        if mesh.semantic_labels is not None:
            labels.append(np.asarray(mesh.semantic_labels, dtype=np.uint8))
        offset += len(mesh.vertices_xyz_m)
    return TriangleMesh(
        np.vstack(vertices),
        np.vstack(faces) if faces else np.empty((0, 3), dtype=np.int32),
        np.vstack(colors),
        np.concatenate(labels) if len(labels) == len(meshes) else None,
    )


def dominant_support_height(
    points_reference,
    normals_reference,
    up_reference,
    *,
    maximum_tilt_deg: float = 15.0,
    bin_size_m: float = 0.006,
) -> float | None:
    """Estimate the dominant horizontal support level in one accepted view."""

    points = np.asarray(points_reference, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals_reference, dtype=float).reshape(-1, 3)
    up = np.asarray(up_reference, dtype=float).reshape(3)
    up /= np.linalg.norm(up)
    horizontal = np.abs(normals @ up) >= np.cos(np.deg2rad(maximum_tilt_deg))
    if np.count_nonzero(horizontal) < 80:
        return None
    heights = points[horizontal] @ up
    bins = np.floor(heights / bin_size_m).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    winning = unique[np.argmax(counts)]
    selected = heights[bins == winning]
    return float(np.median(selected))


def support_height_modes(
    points_reference,
    normals_reference,
    up_reference=(0.0, 0.0, 1.0),
    *,
    maximum_tilt_deg: float = 15.0,
    bin_size_m: float = 0.006,
    minimum_points: int = 60,
    maximum_modes: int = 6,
) -> list[dict]:
    """Return separated horizontal support levels instead of one forced plane."""

    points = np.asarray(points_reference, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals_reference, dtype=float).reshape(-1, 3)
    up = np.asarray(up_reference, dtype=float).reshape(3)
    up /= np.linalg.norm(up)
    horizontal = np.abs(normals @ up) >= np.cos(np.deg2rad(maximum_tilt_deg))
    if np.count_nonzero(horizontal) < minimum_points:
        return []
    heights = points[horizontal] @ up
    bins = np.floor(heights / bin_size_m).astype(np.int64)
    unique, counts = np.unique(bins, return_counts=True)
    candidates = sorted(
        zip(unique.tolist(), counts.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    selected = []
    for bin_index, count in candidates:
        if count < minimum_points:
            continue
        values = heights[bins == bin_index]
        height = float(np.median(values))
        if any(
            abs(height - item["height_m"]) < 2.0 * bin_size_m
            for item in selected
        ):
            continue
        selected.append({"height_m": height, "points": int(count)})
        if len(selected) >= maximum_modes:
            break
    return selected


def unknown_frontier(volume: SceneVolume) -> np.ndarray:
    return volume.unknown & binary_dilation(volume.free, iterations=1)
