"""Rigid scene registration helpers for semantic MuJoCo reconstruction.

The functions in this module are observation-only.  They connect two RGB-D
sessions through a fixed metric AprilTag, estimate a shared planar correction
for a rigid pair of robot bases, and intersect image rays with measured
support planes.  No robot client is imported and no hardware command is sent.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from rollout.apriltag_retarget import (
    detect_tags,
    estimate_tag_camera_pose,
)


@dataclass(frozen=True)
class TransformConsensus:
    transform: np.ndarray
    translation_spread_m: float
    rotation_spread_deg: float
    sample_count: int


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    return points @ transform[:3, :3].T + transform[:3, 3]


def tag_pose_camera(
    image_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    tag_id: int,
    tag_size_m: float,
    family: str = "DICT_APRILTAG_36h11",
) -> tuple[np.ndarray, float]:
    """Return ``T_camera_tag`` and its four-corner reprojection RMS."""

    detections = detect_tags(image_bgr, family)
    detection = next(
        (item for item in detections if item.tag_id == int(tag_id)),
        None,
    )
    if detection is None:
        raise ValueError(
            f"fixed tag {tag_id} missing; saw "
            f"{[item.tag_id for item in detections]}"
        )
    rvec, tvec, rms = estimate_tag_camera_pose(
        detection,
        camera_matrix,
        tag_size_m,
    )
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec
    return transform, rms


def rigid_transform_consensus(
    transforms: list[np.ndarray],
) -> TransformConsensus:
    """Robustly average rigid transforms and report maximum deviations."""

    if not transforms:
        raise ValueError("at least one transform is required")
    values = np.asarray(transforms, dtype=float)
    if values.shape[1:] != (4, 4):
        raise ValueError("transforms must be 4x4")
    translations = values[:, :3, 3]
    center = np.median(translations, axis=0)
    rotation = Rotation.from_matrix(values[:, :3, :3]).mean()
    result = np.eye(4)
    result[:3, :3] = rotation.as_matrix()
    result[:3, 3] = center
    translation_spread = float(
        np.max(np.linalg.norm(translations - center, axis=1))
    )
    rotation_spread = float(
        np.degrees(
            np.max(
                (
                    Rotation.from_matrix(values[:, :3, :3])
                    * rotation.inv()
                ).magnitude()
            )
        )
    )
    return TransformConsensus(
        result,
        translation_spread,
        rotation_spread,
        len(values),
    )


def bridge_camera_from_fixed_tag(
    scene_from_tag: np.ndarray,
    camera_from_tag: np.ndarray,
) -> np.ndarray:
    """Return ``T_scene_camera`` from a tag observed in both sessions."""

    return (
        np.asarray(scene_from_tag, dtype=float).reshape(4, 4)
        @ np.linalg.inv(
            np.asarray(camera_from_tag, dtype=float).reshape(4, 4)
        )
    )


def scaled_camera_matrix(
    camera_matrix: np.ndarray,
    *,
    source_shape_hw: tuple[int, int],
    target_shape_hw: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = source_shape_hw
    target_height, target_width = target_shape_hw
    result = np.asarray(camera_matrix, dtype=float).copy()
    result[0] *= target_width / source_width
    result[1] *= target_height / source_height
    return result


def backproject_depth(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=float)
    rows, columns = np.indices(depth.shape)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    x = (columns - cx) * depth / fx
    y = (rows - cy) * depth / fy
    return np.dstack((x, y, depth))


def depth_layer_foreground_mask(
    mask_rgb: np.ndarray,
    depth_m: np.ndarray,
    confidence: np.ndarray,
    temporal_background_depth_m: np.ndarray,
    *,
    minimum_confidence: int = 1,
    maximum_neighbor_depth_jump_m: float = 0.045,
    minimum_foreground_delta_m: float = 0.025,
    minimum_component_pixels: int = 20,
    minimum_dynamic_pixels: int = 8,
) -> tuple[np.ndarray, list[dict]]:
    """Separate a foreground robot from a touching, deeper SAM false positive.

    The camera is fixed while the arms move.  A robot surface is therefore
    closer than the temporal far-depth envelope in at least part of its
    depth-connected component.  A microscope mistakenly joined to the SAM
    mask stays on the far envelope and is rejected.
    """

    depth = np.asarray(depth_m, dtype=float)
    background = np.asarray(temporal_background_depth_m, dtype=float)
    if depth.shape != background.shape:
        raise ValueError("depth and temporal background shapes differ")
    mask = cv2.resize(
        np.asarray(mask_rgb, dtype=np.uint8),
        (depth.shape[1], depth.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    valid = (
        (mask > 0)
        & np.isfinite(depth)
        & (depth > 0)
        & (np.asarray(confidence) >= minimum_confidence)
    )
    flat_indices = np.flatnonzero(valid)
    parent = np.arange(flat_indices.size, dtype=np.int32)
    rank = np.zeros(flat_indices.size, dtype=np.uint8)
    lookup = np.full(depth.size, -1, dtype=np.int32)
    lookup[flat_indices] = np.arange(flat_indices.size, dtype=np.int32)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    rows, columns = depth.shape
    for flat_index in flat_indices:
        row, column = divmod(int(flat_index), columns)
        first = int(lookup[flat_index])
        if column + 1 < columns:
            neighbor = flat_index + 1
            second = int(lookup[neighbor])
            if (
                second >= 0
                and abs(depth.flat[flat_index] - depth.flat[neighbor])
                <= maximum_neighbor_depth_jump_m
            ):
                union(first, second)
        if row + 1 < rows:
            neighbor = flat_index + columns
            second = int(lookup[neighbor])
            if (
                second >= 0
                and abs(depth.flat[flat_index] - depth.flat[neighbor])
                <= maximum_neighbor_depth_jump_m
            ):
                union(first, second)

    components = {}
    for local_index, flat_index in enumerate(flat_indices):
        components.setdefault(find(local_index), []).append(int(flat_index))
    dynamic = (
        np.isfinite(background)
        & (background - depth >= minimum_foreground_delta_m)
    )
    selected = np.zeros(depth.shape, dtype=np.uint8)
    records = []
    for indices in components.values():
        indices = np.asarray(indices, dtype=int)
        dynamic_pixels = int(np.count_nonzero(dynamic.flat[indices]))
        accepted = bool(
            len(indices) >= minimum_component_pixels
            and dynamic_pixels >= minimum_dynamic_pixels
        )
        if accepted:
            selected.flat[indices] = 255
        records.append(
            {
                "pixels": int(len(indices)),
                "dynamic_pixels": dynamic_pixels,
                "dynamic_fraction": float(dynamic_pixels / len(indices)),
                "accepted": accepted,
                "median_depth_m": float(np.median(depth.flat[indices])),
            }
        )
    records.sort(key=lambda item: item["pixels"], reverse=True)
    return selected, records


def persistent_depth_component_centers(
    point_clouds: list[np.ndarray],
    *,
    voxel_size_m: float = 0.02,
    minimum_views: int = 4,
    minimum_voxels: int = 30,
    expected_base_z_m: float | None = None,
    maximum_base_plane_gap_m: float = 0.15,
) -> tuple[list[np.ndarray], list[dict]]:
    """Find stationary 3D components inside depth-cleaned moving-arm masks."""

    if minimum_views < 2 or len(point_clouds) < minimum_views:
        raise ValueError("insufficient point clouds for persistence")
    counts = Counter()
    for points in point_clouds:
        values = np.asarray(points, dtype=float).reshape(-1, 3)
        values = values[np.all(np.isfinite(values), axis=1)]
        voxels = np.floor(values / voxel_size_m).astype(int)
        counts.update(set(map(tuple, voxels)))
    keys = {
        key for key, count in counts.items() if count >= minimum_views
    }
    parent = {key: key for key in keys}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    for key in keys:
        for offset in offsets:
            neighbor = tuple(
                value + delta for value, delta in zip(key, offset)
            )
            if neighbor in keys:
                union(key, neighbor)
    components = {}
    for key in keys:
        components.setdefault(find(key), []).append(key)
    records = []
    centers = []
    for values in components.values():
        xyz = (
            np.asarray(values, dtype=float) + 0.5
        ) * voxel_size_m
        base_plane_gap = (
            None
            if expected_base_z_m is None
            else float(abs(np.min(xyz[:, 2]) - expected_base_z_m))
        )
        accepted = bool(
            len(xyz) >= minimum_voxels
            and (
                base_plane_gap is None
                or base_plane_gap <= maximum_base_plane_gap_m
            )
        )
        record = {
            "voxel_count": int(len(xyz)),
            "accepted": accepted,
            "center_xyz_m": np.median(xyz, axis=0).tolist(),
            "minimum_xyz_m": np.min(xyz, axis=0).tolist(),
            "maximum_xyz_m": np.max(xyz, axis=0).tolist(),
            "base_plane_gap_m": base_plane_gap,
        }
        records.append(record)
        if accepted:
            centers.append(np.median(xyz, axis=0))
    records.sort(key=lambda item: item["voxel_count"], reverse=True)
    centers.sort(key=lambda value: (value[0], value[1], value[2]))
    return centers, records


def assign_independent_base_translations(
    initial_base_xyz: dict[str, np.ndarray],
    observed_base_xyz: list[np.ndarray],
) -> dict:
    """Assign two persistent components to two bases without changing yaw."""

    names = list(initial_base_xyz)
    if len(names) != 2 or len(observed_base_xyz) != 2:
        raise ValueError("exactly two initial and observed bases are required")
    initial = np.asarray([initial_base_xyz[name][:2] for name in names])
    observed = np.asarray([value[:2] for value in observed_base_xyz])
    candidates = []
    for assignment in permutations(range(2)):
        target = observed[list(assignment)]
        translations = target - initial
        score = float(np.sum(np.linalg.norm(translations, axis=1)))
        candidates.append((score, assignment, target, translations))
    score, assignment, target, translations = min(
        candidates,
        key=lambda item: item[0],
    )
    return {
        "method": "persistent_depth_independent_base_translation",
        "assignment": list(assignment),
        "score_m": score,
        "yaw_source": "reviewed_upright_model",
        "translations_xy_m": {
            name: translations[index].tolist()
            for index, name in enumerate(names)
        },
        "base_xyz_level_m": {
            name: [
                float(target[index, 0]),
                float(target[index, 1]),
                float(initial_base_xyz[name][2]),
            ]
            for index, name in enumerate(names)
        },
    }


def component_base_centers(
    mask_rgb: np.ndarray,
    depth_m: np.ndarray,
    confidence: np.ndarray,
    camera_matrix_rgb: np.ndarray,
    scene_from_camera: np.ndarray,
    *,
    expected_base_xyz: list[np.ndarray] | None = None,
    minimum_component_fraction: float = 0.01,
    minimum_confidence: int = 1,
    base_height_band_m: float = 0.09,
    maximum_anchor_distance_m: float = 0.20,
) -> list[np.ndarray]:
    """Estimate fixed base XY centers from the two largest robot instances."""

    mask = cv2.resize(
        np.asarray(mask_rgb, dtype=np.uint8),
        (depth_m.shape[1], depth_m.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    mask = mask > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    minimum_pixels = max(
        8,
        int(round(mask.size * float(minimum_component_fraction))),
    )
    components = sorted(
        (
            index
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_pixels
        ),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:2]
    if len(components) < 2:
        raise ValueError("two large robot mask components are required")
    scaled = scaled_camera_matrix(
        camera_matrix_rgb,
        source_shape_hw=mask_rgb.shape[:2],
        target_shape_hw=depth_m.shape,
    )
    camera_points = backproject_depth(depth_m, scaled)
    result = []
    for component in components:
        selected = (
            (labels == component)
            & np.isfinite(depth_m)
            & (depth_m > 0)
            & (np.asarray(confidence) >= minimum_confidence)
        )
        points = transform_points(
            camera_points[selected],
            scene_from_camera,
        )
        if len(points) < 12:
            raise ValueError("robot component has insufficient RGB-D points")
        if expected_base_xyz:
            expected = np.asarray(expected_base_xyz, dtype=float)
            costs = []
            for anchor in expected:
                distances = np.linalg.norm(
                    points[:, :2] - anchor[:2],
                    axis=1,
                )
                costs.append(float(np.quantile(distances, 0.05)))
            anchor = expected[int(np.argmin(costs))]
            base_points = points[
                (np.linalg.norm(points[:, :2] - anchor[:2], axis=1)
                 <= maximum_anchor_distance_m)
                & (np.abs(points[:, 2] - anchor[2])
                   <= base_height_band_m)
            ]
        else:
            bottom = float(np.quantile(points[:, 2], 0.05))
            base_points = points[
                (points[:, 2] >= bottom)
                & (points[:, 2] <= bottom + base_height_band_m)
            ]
        if len(base_points) < 8:
            raise ValueError("robot component has insufficient base points")
        result.append(np.median(base_points, axis=0))
    return result


def fit_shared_planar_robot_transform(
    initial_base_xyz: dict[str, np.ndarray],
    observed_base_xyz: list[np.ndarray],
) -> dict:
    """Fit one SE(2) correction while preserving the CAD base baseline."""

    names = list(initial_base_xyz)
    if len(names) != 2 or len(observed_base_xyz) != 2:
        raise ValueError("exactly two initial and observed bases are required")
    initial = np.asarray([initial_base_xyz[name][:2] for name in names])
    observed = np.asarray([value[:2] for value in observed_base_xyz])
    initial_center = initial.mean(axis=0)
    initial_vector = initial[1] - initial[0]
    candidates = []
    for assignment in permutations(range(2)):
        target = observed[list(assignment)]
        target_center = target.mean(axis=0)
        target_vector = target[1] - target[0]
        yaw = float(
            np.arctan2(target_vector[1], target_vector[0])
            - np.arctan2(initial_vector[1], initial_vector[0])
        )
        yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        translation = target_center - rotation @ initial_center
        fitted = initial @ rotation.T + translation
        residuals = np.linalg.norm(fitted - target, axis=1)
        baseline_error = abs(
            np.linalg.norm(target_vector) - np.linalg.norm(initial_vector)
        )
        score = float(np.sqrt(np.mean(residuals**2)) + baseline_error)
        candidates.append(
            (score, assignment, yaw, translation, fitted, residuals)
        )
    score, assignment, yaw, translation, fitted, residuals = min(
        candidates,
        key=lambda item: item[0],
    )
    return {
        "method": "shared_planar_rigid_pair_fit",
        "assignment": list(assignment),
        "translation_xy_m": translation.tolist(),
        "yaw_delta_rad": yaw,
        "rms_residual_m": float(np.sqrt(np.mean(residuals**2))),
        "score_m": score,
        "base_xyz_level_m": {
            name: [
                float(fitted[index, 0]),
                float(fitted[index, 1]),
                float(initial_base_xyz[name][2]),
            ]
            for index, name in enumerate(names)
        },
    }


def apply_shared_planar_transform_to_mjcf(
    source: str | Path,
    output: str | Path,
    *,
    root_bodies: tuple[str, str],
    translation_xy_m: tuple[float, float],
    yaw_delta_rad: float,
) -> None:
    """Write a derived MJCF with a rigid correction applied to both roots."""

    source = Path(source).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir"))
        if not meshdir.is_absolute():
            compiler.set(
                "meshdir",
                str((source.parent / meshdir).resolve()),
            )
    positions = []
    bodies = []
    yaws = []
    for name in root_bodies:
        body = root.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"root body {name!r} missing")
        position = np.asarray(
            [float(item) for item in body.get("pos", "0 0 0").split()],
            dtype=float,
        )
        euler = np.asarray(
            [float(item) for item in body.get("euler", "0 0 0").split()],
            dtype=float,
        )
        positions.append(position)
        yaws.append(float(euler[2]))
        bodies.append(body)
    cosine, sine = np.cos(yaw_delta_rad), np.sin(yaw_delta_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    translation = np.asarray(translation_xy_m, dtype=float)
    for body, position, yaw in zip(bodies, positions, yaws):
        corrected = position.copy()
        corrected[:2] = rotation @ position[:2] + translation
        body.set("pos", " ".join(f"{value:.10f}" for value in corrected))
        body.set("euler", f"0 0 {yaw + yaw_delta_rad:.10f}")
    tree.write(Path(output), encoding="unicode")


def apply_independent_base_translations_to_mjcf(
    source: str | Path,
    output: str | Path,
    *,
    translations_xy_m: dict[str, list[float]],
) -> None:
    """Move independent arm roots in XY while retaining reviewed Z and yaw."""

    source = Path(source).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir"))
        if not meshdir.is_absolute():
            compiler.set(
                "meshdir",
                str((source.parent / meshdir).resolve()),
            )
    for name, translation in translations_xy_m.items():
        body = root.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"root body {name!r} missing")
        position = np.asarray(
            [float(item) for item in body.get("pos", "0 0 0").split()],
            dtype=float,
        )
        position[:2] += np.asarray(translation, dtype=float)
        body.set("pos", " ".join(f"{value:.10f}" for value in position))
    tree.write(Path(output), encoding="unicode")


def intersect_pixel_with_horizontal_plane(
    pixel_xy: tuple[float, float],
    camera_matrix: np.ndarray,
    scene_from_camera: np.ndarray,
    *,
    plane_z_m: float,
) -> np.ndarray:
    """Intersect an RGB camera ray with a Z-up scene plane."""

    x, y = pixel_xy
    inverse = np.linalg.inv(np.asarray(camera_matrix, dtype=float))
    ray_camera = inverse @ np.array([x, y, 1.0])
    transform = np.asarray(scene_from_camera, dtype=float)
    origin = transform[:3, 3]
    direction = transform[:3, :3] @ ray_camera
    if abs(direction[2]) < 1e-9:
        raise ValueError("camera ray is parallel to support plane")
    distance = (float(plane_z_m) - origin[2]) / direction[2]
    if distance <= 0:
        raise ValueError("support plane is behind the camera")
    return origin + distance * direction
