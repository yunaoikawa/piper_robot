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


def reject_base_candidates_inside_semantic_objects(
    candidate_centers_xyz: list[np.ndarray],
    semantic_objects: list[dict],
    *,
    margin_m: float = 0.02,
) -> tuple[list[np.ndarray], list[dict]]:
    """Reject stationary SAM robot candidates inside known non-robot objects.

    A fixed robot base and a static object are both persistent in stopped
    RGB-D views.  If SAM merges an arm with a microscope, arm occlusion can
    also make the microscope pass a temporal foreground test.  The semantic
    scene therefore supplies a second, independent exclusion test: a base
    candidate may not lie inside a completed non-robot object volume.
    """

    accepted = []
    records = []
    for index, value in enumerate(candidate_centers_xyz):
        point = np.asarray(value, dtype=float).reshape(3)
        overlaps = []
        for item in semantic_objects:
            if item.get("semantic_name") in {
                "robot",
                "measured_static_scene",
            }:
                continue
            if item.get("source") == "multiview_rgbd_background_faces":
                continue
            geometry = item.get("geometry", {})
            if geometry.get("kind") != "box":
                continue
            center = np.asarray(
                geometry.get("center_xyz_m", ()),
                dtype=float,
            )
            size = np.asarray(
                geometry.get("size_xyz_m", ()),
                dtype=float,
            )
            if center.shape != (3,) or size.shape != (3,):
                continue
            yaw = float(geometry.get("yaw_rad", 0.0))
            cosine, sine = np.cos(yaw), np.sin(yaw)
            relative = point - center
            local_xy = np.array(
                [
                    cosine * relative[0] + sine * relative[1],
                    -sine * relative[0] + cosine * relative[1],
                ]
            )
            local = np.r_[local_xy, relative[2]]
            if np.all(np.abs(local) <= size / 2.0 + margin_m):
                overlaps.append(
                    {
                        "instance_id": item.get("instance_id"),
                        "semantic_name": item.get("semantic_name"),
                    }
                )
        rejected = bool(overlaps)
        records.append(
            {
                "candidate_index": index,
                "center_xyz_m": point.tolist(),
                "accepted": not rejected,
                "overlapping_semantic_objects": overlaps,
            }
        )
        if not rejected:
            accepted.append(point)
    return accepted, records


def assign_visible_base_translations(
    initial_base_xyz: dict[str, np.ndarray],
    observed_base_xyz: list[np.ndarray],
    *,
    maximum_translation_m: float = 0.15,
    minimum_nearest_ratio: float = 2.0,
) -> dict:
    """Assign a partial set of visible bases and retain unobserved bases.

    This is intentionally conservative.  A candidate is accepted only when
    it is close to one reviewed base and distinctly farther from every other
    base.  Missing bases receive zero translation instead of being pulled
    toward a static object that leaked into a SAM robot mask.
    """

    names = list(initial_base_xyz)
    observed = [
        np.asarray(value, dtype=float).reshape(3)
        for value in observed_base_xyz
    ]
    if not names:
        raise ValueError("at least one initial base is required")
    if not observed:
        raise ValueError("at least one visible base candidate is required")
    if len(observed) > len(names):
        raise ValueError("more visible candidates than model bases")

    distances = np.asarray(
        [
            [
                np.linalg.norm(
                    point[:2]
                    - np.asarray(initial_base_xyz[name], dtype=float)[:2]
                )
                for name in names
            ]
            for point in observed
        ],
        dtype=float,
    )
    candidates = []
    for assigned_name_indices in permutations(range(len(names)), len(observed)):
        values = [
            distances[row, name_index]
            for row, name_index in enumerate(assigned_name_indices)
        ]
        candidates.append(
            (float(sum(values)), assigned_name_indices, values)
        )
    score, assigned_name_indices, assigned_distances = min(
        candidates,
        key=lambda item: item[0],
    )
    assignments = {}
    evidence = []
    for row, (name_index, distance) in enumerate(
        zip(assigned_name_indices, assigned_distances)
    ):
        alternatives = [
            distances[row, other]
            for other in range(len(names))
            if other != name_index
        ]
        nearest_ratio = (
            float(min(alternatives) / max(distance, 1e-9))
            if alternatives
            else float("inf")
        )
        if distance > maximum_translation_m:
            raise ValueError(
                f"visible base candidate requires {distance:.3f} m "
                f"translation, above {maximum_translation_m:.3f} m"
            )
        if nearest_ratio < minimum_nearest_ratio:
            raise ValueError(
                "visible base candidate is ambiguous between reviewed bases: "
                f"ratio={nearest_ratio:.3f}"
            )
        name = names[name_index]
        assignments[name] = observed[row]
        evidence.append(
            {
                "candidate_index": row,
                "base": name,
                "translation_m": float(distance),
                "nearest_alternative_ratio": nearest_ratio,
            }
        )

    translations = {}
    resulting_bases = {}
    for name in names:
        initial = np.asarray(initial_base_xyz[name], dtype=float)
        target = assignments.get(name)
        if target is None:
            translations[name] = np.zeros(2, dtype=float)
            resulting_bases[name] = initial.copy()
        else:
            translations[name] = target[:2] - initial[:2]
            resulting_bases[name] = np.r_[target[:2], initial[2]]
    retained = [name for name in names if name not in assignments]
    return {
        "method": "semantic_exclusion_then_partial_reviewed_base_assignment",
        "assignment": {
            name: (
                "observed_candidate"
                if name in assignments
                else "retained_reviewed_position"
            )
            for name in names
        },
        "score_m": score,
        "yaw_source": "reviewed_upright_model",
        "translations_xy_m": {
            name: value.tolist() for name, value in translations.items()
        },
        "base_xyz_level_m": {
            name: value.tolist() for name, value in resulting_bases.items()
        },
        "observed_bases": list(assignments),
        "retained_unobserved_bases": retained,
        "evidence": evidence,
        "all_bases_observed": not retained,
    }


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


def assign_named_base_translations(
    initial_base_xyz: dict[str, np.ndarray],
    observed_base_xyz: dict[str, np.ndarray],
) -> dict:
    """Move explicitly identified bases without reassigning them by proximity.

    ``observed_base_xyz`` must use the same semantic body names as the model.
    This is the safe path after physical arm identity has been established
    from synchronized joint excitation and image motion.
    """

    if set(initial_base_xyz) != set(observed_base_xyz):
        raise ValueError(
            "initial and observed named bases must contain identical names"
        )
    translations = {
        name: (
            np.asarray(observed_base_xyz[name], dtype=float)[:2]
            - np.asarray(initial_base_xyz[name], dtype=float)[:2]
        )
        for name in initial_base_xyz
    }
    return {
        "method": (
            "persistent_depth_named_by_synchronized_joint_excitation"
        ),
        "assignment": {
            name: name for name in initial_base_xyz
        },
        "score_m": float(
            sum(np.linalg.norm(value) for value in translations.values())
        ),
        "yaw_source": "reviewed_upright_model",
        "translations_xy_m": {
            name: value.tolist() for name, value in translations.items()
        },
        "base_xyz_level_m": {
            name: [
                float(observed_base_xyz[name][0]),
                float(observed_base_xyz[name][1]),
                float(initial_base_xyz[name][2]),
            ]
            for name in initial_base_xyz
        },
    }


def assign_components_by_joint_excitation(
    *,
    qpos_by_view: dict[str, dict[str, np.ndarray]],
    robot_masks_by_view: dict[str, np.ndarray],
    component_centers_px: list[np.ndarray],
    baseline_view: str,
    minimum_joint_excitation_rad: float = 0.1,
    minimum_joint_dominance_ratio: float = 1.5,
    motion_radius_fraction: float = 0.25,
    minimum_motion_density: float = 0.01,
    minimum_assignment_ratio: float = 1.25,
) -> dict:
    """Identify two physical arms from qpos excitation and fixed-camera SAM.

    View names and image-left/image-right are deliberately ignored. For each
    physical controller arm, the most arm-exclusive stopped view is selected
    from synchronized qpos. The SAM-mask change around each projected
    persistent base then assigns that arm to one component.
    """

    arms = ("left", "right")
    if baseline_view not in qpos_by_view:
        raise ValueError("baseline view is missing synchronized qpos")
    if baseline_view not in robot_masks_by_view:
        raise ValueError("baseline view is missing a robot mask")
    if len(component_centers_px) != 2:
        raise ValueError("exactly two projected base components are required")
    if set(qpos_by_view) != set(robot_masks_by_view):
        raise ValueError("qpos and robot masks must contain identical views")

    baseline_q = {
        arm: np.asarray(qpos_by_view[baseline_view][arm], dtype=float)
        for arm in arms
    }
    baseline_mask = np.asarray(
        robot_masks_by_view[baseline_view],
        dtype=bool,
    )
    if baseline_mask.ndim != 2:
        raise ValueError("robot masks must be two-dimensional")
    shape = baseline_mask.shape
    for name, mask in robot_masks_by_view.items():
        if np.asarray(mask).shape != shape:
            raise ValueError(f"robot mask shape changed in view {name}")

    joint_deltas: dict[str, dict[str, float]] = {}
    for view_name, qpos in qpos_by_view.items():
        joint_deltas[view_name] = {
            arm: float(
                np.linalg.norm(
                    np.asarray(qpos[arm], dtype=float) - baseline_q[arm]
                )
            )
            for arm in arms
        }

    selected_views = {}
    excitation_evidence = {}
    epsilon = 1e-9
    for arm in arms:
        other = "right" if arm == "left" else "left"
        candidates = []
        for view_name, deltas in joint_deltas.items():
            if view_name == baseline_view:
                continue
            dominance = deltas[arm] / max(deltas[other], epsilon)
            candidates.append(
                (dominance, deltas[arm], view_name, deltas[other])
            )
        if not candidates:
            raise ValueError(f"no excitation views available for {arm} arm")
        dominance, delta, view_name, other_delta = max(candidates)
        if delta < minimum_joint_excitation_rad:
            raise ValueError(
                f"{arm} arm excitation {delta:.4f} rad is below threshold"
            )
        if dominance < minimum_joint_dominance_ratio:
            raise ValueError(
                f"{arm} arm excitation is not exclusive enough: "
                f"ratio={dominance:.3f}"
            )
        selected_views[arm] = view_name
        excitation_evidence[arm] = {
            "view": view_name,
            "joint_delta_norm_rad": delta,
            "other_arm_joint_delta_norm_rad": other_delta,
            "dominance_ratio": dominance,
        }

    height, width = shape
    radius = float(motion_radius_fraction) * min(height, width)
    if not (0.0 < radius):
        raise ValueError("motion radius must be positive")
    yy, xx = np.ogrid[:height, :width]
    disks = []
    for center in component_centers_px:
        center = np.asarray(center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("projected component center is invalid")
        disks.append(
            (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
        )

    densities = {}
    for arm, view_name in selected_views.items():
        changed = np.logical_xor(
            baseline_mask,
            np.asarray(robot_masks_by_view[view_name], dtype=bool),
        )
        densities[arm] = [
            float(np.mean(changed[disk])) if np.any(disk) else 0.0
            for disk in disks
        ]

    # Only two bijections exist. A product strongly penalizes assigning an arm
    # to a component with no observed motion near its projected base.
    candidates = [
        (
            densities["left"][left_index]
            * densities["right"][right_index],
            {"left": left_index, "right": right_index},
        )
        for left_index, right_index in ((0, 1), (1, 0))
    ]
    product, assignment = max(candidates, key=lambda item: item[0])
    for arm in arms:
        selected = densities[arm][assignment[arm]]
        alternative = densities[arm][1 - assignment[arm]]
        ratio = selected / max(alternative, epsilon)
        if selected < minimum_motion_density:
            raise ValueError(
                f"{arm} arm SAM motion density {selected:.4f} is too low"
            )
        if ratio < minimum_assignment_ratio:
            raise ValueError(
                f"{arm} arm component assignment is ambiguous: "
                f"ratio={ratio:.3f}"
            )
        excitation_evidence[arm].update(
            {
                "component_index": assignment[arm],
                "selected_motion_density": selected,
                "alternative_motion_density": alternative,
                "assignment_ratio": ratio,
            }
        )

    return {
        "accepted": True,
        "baseline_view": baseline_view,
        "physical_arm_to_component_index": assignment,
        "component_centers_px": [
            np.asarray(value, dtype=float).tolist()
            for value in component_centers_px
        ],
        "joint_delta_norm_rad_by_view": joint_deltas,
        "evidence": excitation_evidence,
        "assignment_product": product,
        "thresholds": {
            "minimum_joint_excitation_rad": minimum_joint_excitation_rad,
            "minimum_joint_dominance_ratio": (
                minimum_joint_dominance_ratio
            ),
            "motion_radius_fraction": motion_radius_fraction,
            "minimum_motion_density": minimum_motion_density,
            "minimum_assignment_ratio": minimum_assignment_ratio,
        },
        "policy": (
            "physical controller qpos excitation plus fixed-camera SAM motion; "
            "view names and image-side heuristics are ignored"
        ),
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
