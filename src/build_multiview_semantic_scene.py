#!/usr/bin/env python3
"""Complete a SAM-labelled multiview mesh into a reviewable MuJoCo scene.

Measured surfaces and inferred completion are intentionally separate.  A
scene can be useful for visual review before camera-to-robot calibration, but
collision and motion readiness remain false until an accepted calibration and
synchronized articulated state are both present.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import differential_evolution

from rollout.daily_scene import DailySceneStore, SceneObject
from rollout.semantic_scene_pipeline import (
    ObjectGeometry,
    load_profile,
    robust_oriented_geometry,
    scene_json_ready,
    sha256_file,
)


SCHEMA = "piper_robot.multiview_completed_scene/v1"
CALIBRATION_SCHEMA = "piper_robot.camera_robot_calibration/v1"


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def voxel_components(
    points: np.ndarray,
    *,
    voxel_size_m: float = 0.05,
    minimum_points: int = 3,
) -> list[np.ndarray]:
    """Return 26-connected point components without image-position priors."""

    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return []
    voxel = float(voxel_size_m)
    keys = np.floor(points / voxel).astype(np.int64)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(keys):
        cells.setdefault(tuple(int(item) for item in key), []).append(index)
    unseen = set(cells)
    result = []
    offsets = [
        (x, y, z)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)
    ]
    while unseen:
        seed = unseen.pop()
        queue = [seed]
        indices = list(cells[seed])
        while queue:
            current = queue.pop()
            for offset in offsets:
                neighbor = tuple(
                    current[axis] + offset[axis] for axis in range(3)
                )
                if neighbor not in unseen:
                    continue
                unseen.remove(neighbor)
                queue.append(neighbor)
                indices.extend(cells[neighbor])
        if len(indices) >= minimum_points:
            result.append(points[np.asarray(indices, dtype=np.int64)])
    result.sort(key=len, reverse=True)
    return result


def _triangle_areas_and_normals(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = lengths > 1e-10
    normals[valid] = cross[valid] / lengths[valid, None]
    return lengths / 2, normals


def _grid_components(indices: np.ndarray) -> list[np.ndarray]:
    cells = {tuple(int(item) for item in index) for index in indices}
    result = []
    while cells:
        seed = cells.pop()
        queue = [seed]
        component = [seed]
        while queue:
            x, y = queue.pop()
            for neighbor in (
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
                (x - 1, y - 1),
                (x - 1, y + 1),
                (x + 1, y - 1),
                (x + 1, y + 1),
            ):
                if neighbor in cells:
                    cells.remove(neighbor)
                    queue.append(neighbor)
                    component.append(neighbor)
        result.append(np.asarray(component, dtype=np.int64))
    result.sort(key=len, reverse=True)
    return result


def discover_multilevel_supports(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    *,
    background_label: int = 2,
    height_bin_m: float = 0.005,
    cell_size_m: float = 0.04,
    minimum_area_m2: float = 0.025,
) -> list[dict]:
    """Find level supports and preserve cut-outs as occupied XY cells."""

    areas, normals = _triangle_areas_and_normals(vertices, faces)
    face_background = np.all(labels[faces] == int(background_label), axis=1)
    horizontal = (
        face_background
        & (areas > 1e-8)
        & (np.abs(normals[:, 2]) >= np.cos(np.deg2rad(15.0)))
    )
    if not np.any(horizontal):
        return []
    selected_faces = faces[horizontal]
    selected_areas = areas[horizontal]
    z = np.mean(vertices[selected_faces, 2], axis=1)
    bins = np.rint(z / height_bin_m).astype(np.int64)
    area_by_bin: dict[int, float] = {}
    for key, area in zip(bins, selected_areas):
        area_by_bin[int(key)] = area_by_bin.get(int(key), 0.0) + float(area)
    peaks = []
    for key, area in sorted(
        area_by_bin.items(), key=lambda item: item[1], reverse=True
    ):
        if area < minimum_area_m2:
            continue
        if any(abs(key - old) <= 2 for old in peaks):
            continue
        peaks.append(key)
        if len(peaks) >= 4:
            break

    supports = []
    centers = np.mean(vertices[selected_faces], axis=1)
    for peak in peaks:
        level = peak * height_bin_m
        band = np.abs(centers[:, 2] - level) <= 1.5 * height_bin_m
        if not np.any(band):
            continue
        xy = centers[band, :2]
        cells = np.unique(np.floor(xy / cell_size_m).astype(np.int64), axis=0)
        for component in _grid_components(cells):
            area = len(component) * cell_size_m**2
            if area < minimum_area_m2:
                continue
            collision_boxes = [
                {
                    "center_xy_m": (
                        (cell.astype(float) + 0.5) * cell_size_m
                    ).tolist(),
                    "size_xy_m": [cell_size_m, cell_size_m],
                }
                for cell in component
            ]
            lower = np.min(component, axis=0) * cell_size_m
            upper = (np.max(component, axis=0) + 1) * cell_size_m
            supports.append(
                {
                    "support_id": f"support-{len(supports) + 1}",
                    "height_m": float(np.median(centers[band, 2])),
                    "bounds_xy_m": [lower.tolist(), upper.tolist()],
                    "area_m2": float(area),
                    "collision_boxes": collision_boxes,
                    "source": "measured_horizontal_background_triangles",
                    "holes_preserved": True,
                }
            )
    supports.sort(key=lambda item: (item["height_m"], -item["area_m2"]))
    if not supports:
        return []
    # The lowest bench can be split by robot occlusion.  Merge its disjoint
    # components semantically while retaining every occupied cell, so holes
    # remain holes.  Raised components stay separate as the two work platforms.
    lowest = min(item["height_m"] for item in supports)
    bench_parts = [
        item for item in supports if abs(item["height_m"] - lowest) <= 0.015
    ]
    raised = [
        item for item in supports if abs(item["height_m"] - lowest) > 0.015
    ]
    if len(bench_parts) > 1:
        bounds = np.asarray(
            [bound for item in bench_parts for bound in item["bounds_xy_m"]],
            dtype=float,
        )
        merged = {
            "support_id": "support-bench",
            "height_m": float(
                np.average(
                    [item["height_m"] for item in bench_parts],
                    weights=[item["area_m2"] for item in bench_parts],
                )
            ),
            "bounds_xy_m": [
                np.min(bounds, axis=0).tolist(),
                np.max(bounds, axis=0).tolist(),
            ],
            "area_m2": float(sum(item["area_m2"] for item in bench_parts)),
            "collision_boxes": [
                box for item in bench_parts for box in item["collision_boxes"]
            ],
            "source": "merged_measured_bench_components",
            "holes_preserved": True,
        }
        supports = [merged, *raised]
    for index, support in enumerate(supports):
        if support["support_id"].startswith("support-"):
            support["support_id"] = (
                "support-bench"
                if index == 0
                else f"support-platform-{index}"
            )
    return supports


def _support_for(
    points: np.ndarray,
    supports: list[dict],
    *,
    semantic_name: str,
    profile: dict,
) -> dict | None:
    assignment = profile.get("support_assignment", {})
    semantic_role = assignment.get("semantic_roles", {}).get(semantic_name)
    if semantic_role in {"front_elevated", "rear_elevated"}:
        height_tolerance = float(
            assignment.get("elevated_height_tolerance_m", 0.020)
        )
        highest = max(item["height_m"] for item in supports)
        elevated = [
            item
            for item in supports
            if abs(item["height_m"] - highest) <= height_tolerance
        ]
        if len(elevated) >= 2:
            axis = int(assignment.get("depth_axis", 1))
            if axis not in (0, 1):
                raise ValueError("support_assignment.depth_axis must be 0 or 1")
            sign = float(assignment.get("depth_sign", 1.0))
            if sign not in (-1.0, 1.0):
                raise ValueError("support_assignment.depth_sign must be -1 or 1")
            ordered = sorted(
                elevated,
                key=lambda item: sign
                * float(
                    np.mean(
                        np.asarray(item["bounds_xy_m"], dtype=float)[:, axis]
                    )
                ),
            )
            return (
                ordered[0]
                if semantic_role == "front_elevated"
                else ordered[-1]
            )

    center = np.median(points[:, :2], axis=0)
    below = []
    for support in supports:
        lower, upper = np.asarray(support["bounds_xy_m"], dtype=float)
        margin = 0.06
        if np.all(center >= lower - margin) and np.all(center <= upper + margin):
            if support["height_m"] <= np.quantile(points[:, 2], 0.85) + 0.04:
                below.append(support)
    return max(below, key=lambda item: item["height_m"]) if below else None


def _signed_box_distance(
    points: np.ndarray,
    *,
    center_xy: np.ndarray,
    yaw_rad: float,
    size_xyz: np.ndarray,
    support_height_m: float,
) -> np.ndarray:
    cosine, sine = np.cos(yaw_rad), np.sin(yaw_rad)
    delta = np.asarray(points, dtype=float).copy()
    delta[:, :2] -= np.asarray(center_xy, dtype=float)
    local = np.column_stack(
        (
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
            delta[:, 2] - (float(support_height_m) + size_xyz[2] / 2),
        )
    )
    q = np.abs(local) - size_xyz / 2
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def fit_box_to_semantic_volume(
    observed_surface_points: np.ndarray,
    semantic_voxel_points: np.ndarray,
    known_free_voxel_points: np.ndarray,
    initial: ObjectGeometry,
    *,
    support_height_m: float,
    voxel_size_m: float,
    configuration: dict,
) -> tuple[ObjectGeometry, dict]:
    """Fit a gravity-aligned box using semantic occupancy and known free space.

    The observed mesh is only a partial surface, so a symmetric Chamfer score
    alone can place the hidden half of a box in free space.  The objective
    combines surface distance, missed semantic volume, and candidate volume
    intruding into explicitly observed free voxels.  Unknown space is not
    treated as free.
    """

    observed = np.asarray(observed_surface_points, dtype=float)
    semantic = np.asarray(semantic_voxel_points, dtype=float)
    known_free = np.asarray(known_free_voxel_points, dtype=float)
    minimum = int(configuration.get("minimum_observed_points", 200))
    if len(observed) < minimum or len(semantic) < minimum:
        return initial, {
            "attempted": False,
            "accepted": False,
            "reason": "semantic_volume_points_insufficient",
            "observed_points": int(len(observed)),
            "semantic_voxels": int(len(semantic)),
        }
    maximum_surface = int(configuration.get("maximum_surface_points", 2200))
    maximum_semantic = int(configuration.get("maximum_semantic_voxels", 3000))
    maximum_free = int(configuration.get("maximum_free_voxels", 9000))

    def evenly_sample(values: np.ndarray, maximum: int) -> tuple[np.ndarray, float]:
        if len(values) <= maximum:
            return values, 1.0
        indices = np.linspace(0, len(values) - 1, maximum).astype(np.int64)
        return values[indices], float(len(values)) / float(maximum)

    observed, _ = evenly_sample(observed, maximum_surface)
    semantic, _ = evenly_sample(semantic, maximum_semantic)
    median = np.median(observed, axis=0)
    radius = float(configuration.get("free_space_roi_radius_m", 0.70))
    free_roi = known_free[
        np.all(np.abs(known_free - median) <= radius, axis=1)
    ]
    free_roi, free_weight = evenly_sample(free_roi, maximum_free)
    size = np.asarray(initial.size_xyz_m, dtype=float)
    search = float(configuration.get("center_search_radius_m", 0.20))
    bounds = [
        (float(median[0] - search), float(median[0] + search)),
        (float(median[1] - search), float(median[1] + search)),
        (-np.pi / 2, np.pi / 2),
    ]
    surface_weight = float(configuration.get("surface_weight", 1.0))
    semantic_weight = float(configuration.get("semantic_miss_weight_m", 0.08))
    free_weight_scale = float(
        configuration.get("known_free_intrusion_weight_m", 0.40)
    )
    shell = float(configuration.get("semantic_shell_tolerance_m", 0.015))
    clearance = float(configuration.get("known_free_clearance_m", 0.005))
    candidate_volume = float(np.prod(size))

    def terms(parameters: np.ndarray) -> dict:
        center_xy = parameters[:2]
        yaw = float(parameters[2])
        surface_distance = np.abs(
            _signed_box_distance(
                observed,
                center_xy=center_xy,
                yaw_rad=yaw,
                size_xyz=size,
                support_height_m=support_height_m,
            )
        )
        semantic_distance = _signed_box_distance(
            semantic,
            center_xy=center_xy,
            yaw_rad=yaw,
            size_xyz=size,
            support_height_m=support_height_m,
        )
        if len(free_roi):
            free_distance = _signed_box_distance(
                free_roi,
                center_xy=center_xy,
                yaw_rad=yaw,
                size_xyz=size,
                support_height_m=support_height_m,
            )
            free_volume = (
                np.count_nonzero(free_distance < -clearance)
                * free_weight
                * float(voxel_size_m) ** 3
            )
        else:
            free_volume = 0.0
        return {
            "surface_p65_m": float(np.quantile(surface_distance, 0.65)),
            "surface_p90_m": float(np.quantile(surface_distance, 0.90)),
            "semantic_miss_fraction": float(
                np.mean(semantic_distance > shell)
            ),
            "known_free_intrusion_fraction": float(
                free_volume / max(candidate_volume, 1e-9)
            ),
        }

    def objective(parameters: np.ndarray) -> float:
        value = terms(parameters)
        return float(
            surface_weight
            * (value["surface_p65_m"] + 0.4 * value["surface_p90_m"])
            + semantic_weight * value["semantic_miss_fraction"]
            + free_weight_scale * value["known_free_intrusion_fraction"]
        )

    initial_yaw = float(
        np.arctan2(np.sin(initial.yaw_rad), np.cos(initial.yaw_rad))
    )
    if initial_yaw > np.pi / 2:
        initial_yaw -= np.pi
    elif initial_yaw < -np.pi / 2:
        initial_yaw += np.pi
    initial_parameters = np.asarray(
        [initial.center_xyz_m[0], initial.center_xyz_m[1], initial_yaw],
        dtype=float,
    )
    initial_cost = objective(initial_parameters)
    result = differential_evolution(
        objective,
        bounds,
        seed=int(configuration.get("seed", 17)),
        popsize=int(configuration.get("population_size", 9)),
        maxiter=int(configuration.get("maximum_iterations", 32)),
        polish=True,
        workers=1,
        updating="immediate",
    )
    optimized = np.asarray(result.x, dtype=float)
    optimized_cost = objective(optimized)
    improvement = (
        (initial_cost - optimized_cost) / max(initial_cost, 1e-9)
    )
    accepted = bool(
        np.isfinite(optimized_cost)
        and improvement
        >= float(configuration.get("minimum_improvement_fraction", 0.15))
    )
    geometry = (
        ObjectGeometry(
            kind=initial.kind,
            center_xyz_m=(
                float(optimized[0]),
                float(optimized[1]),
                float(support_height_m + size[2] / 2),
            ),
            size_xyz_m=tuple(float(item) for item in size),
            yaw_rad=float(optimized[2]),
        )
        if accepted
        else initial
    )
    return geometry, {
        "attempted": True,
        "accepted": accepted,
        "method": (
            "gravity_aligned_box_semantic_voxel_symmetric_difference_"
            "plus_known_free_space"
        ),
        "initial": {
            "center_xy_m": initial_parameters[:2].tolist(),
            "yaw_rad": float(initial_parameters[2]),
            "objective": float(initial_cost),
            **terms(initial_parameters),
        },
        "optimized": {
            "center_xy_m": optimized[:2].tolist(),
            "yaw_rad": float(optimized[2]),
            "objective": float(optimized_cost),
            **terms(optimized),
        },
        "improvement_fraction": float(improvement),
        "candidate_volume_m3": candidate_volume,
        "known_free_space_is_authoritative": True,
        "unknown_space_penalized": False,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def _volume_points(
    archive,
    mask: np.ndarray,
    transform: np.ndarray | None,
) -> np.ndarray:
    indices_zyx = np.argwhere(mask)
    origin = np.asarray(archive["origin_xyz_m"], dtype=float)
    voxel = float(archive["voxel_size_m"])
    points = origin + (indices_zyx[:, [2, 1, 0]] + 0.5) * voxel
    return points if transform is None else transform_points(points, transform)


def _write_semantic_mesh(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    label_id: int,
) -> str | None:
    from src.build_semantic_scene import _compact_mesh, _write_obj

    selected = faces[np.all(labels[faces] == label_id, axis=1)]
    if not len(selected):
        return None
    compact_vertices, compact_faces = _compact_mesh(vertices, selected)
    _write_obj(path, compact_vertices, compact_faces)
    return str(path.resolve())


def _pin_nyu_grippers(
    source: Path,
    output: Path,
    profile: dict,
    base_positions: dict[str, np.ndarray] | None,
    yaw: float,
) -> dict:
    from src.build_semantic_scene import (
        _install_configured_end_effectors,
        _install_configured_initial_pose,
    )

    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    meshdir = Path(compiler.get("meshdir", "assets"))
    if not meshdir.is_absolute():
        meshdir = (source.parent / meshdir).resolve()
    compiler.set("meshdir", str(meshdir))
    end_effector = _install_configured_end_effectors(
        root, profile.get("robot_end_effector")
    )
    initial_pose = _install_configured_initial_pose(
        root, profile.get("robot_initial_pose")
    )
    if base_positions:
        for name, position in base_positions.items():
            body = root.find(f".//body[@name='{name}']")
            if body is None:
                raise ValueError(f"robot root body {name!r} is absent")
            body.set("pos", " ".join(f"{item:.10f}" for item in position))
            body.set("euler", f"0 0 {yaw:.10f}")
    tree.write(output, encoding="unicode")
    return {
        "source_model": str(source.resolve()),
        "positioned_model": str(output.resolve()),
        "end_effector": end_effector,
        "initial_pose": initial_pose,
    }


def _display_robot_placement(
    robot_points: np.ndarray,
    supports: list[dict],
    profile: dict,
) -> tuple[dict[str, np.ndarray], float, dict]:
    components = voxel_components(robot_points, voxel_size_m=0.05, minimum_points=40)
    chosen = []
    for component in components:
        center = np.median(component[:, :2], axis=0)
        if any(
            np.linalg.norm(center - np.median(old[:, :2], axis=0)) < 0.18
            for old in chosen
        ):
            continue
        chosen.append(component)
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise RuntimeError("two spatially separate Piper observations were not found")
    bases = []
    for component in chosen:
        threshold = np.quantile(component[:, 2], 0.18)
        base_xy = np.median(component[component[:, 2] <= threshold, :2], axis=0)
        bases.append(base_xy)
    bases.sort(key=lambda value: float(value[0]))
    left_xy, right_xy = bases
    raised = max(supports, key=lambda item: item["height_m"])
    mount_below = float(
        profile.get("robot_placement", {})
        .get("shared_base_height", {})
        .get("mount_below_support_m", 0.197)
    )
    base_z = float(raised["height_m"] - mount_below)
    observed_vector = left_xy - right_xy
    yaw = float(np.arctan2(observed_vector[1], observed_vector[0]) - np.pi / 2)
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
    positions = {
        "left/base_link": np.r_[left_xy, base_z],
        "right/base_link": np.r_[right_xy, base_z],
    }
    return positions, yaw, {
        "required": True,
        "accepted": False,
        "method": "two_largest_sam_robot_3d_components_display_only",
        "reason": "camera_to_robot_extrinsic_not_accepted",
        "base_xyz_level_m": {
            name: value.tolist() for name, value in positions.items()
        },
        "shared_upright_yaw_rad": yaw,
        "support_id": raised["support_id"],
    }


def _validated_calibration(path: str | None) -> dict | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("unsupported camera-to-robot calibration schema")
    if not payload.get("accepted", False):
        return None
    transform = np.asarray(payload.get("T_robot_camera"), dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("accepted calibration lacks T_robot_camera")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("accepted calibration rotation is not orthonormal")
    return payload


def _qpos_from_report(report: dict) -> tuple[list[float] | None, str | None]:
    per_view = report.get("robot_state", {}).get("per_view", {})
    qposes = []
    for name, value in per_view.items():
        stability = (value or {}).get("stability", {})
        qpos = stability.get("representative_qpos_rad")
        if stability.get("accepted") and qpos is not None and len(qpos) == 12:
            qposes.append((name, [float(item) for item in qpos]))
    if not qposes:
        return None, None
    return qposes[0][1], qposes[0][0]


def _write_index(path: Path, scene: dict) -> None:
    readiness = scene["readiness"]
    reasons = "".join(f"<li>{item}</li>" for item in readiness["reasons"])
    optimization_rows = []
    for item in scene.get("objects", ()):
        report = item.get("semantic_volume_fit", {})
        if not report.get("attempted"):
            continue
        before = report.get("initial", {})
        after = report.get("optimized", {})
        optimization_rows.append(
            "<tr>"
            f"<td>{item['semantic_name']}</td>"
            f"<td>{report.get('accepted')}</td>"
            f"<td>{before.get('objective', float('nan')):.4f}</td>"
            f"<td>{after.get('objective', float('nan')):.4f}</td>"
            f"<td>{100 * report.get('improvement_fraction', 0.0):.1f}%</td>"
            f"<td>{100 * before.get('known_free_intrusion_fraction', 0.0):.2f}%</td>"
            f"<td>{100 * after.get('known_free_intrusion_fraction', 0.0):.2f}%</td>"
            f"<td>{np.degrees(before.get('yaw_rad', 0.0)):.1f}°</td>"
            f"<td>{np.degrees(after.get('yaw_rad', 0.0)):.1f}°</td>"
            "</tr>"
        )
    optimization = (
        "<h2>体積最適化</h2><div class=\"scroll\"><table><thead><tr>"
        "<th>物体</th><th>採用</th><th>前loss</th><th>後loss</th>"
        "<th>改善</th><th>free侵入 前</th><th>free侵入 後</th>"
        "<th>yaw前</th><th>yaw後</th></tr></thead><tbody>"
        + "".join(optimization_rows)
        + "</tbody></table></div>"
        if optimization_rows
        else ""
    )
    path.write_text(
        """<!doctype html><meta name="viewport" content="width=device-width">
<meta charset="utf-8"><title>Pasteur semantic scene</title>
<style>body{font:16px system-ui;margin:22px;max-width:760px;background:#111827;
color:#f9fafb}a{display:block;margin:12px 0;padding:14px;border-radius:12px;
background:#1f2937;color:#7dd3fc;text-decoration:none}.ok{color:#86efac}
.no{color:#fca5a5}code{overflow-wrap:anywhere}.scroll{overflow-x:auto}
table{border-collapse:collapse;min-width:720px}th,td{padding:8px;border:1px solid #475569;
text-align:right}th:first-child,td:first-child{text-align:left}</style>
<h1>SAM-first 3D / MuJoCo</h1>
<p>表示: <b class="ok">%s</b>　衝突: <b class="%s">%s</b>　動作: <b class="%s">%s</b></p>
<a href="semantic_3d.html">意味付き3D（観測面／推論形状）</a>
<a href="mujoco_home.html">MuJoCo（NYU gripper・home）</a>
<a href="source_esdf_scene.html">元の保守的ESDF（インタラクティブ）</a>
<a href="scene.json">scene.json</a>
%s
<h2>未承認理由</h2><ul>%s</ul>
<p>このページの表示可否と実機動作許可は別です。現在のモデルは実機へ命令しません。</p>
"""
        % (
            readiness["display_ready"],
            "ok" if readiness["collision_ready"] else "no",
            readiness["collision_ready"],
            "ok" if readiness["motion_ready"] else "no",
            readiness["motion_ready"],
            optimization,
            reasons,
        ),
        encoding="utf-8",
    )


def build(args) -> dict:
    from src import build_semantic_scene as single
    from src.render_mujoco_mobile import render

    report_path = Path(args.multiview_report).resolve()
    report = json.loads(report_path.read_text())
    if report.get("schema") != "piper_robot.multiview_semantic_scene/v1":
        raise ValueError("unsupported multiview report")
    if not report.get("readiness", {}).get("display_ready", False):
        raise ValueError("multiview source is not display-ready")
    source_dir = report_path.parent
    mesh_path = source_dir / "scene_mesh_multiview.npz"
    archive = np.load(mesh_path)
    vertices = np.asarray(archive["vertices_xyz_m"], dtype=float)
    faces = np.asarray(archive["faces"], dtype=np.int32)
    colors = np.asarray(archive["colors_rgb"], dtype=np.uint8)
    labels = np.asarray(archive["semantic_labels"], dtype=np.int32)
    source_esdf = Path(report["artifacts"]["esdf"]).resolve()
    volume_archive = np.load(source_esdf)
    profile, catalog = load_profile(args.profile)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    calibration = _validated_calibration(
        getattr(args, "calibration_report", None)
    )
    frame = "gravity_levelled_first_record3d_camera"
    transform_report = None
    robot_from_level = None
    if calibration is not None:
        level_from_camera = np.asarray(
            report.get("coordinate_frame", {}).get("T_level_first_camera"),
            dtype=float,
        )
        if level_from_camera.shape != (4, 4):
            raise ValueError(
                "multiview report predates T_level_first_camera; reconstruct it"
            )
        robot_from_camera = np.asarray(
            calibration["T_robot_camera"], dtype=float
        )
        robot_from_level = robot_from_camera @ np.linalg.inv(level_from_camera)
        vertices = transform_points(vertices, robot_from_level)
        frame = "bimanual_piper_robot_base"
        transform_report = {
            "T_robot_level": robot_from_level.tolist(),
            "calibration": str(Path(args.calibration_report).resolve()),
        }

    label_ids = {
        name: int(value)
        for name, value in report["semantics"]["label_ids"].items()
    }
    volume_labels = np.asarray(volume_archive["semantic_labels"], dtype=np.uint8)
    volume_observed = np.asarray(volume_archive["observed"], dtype=bool)
    known_free_points = _volume_points(
        volume_archive,
        volume_observed & (volume_labels == 1),
        robot_from_level,
    )
    supports = discover_multilevel_supports(vertices, faces, labels)
    if len(supports) < 3:
        raise RuntimeError(
            f"expected bench plus two platforms, found {len(supports)} supports"
        )
    for support in supports:
        face_centers = np.mean(vertices[faces], axis=1)
        lower_xy, upper_xy = np.asarray(support["bounds_xy_m"], dtype=float)
        support_faces = faces[
            np.all(labels[faces] == 2, axis=1)
            & (
                np.abs(face_centers[:, 2] - support["height_m"])
                <= 0.010
            )
            & np.all(face_centers[:, :2] >= lower_xy, axis=1)
            & np.all(face_centers[:, :2] <= upper_xy, axis=1)
        ]
        if len(support_faces):
            compact_vertices, compact_faces = single._compact_mesh(
                vertices, support_faces
            )
            support_mesh = output / f"{support['support_id']}.obj"
            single._write_obj(support_mesh, compact_vertices, compact_faces)
            support["observed_mesh"] = str(support_mesh.resolve())

    observed = []
    records = []
    selected_points: dict[str, np.ndarray] = {}
    for name in profile.get("objects", ()):
        label_id = label_ids.get(name)
        if label_id is None:
            continue
        points = vertices[labels == label_id]
        components = voxel_components(points, voxel_size_m=0.05)
        if not components and len(points) >= 3:
            components = [points]
        if not components:
            continue
        selected = components[0]
        if name == "robot":
            selected = points
        selected_points[name] = selected
        mesh_file = _write_semantic_mesh(
            output / f"observed_{name}.obj",
            vertices,
            faces,
            labels,
            label_id,
        )
        if mesh_file:
            semantic_faces = faces[np.all(labels[faces] == label_id, axis=1)]
            compact_vertices, compact_faces = single._compact_mesh(
                vertices, semantic_faces
            )
            observed.append(
                {
                    "semantic_name": name,
                    "vertices": compact_vertices,
                    "faces": compact_faces,
                }
            )
        if name == "robot":
            continue
        definition = catalog[name]
        support = _support_for(
            selected,
            supports,
            semantic_name=name,
            profile=profile,
        )
        geometry = robust_oriented_geometry(
            selected,
            catalog=definition,
            support_height_m=None if support is None else support["height_m"],
        )
        volume_fit = {
            "attempted": False,
            "accepted": False,
            "reason": "not_an_eligible_box_template",
        }
        fit_configuration = profile.get("semantic_volume_fit", {})
        if (
            fit_configuration.get("enabled", False)
            and definition.completion == "template"
            and definition.primitive == "box"
            and support is not None
        ):
            semantic_voxels = _volume_points(
                volume_archive,
                volume_observed & (volume_labels == label_id),
                robot_from_level,
            )
            geometry, volume_fit = fit_box_to_semantic_volume(
                selected,
                semantic_voxels,
                known_free_points,
                geometry,
                support_height_m=float(support["height_m"]),
                voxel_size_m=float(volume_archive["voxel_size_m"]),
                configuration=fit_configuration,
            )
        view_observations = [
            item
            for values in report["semantics"]["views"].values()
            for item in values
            if item["semantic_name"] == name
        ]
        scores = [float(item["sam_score"]) for item in view_observations]
        confidence = float(np.median(scores)) if scores else 0.0
        status = (
            "uncertain"
            if definition.transparent
            or len(selected) < 50
            or confidence < definition.minimum_confidence
            else "auto_observed"
        )
        record = {
            "instance_id": f"{name}-1",
            "semantic_name": name,
            "geometry": asdict(geometry),
            "completion": definition.completion,
            "color": definition.color,
            "confidence": confidence,
            "status": status,
            "source": "sam_multiview_rgbd_plus_catalog_completion",
            "transparent": definition.transparent,
            "support_id": None if support is None else support["support_id"],
            "observed_points": int(len(selected)),
            "observed_views": len(
                {
                    view_name
                    for view_name, values in report["semantics"]["views"].items()
                    if any(item["semantic_name"] == name for item in values)
                }
            ),
            "measured_and_inferred_separate": True,
            "observed_mesh": mesh_file,
            "semantic_volume_fit": volume_fit,
        }
        if volume_fit.get("accepted"):
            record["source"] += "+semantic_volume_fit"
        if name == "microscope" and mesh_file:
            record["completion"] = "observed_mesh"
            record["collision_boxes"] = single._observed_voxel_boxes(
                selected, voxel_size_m=0.03
            )
        records.append(record)

    robot_points = selected_points.get("robot")
    if robot_points is None:
        raise RuntimeError("SAM-labelled robot points are missing")
    runtime_profile = dict(profile)
    source_robot = Path(profile["robot_model"]).resolve()
    positioned_robot = output / "positioned_robot.xml"
    if calibration is None:
        base_positions, yaw, robot_placement = _display_robot_placement(
            robot_points, supports, profile
        )
    else:
        base_positions, yaw = None, 0.0
        robot_placement = {
            "required": True,
            "accepted": True,
            "method": "accepted_camera_to_robot_extrinsic",
            "calibration": str(Path(args.calibration_report).resolve()),
        }
    pinned = _pin_nyu_grippers(
        source_robot,
        positioned_robot,
        profile,
        base_positions,
        yaw,
    )
    robot_placement.update(pinned)
    runtime_profile["robot_model"] = str(positioned_robot.resolve())
    records.insert(
        0,
        {
            "instance_id": "bimanual-piper",
            "semantic_name": "robot",
            "geometry": {
                "kind": "box",
                "center_xyz_m": np.median(robot_points, axis=0).tolist(),
                "size_xyz_m": np.ptp(robot_points, axis=0).tolist(),
                "yaw_rad": yaw,
            },
            "completion": "exact_cad",
            "color": catalog["robot"].color,
            "confidence": 1.0,
            "status": "confirmed" if calibration else "display_only",
            "source": "exact_piper_cad_plus_sam_alignment",
            "transparent": False,
            "observed_points": int(len(robot_points)),
            "observed_mesh": _write_semantic_mesh(
                output / "observed_robot.obj",
                vertices,
                faces,
                labels,
                label_ids["robot"],
            ),
        },
    )

    background_faces = faces[np.all(labels[faces] == 2, axis=1)]
    static_mesh_path = output / "observed_static_scene.obj"
    if len(background_faces):
        compact_vertices, compact_faces = single._compact_mesh(
            vertices, background_faces
        )
        single._write_obj(static_mesh_path, compact_vertices, compact_faces)
        observed.append(
            {
                "semantic_name": "measured_static_scene",
                "vertices": compact_vertices,
                "faces": compact_faces,
            }
        )
        records.append(
            {
                "instance_id": "measured-static-scene",
                "semantic_name": "measured_static_scene",
                "geometry": {
                    "kind": "box",
                    "center_xyz_m": np.median(compact_vertices, axis=0).tolist(),
                    "size_xyz_m": np.maximum(
                        np.ptp(compact_vertices, axis=0), 0.002
                    ).tolist(),
                    "yaw_rad": 0.0,
                },
                "completion": "observed_mesh",
                "color": "#1f2937",
                "confidence": 1.0,
                "status": "measured",
                "source": "multiview_rgbd_background_faces",
                "transparent": False,
                "observed_mesh": str(static_mesh_path.resolve()),
                "collision_boxes": [],
            }
        )
    observed_surface_objects = tuple(
        {
            *profile.get("observed_surface_objects", ()),
            "measured_static_scene",
            "robot",
        }
    )
    runtime_profile["observed_surface_objects"] = list(observed_surface_objects)

    qpos, qpos_view = _qpos_from_report(report)
    expected_calibration_id = str(
        calibration.get("calibration_id")
        if calibration
        else profile.get("calibration_id", "display-only")
    )
    consumed_daily = None
    if getattr(args, "daily_scene", None):
        store = DailySceneStore(args.daily_scene)
        existing = store.load()
        if existing is not None and existing.status == "confirmed":
            if existing.calibration_id != expected_calibration_id:
                raise ValueError(
                    "confirmed daily scene belongs to a different calibration"
                )
            decisions = {
                item.instance_id: item for item in existing.objects
            }
            expected_ids = {
                item["instance_id"]
                for item in records
                if item["semantic_name"] != "measured_static_scene"
            }
            if set(decisions) != expected_ids:
                raise ValueError(
                    "confirmed daily scene object set differs from reconstruction"
                )
            reviewed = []
            for item in records:
                decision = decisions.get(item["instance_id"])
                if decision is None:
                    reviewed.append(item)
                    continue
                if decision.status == "absent":
                    continue
                updated = dict(item)
                updated["geometry"] = dict(decision.geometry)
                updated["status"] = "confirmed"
                updated["confidence"] = max(
                    float(updated["confidence"]), float(decision.confidence)
                )
                updated["source"] = (
                    f"{updated['source']}+operator_confirmed_daily_scene"
                )
                reviewed.append(updated)
            records = reviewed
            consumed_daily = existing
        elif getattr(args, "resume_confirmed", False):
            raise ValueError(
                "--resume-confirmed requires a confirmed matching daily scene"
            )
    elif getattr(args, "resume_confirmed", False):
        raise ValueError("--resume-confirmed requires --daily-scene")

    scene_xml = output / "scene.xml"
    single._write_mjcf(
        scene_xml,
        records,
        runtime_profile,
        robot_qpos=qpos,
        supports=supports,
    )
    compile_report = {"ok": False}
    penetrations = []
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene_xml))
        data = mujoco.MjData(model)
        required = set(
            profile.get("robot_end_effector", {}).get(
                "required_visual_geoms", ()
            )
        )
        geom_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
            for index in range(model.ngeom)
        }
        missing = sorted(required - geom_names)
        forbidden = [
            name
            for name in profile.get("robot_end_effector", {}).get(
                "forbidden_bodies", ()
            )
            if name
            in {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
                for index in range(model.nbody)
            }
        ]
        compile_report = {
            "ok": not missing and not forbidden,
            "nq": int(model.nq),
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "missing_required_nyu_geoms": missing,
            "forbidden_stock_bodies": forbidden,
        }
        if calibration is not None and qpos is not None:
            penetrations = single._robot_environment_penetrations(
                model,
                data,
                profile=runtime_profile,
                keyframe="synchronized",
                tolerance_m=float(
                    profile.get(
                        "robot_environment_penetration_tolerance_m", 0.001
                    )
                ),
            )
    except Exception as error:
        compile_report = {"ok": False, "error": str(error)}

    single._write_mobile_view(
        output / "semantic_3d.html",
        records,
        observed,
        supports,
        camera_eye=single._viewer_camera_eye(profile),
        observed_surface_objects=observed_surface_objects,
    )
    mujoco_view = None
    try:
        render(
            scene_xml,
            output / "mujoco_home.html",
            keyframe="home",
            camera_eye=single._viewer_camera_eye(profile),
        )
        mujoco_view = str((output / "mujoco_home.html").resolve())
    except Exception as error:
        compile_report["mobile_render_error"] = str(error)

    source_viewer = Path(report["artifacts"]["viewer"]).resolve()
    copied_source_viewer = output / "source_esdf_scene.html"
    shutil.copy2(source_viewer, copied_source_viewer)
    esdf_link = output / "esdf_source.html"
    esdf_link.write_text(
        f"""<!doctype html><meta name="viewport" content="width=device-width">
<meta charset="utf-8"><title>ESDF provenance</title>
<style>body{{font:16px system-ui;margin:24px;background:#111827;color:white}}
code{{overflow-wrap:anywhere}}</style><h1>元の保守的ESDF</h1>
<p><a href="source_esdf_scene.html">インタラクティブESDF／点群を開く</a></p>
<p>座標系: {frame}</p><p>NPZ: <code>{source_esdf}</code></p>
<p>これはRGB-D観測空間のESDFです。未観測空間は衝突として扱います。</p>""",
        encoding="utf-8",
    )

    uncertain = [
        item["instance_id"]
        for item in records
        if item["status"] in {"uncertain", "display_only"}
    ]
    reasons = []
    if calibration is None:
        reasons.extend(
            [
                "camera_to_robot_extrinsic_not_accepted",
                "articulated_base_placement_unaccepted",
            ]
        )
    if qpos is None:
        reasons.append("synchronized_articulated_state_missing")
    if uncertain:
        reasons.append("operator_object_confirmation_required")
    if not compile_report.get("ok"):
        reasons.append("mujoco_compile_or_asset_regression_failed")
    if penetrations:
        reasons.append("robot_environment_penetration_over_1mm")
    collision_ready = not reasons
    readiness = {
        "display_ready": bool(compile_report.get("ok")),
        "collision_ready": collision_ready,
        "motion_ready": False,
        "reasons": reasons + ["motion_requires_independent_plan_validation"],
        "uncertain_instances": uncertain,
        "promotion_contract": {
            "accepted_camera_to_robot_calibration": calibration is not None,
            "synchronized_qpos": qpos is not None,
            "operator_scene_confirmation": not uncertain,
            "mujoco_compiles_with_pinned_nyu_grippers": bool(
                compile_report.get("ok")
            ),
            "maximum_penetration_m": (
                max(
                    item["penetration_depth_m"] for item in penetrations
                )
                if penetrations
                else 0.0
            ),
        },
    }
    scene = {
        "schema": SCHEMA,
        "created_at_s": time.time(),
        "inputs": {
            "multiview_report": {
                "path": str(report_path),
                "sha256": sha256_file(report_path),
            },
            "mesh": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
            "profile": str(Path(args.profile).resolve()),
        },
        "frame": frame,
        "frame_transform": transform_report,
        "robot_state": {
            "qpos": qpos,
            "source_view": qpos_view,
            "ordering": "left_joint1_to_6_then_right_joint1_to_6",
            "commands_sent": False,
        },
        "robot_placement": robot_placement,
        "objects": records,
        "supports": supports,
        "penetration_checks": penetrations,
        "mujoco_compile": compile_report,
        "readiness": readiness,
        "artifacts": {
            "index": str((output / "index.html").resolve()),
            "semantic_3d": str((output / "semantic_3d.html").resolve()),
            "mujoco": str(scene_xml.resolve()),
            "mujoco_mobile": mujoco_view,
            "source_esdf": str(source_esdf),
            "source_esdf_note": str(esdf_link.resolve()),
            "source_esdf_interactive": str(copied_source_viewer.resolve()),
        },
    }
    serializable = scene_json_ready(scene)
    scene_path = output / "scene.json"
    scene_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_index(output / "index.html", serializable)

    if consumed_daily is not None:
        serializable["daily_scene"] = {
            "path": str(Path(args.daily_scene).resolve()),
            "revision": consumed_daily.revision,
            "status": consumed_daily.status,
            "confirmed_by": consumed_daily.confirmed_by,
        }
        scene_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    elif getattr(args, "daily_scene", None):
        daily_objects = [
            SceneObject(
                instance_id=item["instance_id"],
                semantic_name=item["semantic_name"],
                geometry=item["geometry"],
                role=item.get("support_id"),
                confidence=float(item["confidence"]),
                status=(
                    "uncertain"
                    if item["status"] not in {"confirmed", "absent"}
                    else item["status"]
                ),
                source=item["source"],
                transparent=bool(item["transparent"]),
                depth_quality=(
                    "sparse_or_support_depth"
                    if item["transparent"]
                    else "multiview_observed"
                ),
            )
            for item in records
            if item["semantic_name"] != "measured_static_scene"
        ]
        proposed = DailySceneStore(args.daily_scene).propose(
            objects=daily_objects,
            calibration_id=expected_calibration_id,
            camera_ids={"rgbd": str(profile.get("camera_id", "record3d-head"))},
            images={
                "semantic 3D": str((output / "semantic_3d.html").resolve()),
                "MuJoCo": str((output / "mujoco_home.html").resolve()),
            },
            reason="sam_first_multiview_semantic_completion",
        )
        serializable["daily_scene"] = {
            "path": str(Path(args.daily_scene).resolve()),
            "revision": proposed.revision,
            "status": proposed.status,
        }
        scene_path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return serializable
