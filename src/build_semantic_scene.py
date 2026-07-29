#!/usr/bin/env python3
"""Build a completed semantic 3D scene from SAM masks and organized RGB-D."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import cv2
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.daily_scene import DailySceneStore, SceneObject
from rollout.sam_segmentation import (
    MaskCandidate,
    SamSegmentationClient,
    SegmentationResult,
)
from rollout.sam_segmentation import (
    compute_candidate_roi,
    extract_enlarged_roi,
    remap_segmentation_result_from_roi,
)
from rollout.semantic_scene_pipeline import (
    MaskObservation,
    SCHEMA,
    aabb_intersections,
    choose_support,
    conservative_scene_esdf,
    detect_unknown_objects,
    discover_supports,
    exclusive_masks,
    load_mask,
    load_organized_mesh,
    load_profile,
    quality_score,
    robust_oriented_geometry,
    scene_json_ready,
    sha256_file,
)
from src.render_mujoco_mobile import _box, _cylinder


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices]
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n")


def _compact_mesh(vertices: np.ndarray, faces: np.ndarray):
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], remap[faces]


def _primitive_mesh(record: dict):
    geometry = record["geometry"]
    size = np.asarray(geometry["size_xyz_m"], dtype=float)
    if geometry["kind"] == "cylinder":
        vertices, faces = _cylinder(size[0] / 2, size[2] / 2, segments=32)
    else:
        vertices, faces = _box(size / 2)
    yaw = float(geometry["yaw_rad"])
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    center = np.asarray(geometry["center_xyz_m"])
    return (rotation @ vertices.T).T + center, faces


def _parse_mask_specs(specs: list[str], profile: dict) -> list[MaskObservation]:
    result = []
    score = float(profile.get("accepted_mask_score", 1.0))
    for index, spec in enumerate(specs):
        label, separator, path = spec.partition("=")
        if not separator:
            raise ValueError(f"expected LABEL=PATH, got {spec!r}")
        result.append(
            MaskObservation(
                instance_id=f"{label}-{index + 1}",
                semantic_name=label,
                prompt="operator_accepted_mask",
                mask_path=str(Path(path).resolve()),
                sam_score=score,
                model="accepted_mask",
                inference_ms=0.0,
            )
        )
    return result


def _position_articulated_model(
    *,
    profile: dict,
    calibration_report: str | None,
    owned: list[tuple[MaskObservation, np.ndarray]],
    mesh: dict[str, np.ndarray],
    shape_hw: tuple[int, int],
    output_dir: Path,
) -> tuple[dict, dict]:
    """Place root bodies from named calibration anchors and SAM components."""

    runtime_profile = dict(profile)
    model_path = profile.get("robot_model")
    placement = profile.get("robot_placement")
    if not model_path:
        return runtime_profile, {
            "required": False,
            "accepted": True,
            "method": "no_articulated_model",
        }
    if not placement or not calibration_report:
        return runtime_profile, {
            "required": True,
            "accepted": False,
            "method": "model_default_only",
            "reason": "calibration_report_or_robot_placement_missing",
        }
    calibration = json.loads(Path(calibration_report).read_text())
    anchors = calibration.get(placement.get("anchor_map", "anchor_xyz_level_m"))
    if not isinstance(anchors, dict):
        raise ValueError("calibration report does not contain the robot anchor map")
    robot_masks = [
        mask
        for observation, mask in owned
        if observation.semantic_name == placement.get("semantic_name", "robot")
    ]
    if not robot_masks:
        return runtime_profile, {
            "required": True,
            "accepted": False,
            "method": "named_anchors_plus_sam_components",
            "reason": "robot_sam_mask_missing",
        }
    combined = np.logical_or.reduce(robot_masks)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined.astype(np.uint8), 8
    )
    instance_configs = list(placement["instances"])
    component_ids = sorted(
        range(1, count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )[: len(instance_configs)]
    if len(component_ids) < len(instance_configs):
        return runtime_profile, {
            "required": True,
            "accepted": False,
            "method": "named_anchors_plus_sam_components",
            "reason": "too_few_robot_components",
        }
    points_grid = mesh["vertices"].reshape(*shape_hw, 3)
    valid_grid = mesh["valid"].reshape(shape_hw)
    component_points = [
        points_grid[(labels == component) & valid_grid]
        for component in component_ids
    ]
    if any(len(points) < 3 for points in component_points):
        return runtime_profile, {
            "required": True,
            "accepted": False,
            "method": "named_anchors_plus_sam_components",
            "reason": "robot_component_depth_insufficient",
        }
    remaining = list(component_points)
    positions = {}
    for instance in instance_configs:
        anchor_name = str(instance["anchor"])
        anchor = _vector3(anchors[anchor_name], f"anchor {anchor_name}")
        selected_index = min(
            range(len(remaining)),
            key=lambda index: float(
                np.linalg.norm(
                    np.median(remaining[index][:, :2], axis=0) - anchor[:2]
                )
            ),
        )
        selected = remaining.pop(selected_index)
        anchor[2] = float(np.quantile(selected[:, 2], 0.01))
        positions[str(instance["body"])] = anchor
    first = positions[str(instance_configs[0]["body"])]
    second = positions[str(instance_configs[1]["body"])]
    baseline = first[:2] - second[:2]
    if np.linalg.norm(baseline) < 1e-6:
        raise ValueError("robot base anchors are coincident")
    yaw = float(
        np.arctan2(baseline[1], baseline[0])
        + float(placement.get("yaw_offset_rad", np.pi / 2))
    )
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    source = Path(model_path).resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(source.parent / "assets"))
    for body_name, position in positions.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"{source}: root body {body_name!r} not found")
        body.set("pos", " ".join(f"{value:.10f}" for value in position))
        body.set("euler", f"0 0 {yaw:.10f}")
    positioned = output_dir / "positioned_robot.xml"
    tree.write(positioned, encoding="unicode")
    runtime_profile["robot_model"] = str(positioned.resolve())
    return runtime_profile, {
        "required": True,
        "accepted": True,
        "method": "named_calibration_anchors_plus_sam_component_base_heights",
        "calibration_report": str(Path(calibration_report).resolve()),
        "source_model": str(source),
        "positioned_model": str(positioned.resolve()),
        "base_xyz_level_m": {
            name: value.tolist() for name, value in positions.items()
        },
        "shared_upright_yaw_rad": yaw,
    }


def _vector3(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite numbers")
    return result.copy()


def _run_sam(
    rgb: np.ndarray,
    profile: dict,
    catalog: dict,
    endpoint: str,
    output_dir: Path,
) -> list[MaskObservation]:
    client = SamSegmentationClient(
        endpoint, timeout_ms=int(profile.get("sam_timeout_ms", 20000))
    )
    observations = []

    def maximum_instances(name: str) -> int | None:
        raw = profile.get("maximum_instances", {}).get(name)
        return None if raw is None else max(1, int(raw))

    def box_iou(first: np.ndarray, second: np.ndarray) -> float:
        x0 = max(float(first[0]), float(second[0]))
        y0 = max(float(first[1]), float(second[1]))
        x1 = min(float(first[2]), float(second[2]))
        y1 = min(float(first[3]), float(second[3]))
        intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        first_area = max(0.0, float(first[2] - first[0])) * max(
            0.0, float(first[3] - first[1])
        )
        second_area = max(0.0, float(second[2] - second[0])) * max(
            0.0, float(second[3] - second[1])
        )
        union = first_area + second_area - intersection
        return 0.0 if union <= 0.0 else intersection / union

    def limited_candidates(
        candidates: tuple[MaskCandidate, ...], name: str
    ) -> tuple[MaskCandidate, ...]:
        limit = maximum_instances(name)
        if limit is None:
            return candidates
        selected = []
        threshold = float(profile.get("sam_instance_nms_iou", 0.55))
        for candidate in sorted(
            candidates, key=lambda item: item.score, reverse=True
        ):
            if any(
                box_iou(candidate.box_xyxy, kept.box_xyxy) >= threshold
                for kept in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= limit:
                break
        return tuple(selected)

    def tiled_fallback(
        prompt: str, name: str, frame_id: int
    ) -> tuple[SegmentationResult | None, int]:
        columns, rows = [
            int(item)
            for item in profile.get("sam_fallback_grid", [5, 4])
        ]
        fraction = float(profile.get("sam_fallback_window_fraction", 0.30))
        height, width = rgb.shape[:2]
        tile_width = max(32, int(round(width * fraction)))
        tile_height = max(32, int(round(height * fraction)))
        starts_x = np.rint(
            np.linspace(0, max(0, width - tile_width), columns)
        ).astype(int)
        starts_y = np.rint(
            np.linspace(0, max(0, height - tile_height), rows)
        ).astype(int)
        proposals = []
        inference_ms = 0.0
        model = "sam3"
        for y0 in starts_y:
            for x0 in starts_x:
                x1 = min(width, int(x0) + tile_width)
                y1 = min(height, int(y0) + tile_height)
                frame_id += 1
                result = client.segment(
                    rgb[int(y0):y1, int(x0):x1],
                    frame_id=frame_id,
                    timestamp=time.time(),
                    prompt=prompt,
                    confidence_threshold=float(
                        profile.get("sam_candidate_threshold", 0.20)
                    ),
                )
                inference_ms += float(result.inference_ms)
                model = result.model
                for candidate in result.candidates:
                    proposals.append(
                        (
                            float(candidate.score),
                            int(x0),
                            int(y0),
                            candidate,
                        )
                    )
        if not proposals:
            return None, frame_id
        candidates = []
        for _, x0, y0, candidate in sorted(
            proposals, key=lambda item: item[0], reverse=True
        ):
            full_mask = np.zeros((height, width), dtype=bool)
            tile_mask = np.asarray(candidate.mask, dtype=bool)
            tile_height_actual, tile_width_actual = tile_mask.shape
            full_mask[
                y0:y0 + tile_height_actual,
                x0:x0 + tile_width_actual,
            ] = tile_mask
            offset = np.array([x0, y0, x0, y0], dtype=float)
            candidates.append(
                MaskCandidate(
                    mask=full_mask,
                    box_xyxy=np.asarray(candidate.box_xyxy, dtype=float)
                    + offset,
                    score=float(candidate.score),
                )
            )
        candidates = list(limited_candidates(tuple(candidates), name))
        return (
            SegmentationResult(
                frame_id=frame_id,
                source_timestamp=time.time(),
                model=model,
                inference_ms=inference_ms,
                candidates=tuple(candidates),
            ),
            frame_id,
        )

    try:
        frame_id = 0
        for name in profile.get("objects", list(catalog)):
            definition = catalog[name]
            best_result = None
            best_prompt = None
            for prompt in definition.prompts:
                frame_id += 1
                print(
                    f"SAM coarse: object={name} prompt={prompt!r}",
                    flush=True,
                )
                coarse = client.segment(
                    rgb,
                    frame_id=frame_id,
                    timestamp=time.time(),
                    prompt=prompt,
                    confidence_threshold=float(
                        profile.get("sam_candidate_threshold", 0.20)
                    ),
                )
                retry_count = int(profile.get("sam_full_frame_attempts", 3))
                for attempt in range(2, retry_count + 1):
                    if coarse.candidates:
                        break
                    frame_id += 1
                    print(
                        "SAM coarse retry: "
                        f"object={name} prompt={prompt!r} "
                        f"attempt={attempt}",
                        flush=True,
                    )
                    coarse = client.segment(
                        rgb,
                        frame_id=frame_id,
                        timestamp=time.time(),
                        prompt=prompt,
                        confidence_threshold=float(
                            profile.get("sam_candidate_threshold", 0.20)
                        ),
                    )
                if (
                    not coarse.candidates
                    and name
                    in set(profile.get("sam_tiled_fallback_objects", ()))
                ):
                    print(
                        f"SAM tiled fallback: object={name} prompt={prompt!r}",
                        flush=True,
                    )
                    tiled, frame_id = tiled_fallback(prompt, name, frame_id)
                    if tiled is not None:
                        coarse = tiled
                result = coarse
                if coarse.candidates:
                    try:
                        refinement_candidates = limited_candidates(
                            tuple(coarse.candidates), name
                        )
                        roi = compute_candidate_roi(
                            refinement_candidates,
                            full_shape_hw=rgb.shape[:2],
                            padding_px=float(profile.get("sam_roi_padding_px", 24.0)),
                            scale=float(profile.get("sam_roi_scale", 4.0)),
                        )
                        x0, y0, x1, y1 = roi.crop_xyxy
                        crop_fraction = (
                            (x1 - x0) * (y1 - y0)
                            / float(rgb.shape[0] * rgb.shape[1])
                        )
                        if crop_fraction > float(
                            profile.get(
                                "sam_roi_max_full_frame_fraction", 0.35
                            )
                        ):
                            raise ValueError(
                                "candidate union is too large for useful ROI "
                                f"refinement ({crop_fraction:.3f} of frame)"
                            )
                        enlarged = extract_enlarged_roi(rgb, roi)
                        frame_id += 1
                        print(
                            "SAM fine: "
                            f"object={name} prompt={prompt!r} "
                            f"shape={enlarged.shape[:2]}",
                            flush=True,
                        )
                        fine = client.segment(
                            enlarged,
                            frame_id=frame_id,
                            timestamp=time.time(),
                            prompt=prompt,
                            confidence_threshold=float(
                                profile.get("sam_candidate_threshold", 0.20)
                            ),
                        )
                        if fine.candidates:
                            result, _ = remap_segmentation_result_from_roi(fine, roi)
                    except ValueError:
                        # A full-frame or already-large candidate needs no ROI.
                        result = coarse
                if result.candidates:
                    candidate_score = max(
                        item.score for item in result.candidates
                    )
                    current_score = (
                        -1.0
                        if best_result is None
                        else max(item.score for item in best_result.candidates)
                    )
                    if candidate_score > current_score:
                        best_result = result
                        best_prompt = prompt
            if best_result is None:
                continue
            candidates = limited_candidates(
                tuple(best_result.candidates), name
            )
            for instance_index, candidate in enumerate(
                sorted(candidates, key=lambda item: item.score, reverse=True),
                1,
            ):
                path = output_dir / "masks" / f"{name}-{instance_index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(path), candidate.mask.astype(np.uint8) * 255
                ):
                    raise RuntimeError(f"failed to write {path}")
                observations.append(
                    MaskObservation(
                        instance_id=f"{name}-{instance_index}",
                        semantic_name=name,
                        prompt=str(best_prompt),
                        mask_path=str(path.resolve()),
                        sam_score=float(candidate.score),
                        model=str(best_result.model),
                        inference_ms=float(best_result.inference_ms),
                    )
                )
    finally:
        client.close()
    return observations


def _anchor_matched_observations(
    observations: list[MaskObservation],
    *,
    profile: dict,
    calibration_report: str | None,
    mesh: dict[str, np.ndarray],
    shape_hw: tuple[int, int],
) -> list[MaskObservation]:
    """Select semantic instances nearest named 3D anchors.

    SAM confidence alone cannot distinguish a robot arm from a visually
    similar microscope strut or a bottle cap from a gripper cap.  Anchors are
    projected into the organized RGB-D image, then a minimum-cost one-to-one
    assignment selects masks by image distance with confidence as a tie-break.
    """

    if not calibration_report:
        return observations
    report = json.loads(Path(calibration_report).read_text())
    anchors_by_semantic: dict[str, list[np.ndarray]] = {}
    placement = profile.get("robot_placement", {})
    robot_name = placement.get("semantic_name")
    anchor_map = report.get(placement.get("anchor_map", ""), {})
    robot_anchors = [
        anchor_map.get(item.get("anchor"))
        for item in placement.get("instances", ())
    ]
    if robot_name and all(item is not None for item in robot_anchors):
        anchors_by_semantic[str(robot_name)] = [
            _vector3(item, f"{robot_name} anchor")
            for item in robot_anchors
        ]

    for semantic_name, paths in profile.get(
        "semantic_anchor_selection", {}
    ).items():
        selected = []
        for path in paths:
            value = report
            try:
                for key in path:
                    value = value[key]
                selected.append(_vector3(value, f"{semantic_name} anchor"))
            except (KeyError, TypeError, ValueError):
                continue
        if selected:
            anchors_by_semantic[str(semantic_name)] = selected

    vertices = mesh["vertices"]
    valid_indices = np.flatnonzero(mesh["valid"])
    valid_vertices = vertices[valid_indices]
    diagonal = float(np.hypot(*shape_hw))
    retained = list(observations)
    for semantic_name, anchors in anchors_by_semantic.items():
        candidates = [
            item for item in observations
            if item.semantic_name == semantic_name
        ]
        if len(candidates) <= len(anchors):
            continue
        anchor_pixels = []
        for anchor in anchors:
            delta = valid_vertices - anchor
            # Image association should be dominated by lateral placement; Z
            # remains a weak cue because the visible mask rarely reaches a
            # base hidden below a platform.
            distance = (
                delta[:, 0] ** 2
                + delta[:, 1] ** 2
                + 0.10 * delta[:, 2] ** 2
            )
            flat_index = int(valid_indices[int(np.argmin(distance))])
            row, column = np.unravel_index(flat_index, shape_hw)
            anchor_pixels.append(np.array([column, row], dtype=float))
        costs = np.zeros((len(anchors), len(candidates)), dtype=float)
        for candidate_index, candidate in enumerate(candidates):
            mask = load_mask(candidate.mask_path, shape_hw)
            rows, columns = np.nonzero(mask)
            if not len(columns):
                costs[:, candidate_index] = 1e6
                continue
            pixels = np.column_stack((columns, rows)).astype(float)
            for anchor_index, anchor_pixel in enumerate(anchor_pixels):
                pixel_distance = np.min(
                    np.linalg.norm(pixels - anchor_pixel, axis=1)
                ) / max(diagonal, 1.0)
                costs[anchor_index, candidate_index] = (
                    pixel_distance - 0.02 * candidate.sam_score
                )
        best = None
        for assignment in itertools.permutations(
            range(len(candidates)), len(anchors)
        ):
            cost = sum(
                costs[anchor_index, candidate_index]
                for anchor_index, candidate_index in enumerate(assignment)
            )
            if best is None or cost < best[0]:
                best = (cost, assignment)
        selected_ids = {
            id(candidates[index]) for index in best[1]
        }
        retained = [
            item for item in retained
            if item.semantic_name != semantic_name or id(item) in selected_ids
        ]
        print(
            "Anchor selection: "
            f"object={semantic_name} kept={len(selected_ids)} "
            f"from={len(candidates)}",
            flush=True,
        )
    return retained


def _overlay(rgb, owned, objects, unknown, output: Path):
    image = rgb.copy()
    object_by_id = {item["instance_id"]: item for item in objects}
    palette = [
        (38, 198, 218), (225, 51, 209), (47, 210, 91),
        (242, 215, 47), (26, 140, 255), (192, 64, 255),
    ]
    for index, (observation, mask) in enumerate(owned):
        record = object_by_id.get(observation.instance_id)
        if record is None:
            continue
        color = palette[index % len(palette)]
        layer = image.copy()
        layer[mask] = color
        image = cv2.addWeighted(image, 0.72, layer, 0.28, 0)
        rows, columns = np.nonzero(mask)
        if len(rows):
            cv2.putText(
                image,
                f"{observation.semantic_name} {record['confidence']:.2f}",
                (int(columns.min()), max(24, int(rows.min()))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
    for item in unknown:
        contour, _ = cv2.findContours(
            item["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(image, contour, -1, (0, 0, 255), 3)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"failed to write {output}")


def _write_mjcf(
    path: Path,
    objects: list[dict],
    profile: dict,
    robot_qpos: list[float] | None = None,
    supports: list[dict] | None = None,
) -> None:
    includes = ""
    robot_model = profile.get("robot_model")
    compiler = '  <compiler angle="radian" autolimits="true"/>\n'
    if robot_model:
        robot_path = Path(robot_model).resolve()
        includes = f'  <include file="{escape(str(robot_path))}"/>\n'
        robot_root = ET.parse(robot_path).getroot()
        robot_compiler = robot_root.find("compiler")
        configured_meshdir = (
            "assets"
            if robot_compiler is None
            else robot_compiler.get("meshdir", "assets")
        )
        meshdir = Path(configured_meshdir)
        if not meshdir.is_absolute():
            meshdir = (robot_path.parent / meshdir).resolve()
        compiler = (
            '  <compiler angle="radian" autolimits="true" '
            f'meshdir="{escape(str(meshdir))}"/>\n'
        )
    geoms = []
    for record in objects:
        if record.get("completion") == "exact_cad" and robot_model:
            # The included articulated model is authoritative.  A SAM-derived
            # AABB remains in scene.json as an observation, never as duplicate
            # robot collision geometry.
            continue
        geometry = record["geometry"]
        center = geometry["center_xyz_m"]
        size = geometry["size_xyz_m"]
        yaw = geometry["yaw_rad"]
        rgba = record.get("rgba", [0.45, 0.62, 0.78, 0.84])
        rgba_text = " ".join(f"{float(item):.4f}" for item in rgba)
        if geometry["kind"] == "cylinder":
            geom = (
                f'<geom type="cylinder" size="{size[0] / 2:.8f} {size[2] / 2:.8f}"'
            )
        else:
            geom = (
                f'<geom type="box" size="{size[0] / 2:.8f} '
                f'{size[1] / 2:.8f} {size[2] / 2:.8f}"'
            )
        geoms.append(
            f'    <body name="{escape(record["instance_id"])}" '
            f'pos="{center[0]:.8f} {center[1]:.8f} {center[2]:.8f}" '
            f'euler="0 0 {yaw:.8f}">\n'
            f'      {geom} rgba="{rgba_text}" contype="1" conaffinity="1"/>\n'
            "    </body>"
        )
    support_thickness = float(profile.get("support_collision_thickness_m", 0.025))
    for support in supports or []:
        lower, upper = np.asarray(support["bounds_xy_m"], dtype=float)
        half_xy = np.maximum((upper - lower) / 2, 0.002)
        center_xy = (lower + upper) / 2
        center_z = float(support["height_m"]) - support_thickness / 2
        geoms.append(
            f'    <body name="{escape(support["support_id"])}" '
            f'pos="{center_xy[0]:.8f} {center_xy[1]:.8f} {center_z:.8f}">\n'
            f'      <geom type="box" size="{half_xy[0]:.8f} '
            f'{half_xy[1]:.8f} {support_thickness / 2:.8f}" '
            'rgba="0.18 0.21 0.25 0.75" contype="1" conaffinity="1" '
            'friction="1 0.01 0.01"/>\n'
            "    </body>"
        )
    keyframe = ""
    if robot_qpos is not None:
        keyframe = (
            '  <keyframe><key name="synchronized" qpos="'
            + " ".join(f"{float(item):.10f}" for item in robot_qpos)
            + '"/></keyframe>\n'
        )
    path.write_text(
        '<mujoco model="sam_first_semantic_scene">\n'
        f"{includes}"
        f"{compiler}"
        "  <worldbody>\n"
        + "\n".join(geoms)
        + "\n  </worldbody>\n"
        + keyframe
        + "</mujoco>\n"
    )


def _write_mobile_view(
    path: Path,
    objects: list[dict],
    observed: list[dict],
    supports: list[dict] | None = None,
) -> None:
    traces = []
    for record in objects:
        vertices, faces = _primitive_mesh(record)
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                name=f"inferred: {record['semantic_name']}",
                color=record["color"], opacity=0.72, flatshading=False,
                visible=True,
            )
        )
    for support in supports or []:
        lower, upper = np.asarray(support["bounds_xy_m"], dtype=float)
        thickness = 0.025
        record = {
            "geometry": {
                "kind": "box",
                "center_xyz_m": [
                    *((lower + upper) / 2).tolist(),
                    float(support["height_m"]) - thickness / 2,
                ],
                "size_xyz_m": [
                    *(upper - lower).tolist(),
                    thickness,
                ],
                "yaw_rad": 0.0,
            }
        }
        vertices, faces = _primitive_mesh(record)
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                name=support["support_id"],
                color="#343b46", opacity=0.72, flatshading=False,
                visible=True,
            )
        )
    for surface in observed:
        vertices = surface["vertices"]
        faces = surface["faces"]
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                name=f"observed: {surface['semantic_name']}",
                color="#1f2937", opacity=0.28, flatshading=False,
                visible=True,
            )
        )
    inferred_count = len(objects) + len(supports or [])
    observed_count = len(observed)
    figure = go.Figure(traces)
    figure.update_layout(
        title="SAM-first semantic scene — inferred + observed",
        paper_bgcolor="#f8fafc",
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "両方",
                    "method": "update",
                    "args": [
                        {"visible": [True] * (inferred_count + observed_count)}
                    ],
                },
                {
                    "label": "推論形状",
                    "method": "update",
                    "args": [
                        {
                            "visible": [True] * inferred_count
                            + [False] * observed_count
                        }
                    ],
                },
                {
                    "label": "観測面",
                    "method": "update",
                    "args": [
                        {
                            "visible": [False] * inferred_count
                            + [True] * observed_count
                        }
                    ],
                },
            ],
        }],
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "X / right (m)"},
            "yaxis": {"title": "Y / forward (m)"},
            "zaxis": {"title": "Z / up (m)"},
            "camera": {"eye": {"x": 1.35, "y": -1.55, "z": 0.95}},
        },
    )
    figure.write_html(
        path, include_plotlyjs=True, full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )


def _write_esdf_mobile(path: Path, esdf: dict) -> None:
    origin = np.asarray(esdf["origin_xyz_m"], dtype=float)
    voxel = float(esdf["voxel_size_m"])
    observed_free = np.asarray(esdf["observed"]) & ~np.asarray(esdf["occupied"])
    frontier = np.asarray(esdf["unknown_frontier"])

    def sampled_points(mask, maximum=18000):
        indices = np.argwhere(mask)
        if len(indices) > maximum:
            indices = indices[:: int(np.ceil(len(indices) / maximum))]
        return origin + (indices + 0.5) * voxel, indices

    free_points, free_indices = sampled_points(observed_free)
    frontier_points, _ = sampled_points(frontier)
    clearance = (
        np.asarray(esdf["esdf_m"])[tuple(free_indices.T)]
        if len(free_indices)
        else np.empty(0)
    )
    traces = [
        go.Scatter3d(
            x=free_points[:, 0], y=free_points[:, 1], z=free_points[:, 2],
            mode="markers", name="observed free",
            marker={
                "size": 2,
                "color": clearance * 1000,
                "colorscale": "Turbo",
                "colorbar": {"title": "clearance mm"},
                "opacity": 0.62,
            },
        ),
        go.Scatter3d(
            x=frontier_points[:, 0],
            y=frontier_points[:, 1],
            z=frontier_points[:, 2],
            mode="markers",
            name="unknown boundary (collision)",
            marker={"size": 2, "color": "#a855f7", "opacity": 0.72},
        ),
    ]
    figure = go.Figure(traces)
    figure.update_layout(
        title="Conservative ESDF — unknown space remains collision",
        paper_bgcolor="#111827",
        font={"color": "#f9fafb"},
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "X / right (m)"},
            "yaxis": {"title": "Y / forward (m)"},
            "zaxis": {"title": "Z / up (m)"},
            "camera": {"eye": {"x": 1.35, "y": -1.55, "z": 0.95}},
        },
    )
    figure.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )


def _model_collision_records(model, data) -> list[dict]:
    records = []
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) != 3:
            continue
        aabb = np.asarray(model.geom_aabb[geom_id], dtype=float)
        local_center = aabb[:3]
        local_half = aabb[3:6]
        geom_rotation = np.asarray(
            data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        world_center = (
            geom_rotation @ local_center
            + np.asarray(data.geom_xpos[geom_id], dtype=float)
        )
        world_half = np.abs(geom_rotation) @ local_half
        records.append(
            {
                "instance_id": f"robot-collision-geom-{geom_id}",
                "geometry": {
                    "center_xyz_m": world_center.tolist(),
                    "size_xyz_m": (2 * world_half).tolist(),
                },
            }
        )
    return records


def _write_esdf_artifacts(
    *,
    output: Path,
    mesh: dict,
    records: list[dict],
    supports: list[dict],
    profile: dict,
    camera_origin: np.ndarray | None,
    model=None,
    data=None,
) -> dict:
    esdf_objects = [
        item for item in records if item.get("completion") != "exact_cad"
    ]
    if model is not None and data is not None:
        esdf_objects.extend(_model_collision_records(model, data))
    esdf = conservative_scene_esdf(
        vertices=mesh["vertices"],
        valid=mesh["valid"],
        objects=esdf_objects,
        supports=supports,
        camera_origin_m=camera_origin,
        voxel_size_m=float(profile.get("esdf_voxel_size_m", 0.02)),
        maximum_voxels=int(profile.get("esdf_maximum_voxels", 2_000_000)),
    )
    np.savez_compressed(
        output / "scene_esdf.npz",
        **{key: value for key, value in esdf.items() if key != "schema"},
    )
    _write_esdf_mobile(output / "esdf.html", esdf)
    return {
        "schema": esdf["schema"],
        "shape": list(np.asarray(esdf["esdf_m"]).shape),
        "voxel_size_m": float(esdf["voxel_size_m"]),
        "observed_fraction": float(np.mean(esdf["observed"])),
        "occupied_fraction": float(np.mean(esdf["occupied"])),
        "unknown_collision_fraction": float(
            np.mean(esdf["unknown_collision"])
        ),
    }


def _camera_origin_from_report(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    transform = payload.get("T_level_camera")
    return (
        None
        if transform is None
        else np.asarray(transform, dtype=float)[:3, 3]
    )


def _read_obj(path: str | Path) -> dict:
    vertices = []
    faces = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(item) for item in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append(
                [int(item.split("/")[0]) - 1 for item in line.split()[1:4]]
            )
    return {
        "vertices": np.asarray(vertices, dtype=float),
        "faces": np.asarray(faces, dtype=np.int32),
    }


def _resume_confirmed(args) -> dict:
    if not args.daily_scene:
        raise ValueError("--resume-confirmed requires --daily-scene")
    scene_path = Path(args.output_dir) / "scene.json"
    if not scene_path.exists():
        raise FileNotFoundError(
            f"{scene_path}: run the SAM-first reconstruction before resuming"
        )
    scene = json.loads(scene_path.read_text())
    store = DailySceneStore(args.daily_scene)
    daily = store.require_confirmed(
        calibration_id=scene.get("daily_scene", {}).get("calibration_id")
    )
    if scene.get("daily_scene", {}).get("revision") != daily.revision:
        raise ValueError("confirmed daily scene revision does not match reconstruction")
    profile, catalog = load_profile(args.profile)
    accepted = {item.instance_id: item for item in daily.objects}
    retained = []
    for record in scene["objects"]:
        item = accepted.get(record["instance_id"])
        if item is None:
            raise ValueError(
                f"confirmed scene is missing {record['instance_id']}"
            )
        if item.status == "absent":
            continue
        record["semantic_name"] = item.semantic_name
        record["status"] = item.status
        record["geometry"] = item.geometry
        record["support_id"] = item.role
        record["confidence"] = max(record["confidence"], item.confidence)
        definition = catalog.get(item.semantic_name)
        if definition is not None:
            record["completion"] = definition.completion
            record["color"] = definition.color
            record["transparent"] = definition.transparent
            if definition.primitive:
                record["geometry"]["kind"] = definition.primitive
        retained.append(record)
    scene["objects"] = retained
    if any(item["status"] != "confirmed" for item in retained):
        raise ValueError("every object must be confirmed before resume")
    intersections = aabb_intersections(scene["objects"])
    scene["intersections"] = intersections
    extrinsic = bool(
        json.loads(Path(args.profile).read_text()).get(
            "accepted_camera_to_robot", False
        )
    )
    scene["readiness"] = {
        "display_ready": True,
        "collision_ready": not intersections,
        "motion_ready": not intersections and extrinsic,
        "reasons": (
            (["completed_geometry_intersection"] if intersections else [])
            + (
                []
                if extrinsic
                else ["camera_to_robot_extrinsic_not_accepted"]
            )
        ),
    }
    positioned_model = scene.get("robot_placement", {}).get(
        "positioned_model"
    )
    if positioned_model:
        profile["robot_model"] = positioned_model
    _write_mjcf(
        Path(args.output_dir) / "scene.xml",
        scene["objects"],
        profile,
        scene.get("robot_state", {}).get("qpos"),
        scene.get("supports", []),
    )
    model = None
    data = None
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(
            str(Path(args.output_dir) / "scene.xml")
        )
        data = mujoco.MjData(model)
        qpos = scene.get("robot_state", {}).get("qpos")
        if qpos is not None:
            mujoco.mj_resetDataKeyframe(
                model, data, int(model.key("synchronized").id)
            )
        mujoco.mj_forward(model, data)
        scene["mujoco_compile"] = {
            "ok": True,
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nq": int(model.nq),
        }
        try:
            from src.render_mujoco_mobile import render

            render(
                Path(args.output_dir) / "scene.xml",
                Path(args.output_dir) / "mujoco.html",
                keyframe="synchronized" if qpos is not None else None,
            )
            scene["artifacts"]["mujoco_mobile_view"] = str(
                (Path(args.output_dir) / "mujoco.html").resolve()
            )
        except Exception as error:
            scene["mujoco_compile"]["mobile_render_error"] = str(error)
        if model.nq:
            try:
                from src.render_mujoco_articulation import render_articulation

                articulation = render_articulation(
                    Path(args.output_dir) / "scene.xml",
                    Path(args.output_dir) / "articulation.mp4",
                )
                scene["articulation_check"] = articulation
                scene["artifacts"]["articulation_video"] = articulation.get(
                    "path"
                )
            except Exception as error:
                scene["articulation_check"] = {
                    "ok": False,
                    "error": str(error),
                }
    except Exception as error:
        scene["readiness"]["collision_ready"] = False
        scene["readiness"]["motion_ready"] = False
        scene["readiness"]["reasons"].append("mujoco_compile_failed")
        scene["mujoco_compile"] = {"ok": False, "error": str(error)}
    observed = []
    for record in scene["objects"]:
        if record.get("observed_mesh"):
            mesh = _read_obj(record["observed_mesh"])
            observed.append(
                {
                    "semantic_name": record["semantic_name"],
                    **mesh,
                }
            )
    _write_mobile_view(
        Path(args.output_dir) / "index.html",
        scene["objects"],
        observed,
        scene.get("supports", []),
    )
    mesh = load_organized_mesh(scene["inputs"]["mesh"]["path"])
    calibration_report = scene.get("robot_placement", {}).get(
        "calibration_report"
    )
    scene["esdf"] = _write_esdf_artifacts(
        output=Path(args.output_dir),
        mesh=mesh,
        records=scene["objects"],
        supports=scene.get("supports", []),
        profile=profile,
        camera_origin=_camera_origin_from_report(calibration_report),
        model=model,
        data=data,
    )
    scene["artifacts"]["esdf"] = str(
        (Path(args.output_dir) / "scene_esdf.npz").resolve()
    )
    scene["artifacts"]["esdf_mobile_view"] = str(
        (Path(args.output_dir) / "esdf.html").resolve()
    )
    scene["resumed_from_confirmed_revision"] = daily.revision
    scene["daily_scene"]["status"] = daily.status
    scene["daily_scene"]["confirmed_by"] = daily.confirmed_by
    scene_path.write_text(json.dumps(scene, indent=2) + "\n")
    return scene


def build(args) -> dict:
    if args.resume_confirmed:
        return _resume_confirmed(args)
    if not args.rgb or not args.mesh:
        raise ValueError("--rgb and --mesh are required for reconstruction")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    profile, catalog = load_profile(args.profile)
    robot_qpos = None
    robot_state_source = None
    if args.capture:
        capture_manifest = Path(args.capture) / "manifest.json"
        if capture_manifest.exists():
            capture_payload = json.loads(capture_manifest.read_text())
            state = capture_payload.get("robot_state", {}).get("after", {})
            left_q = state.get("left_joint_positions_rad")
            right_q = state.get("right_joint_positions_rad")
            if left_q is not None and right_q is not None:
                robot_qpos = [
                    *[float(item) for item in left_q],
                    *[float(item) for item in right_q],
                ]
                robot_state_source = str(capture_manifest.resolve())
    rgb_full = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if rgb_full is None:
        raise FileNotFoundError(args.rgb)
    mesh = load_organized_mesh(args.mesh)
    shape_hw = tuple(
        int(item)
        for item in profile.get("organized_shape_hw", rgb_full.shape[:2])
    )
    rgb_organized = (
        rgb_full
        if rgb_full.shape[:2] == shape_hw
        else cv2.resize(
            rgb_full,
            (shape_hw[1], shape_hw[0]),
            interpolation=cv2.INTER_AREA,
        )
    )
    if len(mesh["vertices"]) != int(np.prod(shape_hw)):
        raise ValueError("profile organized_shape_hw does not match mesh")

    observations = (
        _parse_mask_specs(args.mask, profile)
        if args.mask
        else _run_sam(rgb_full, profile, catalog, args.sam_endpoint, output)
    )
    if not observations:
        raise RuntimeError("SAM did not detect any configured object")
    observations = _anchor_matched_observations(
        observations,
        profile=profile,
        calibration_report=getattr(args, "calibration_report", None),
        mesh=mesh,
        shape_hw=shape_hw,
    )
    owned = exclusive_masks(observations, shape_hw)
    profile, robot_placement = _position_articulated_model(
        profile=profile,
        calibration_report=getattr(args, "calibration_report", None),
        owned=owned,
        mesh=mesh,
        shape_hw=shape_hw,
        output_dir=output,
    )
    supports = discover_supports(
        mesh["vertices"], mesh["valid"], shape_hw,
        height_tolerance_m=float(profile.get("support_height_tolerance_m", 0.008)),
        minimum_area_fraction=float(profile.get("support_minimum_area_fraction", 0.004)),
    )
    faces = mesh["faces"]
    valid_image = mesh["valid"].reshape(shape_hw)
    claimed = np.zeros(shape_hw, dtype=bool)
    records = []
    observed_surfaces = []
    for observation, mask in owned:
        claimed |= mask
        vertex_mask = mask.reshape(-1) & mesh["valid"]
        points = mesh["vertices"][vertex_mask]
        minimum_depth_points = int(profile.get("minimum_depth_points", 12))
        if len(points) < 3:
            continue
        definition = catalog.get(observation.semantic_name)
        support = choose_support(points, supports)
        geometry = robust_oriented_geometry(
            points,
            catalog=definition,
            support_height_m=None if support is None else support["height_m"],
        )
        confidence, terms = quality_score(
            sam_score=observation.sam_score,
            mask=mask,
            valid_depth=valid_image,
            geometry=geometry,
            catalog=definition,
            support_found=support is not None,
        )
        if len(points) < minimum_depth_points:
            sample_fraction = len(points) / max(1, minimum_depth_points)
            confidence *= sample_fraction
            terms["depth_sample_count"] = int(len(points))
            terms["minimum_depth_points"] = minimum_depth_points
            terms["depth_sample_factor"] = float(sample_fraction)
        threshold = (
            definition.minimum_confidence
            if definition is not None
            else float(profile.get("default_minimum_confidence", 0.72))
        )
        selected_faces = faces[np.all(vertex_mask[faces], axis=1)]
        observed_path = None
        if len(selected_faces):
            object_vertices, object_faces = _compact_mesh(
                mesh["vertices"], selected_faces
            )
            observed_path = output / "observed" / f"{observation.instance_id}.obj"
            observed_path.parent.mkdir(parents=True, exist_ok=True)
            _write_obj(observed_path, object_vertices, object_faces)
            observed_surfaces.append(
                {
                    "semantic_name": observation.semantic_name,
                    "vertices": object_vertices,
                    "faces": object_faces,
                }
            )
        record = {
            "instance_id": observation.instance_id,
            "semantic_name": observation.semantic_name,
            "status": "confirmed" if confidence >= threshold else "uncertain",
            "confidence": confidence,
            "minimum_confidence": threshold,
            "source": "sam_rgbd",
            "transparent": bool(definition.transparent if definition else False),
            "mask_path": observation.mask_path,
            "observed_mesh": None if observed_path is None else str(observed_path.resolve()),
            "support_id": None if support is None else support["support_id"],
            "completion": (
                definition.completion if definition else "primitive"
            ),
            "geometry": asdict(geometry),
            "color": definition.color if definition else "#8b9bb4",
            "sam": {
                "prompt": observation.prompt,
                "score": observation.sam_score,
                "model": observation.model,
                "inference_ms": observation.inference_ms,
            },
            "quality": terms,
        }
        if record["completion"] == "exact_cad":
            record["pose_source"] = (
                "synchronized_capture_joint_state"
                if robot_qpos is not None
                else "model_default_unverified"
            )
            record["model_source"] = (
                robot_placement.get("positioned_model")
                or robot_placement.get("source_model")
                or (definition.model if definition else None)
            )
        records.append(record)

    unknown = detect_unknown_objects(
        vertices=mesh["vertices"],
        valid=mesh["valid"],
        shape_hw=shape_hw,
        claimed=claimed,
        supports=supports,
        minimum_area_fraction=float(profile.get("unknown_minimum_area_fraction", 0.008)),
    )
    for item in unknown:
        geometry = robust_oriented_geometry(
            item["points"],
            catalog=None,
            support_height_m=item["support"]["height_m"],
        )
        mask_path = output / "masks" / f"{item['instance_id']}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), item["mask"].astype(np.uint8) * 255)
        records.append(
            {
                "instance_id": item["instance_id"],
                "semantic_name": "unknown object",
                "status": "uncertain",
                "confidence": 0.0,
                "minimum_confidence": 1.0,
                "source": "unclaimed_rgbd_component",
                "transparent": False,
                "mask_path": str(mask_path.resolve()),
                "observed_mesh": None,
                "support_id": item["support"]["support_id"],
                "completion": "primitive",
                "geometry": asdict(geometry),
                "color": "#ef4444",
                "quality": {"reason": "requires_operator_identity"},
            }
        )

    intersections = aabb_intersections(records)
    all_automatic = all(item["status"] == "confirmed" for item in records)
    accepted_extrinsic = bool(profile.get("accepted_camera_to_robot", False))
    articulated_state_ready = (
        not profile.get("robot_model") or robot_qpos is not None
    )
    articulated_placement_ready = bool(robot_placement["accepted"])
    readiness = {
        "display_ready": bool(records),
        "collision_ready": (
            bool(records)
            and not intersections
            and all_automatic
            and articulated_state_ready
            and articulated_placement_ready
        ),
        "motion_ready": (
            bool(records)
            and not intersections
            and all_automatic
            and articulated_state_ready
            and articulated_placement_ready
            and accepted_extrinsic
        ),
        "reasons": [],
    }
    if not all_automatic:
        readiness["reasons"].append("operator_confirmation_required")
    if intersections:
        readiness["reasons"].append("completed_geometry_intersection")
    if not accepted_extrinsic:
        readiness["reasons"].append("camera_to_robot_extrinsic_not_accepted")
    if not articulated_state_ready:
        readiness["reasons"].append("synchronized_articulated_state_missing")
    if not articulated_placement_ready:
        readiness["reasons"].append("articulated_base_placement_unaccepted")

    _overlay(
        rgb_organized,
        owned,
        records,
        unknown,
        output / "sam_overlay.png",
    )
    _write_mobile_view(
        output / "index.html", records, observed_surfaces, supports
    )
    _write_mjcf(
        output / "scene.xml", records, profile, robot_qpos, supports
    )
    compile_result = {"ok": False}
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
        compile_result = {
            "ok": True,
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nq": int(model.nq),
        }
    except Exception as error:
        compile_result["error"] = str(error)
        readiness["collision_ready"] = False
        readiness["motion_ready"] = False
        readiness["reasons"].append("mujoco_compile_failed")
        model = None
    scene_mobile = None
    articulation = {"ok": False, "reason": "mujoco_compile_failed"}
    if model is not None:
        try:
            from src.render_mujoco_mobile import render

            render(
                output / "scene.xml",
                output / "mujoco.html",
                keyframe="synchronized" if robot_qpos is not None else None,
            )
            scene_mobile = str((output / "mujoco.html").resolve())
        except Exception as error:
            compile_result["mobile_render_error"] = str(error)
        if model.nq:
            try:
                from src.render_mujoco_articulation import render_articulation

                articulation = render_articulation(
                    output / "scene.xml", output / "articulation.mp4"
                )
            except Exception as error:
                articulation = {"ok": False, "error": str(error)}
        else:
            articulation = {"ok": False, "reason": "model_has_no_dof"}

    data = None
    if model is not None:
        data = mujoco.MjData(model)
        if robot_qpos is not None:
            mujoco.mj_resetDataKeyframe(
                model, data, int(model.key("synchronized").id)
            )
        mujoco.mj_forward(model, data)
    esdf_report = _write_esdf_artifacts(
        output=output,
        mesh=mesh,
        records=records,
        supports=supports,
        profile=profile,
        camera_origin=_camera_origin_from_report(
            getattr(args, "calibration_report", None)
        ),
        model=model,
        data=data,
    )

    scene = {
        "schema": SCHEMA,
        "created_at_s": time.time(),
        "capture": None if args.capture is None else str(Path(args.capture).resolve()),
        "inputs": {
            "rgb": {"path": str(Path(args.rgb).resolve()), "sha256": sha256_file(args.rgb)},
            "mesh": {"path": str(Path(args.mesh).resolve()), "sha256": sha256_file(args.mesh)},
            "profile": str(Path(args.profile).resolve()),
            "catalog": profile["catalog_path"],
        },
        "frame": "levelled_rgbd",
        "robot_state": {
            "qpos": robot_qpos,
            "source": robot_state_source,
            "ordering": "left_joint1_to_6_then_right_joint1_to_6",
        },
        "robot_placement": robot_placement,
        "objects": records,
        "supports": supports,
        "intersections": intersections,
        "readiness": readiness,
        "mujoco_compile": compile_result,
        "artifacts": {
            "sam_overlay": str((output / "sam_overlay.png").resolve()),
            "mobile_view": str((output / "index.html").resolve()),
            "mujoco": str((output / "scene.xml").resolve()),
            "mujoco_mobile_view": scene_mobile,
            "articulation_video": articulation.get("path"),
            "esdf": str((output / "scene_esdf.npz").resolve()),
            "esdf_mobile_view": str((output / "esdf.html").resolve()),
        },
        "articulation_check": articulation,
        "esdf": esdf_report,
    }
    serializable = scene_json_ready(scene)
    (output / "scene.json").write_text(json.dumps(serializable, indent=2) + "\n")

    if args.daily_scene:
        calibration_id = str(profile.get("calibration_id", "levelled_rgbd_only"))
        daily_objects = [
            SceneObject(
                instance_id=item["instance_id"],
                semantic_name=item["semantic_name"],
                geometry=item["geometry"],
                role=item.get("support_id"),
                confidence=item["confidence"],
                status=item["status"],
                source=item["source"],
                transparent=item["transparent"],
                mask_path=item["mask_path"],
                depth_quality=(
                    "accepted" if item["status"] == "confirmed" else "uncertain"
                ),
            )
            for item in records
        ]
        proposed = DailySceneStore(args.daily_scene).propose(
            objects=daily_objects,
            calibration_id=calibration_id,
            camera_ids={"rgbd": str(profile.get("camera_id", "unknown"))},
            images={
                "SAM overlay": str((output / "sam_overlay.png").resolve()),
            },
            reason="sam_first_semantic_reconstruction",
        )
        if all_automatic:
            proposed = DailySceneStore(args.daily_scene).confirm(
                revision=proposed.revision,
                operator="semantic-scene-auto-gate",
            )
        serializable["daily_scene"] = {
            "path": str(Path(args.daily_scene).resolve()),
            "revision": proposed.revision,
            "calibration_id": proposed.calibration_id,
            "status": proposed.status,
        }
        (output / "scene.json").write_text(
            json.dumps(serializable, indent=2) + "\n"
        )
    return serializable


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture")
    parser.add_argument("--rgb")
    parser.add_argument("--mesh")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument(
        "--mask",
        action="append",
        default=[],
        help="accepted LABEL=PATH mask; supplying any masks disables live SAM",
    )
    parser.add_argument("--daily-scene")
    parser.add_argument("--calibration-report")
    parser.add_argument(
        "--resume-confirmed",
        action="store_true",
        help="resume an existing output after its daily-scene revision is confirmed",
    )
    args = parser.parse_args(argv)
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
