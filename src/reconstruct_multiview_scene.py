#!/usr/bin/env python3
"""Build a SAM-labelled, tag-free multiview RGB-D scene.

This is an observation-only stage.  It improves relative geometry using one
continuous Record3D session, but deliberately keeps collision and motion
readiness false until robot/CAD alignment is independently accepted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.multiview_scene import (
    MultiviewFrame,
    PoseRefinement,
    automatic_world_grid,
    depth_points_and_normals,
    gravity_level_transform,
    integrate_multiview_projective_depth,
    merge_triangle_meshes,
    normalize_record3d_poses,
    refine_pose_point_to_plane,
    support_height_modes,
    transform_points_normals,
)
from rollout.scene_3d import backproject, scaled_camera_matrix
from rollout.scene_semantics import (
    LABEL_BACKGROUND,
    LABEL_FREE,
    LABEL_UNKNOWN,
)
from rollout.scene_volume import (
    TriangleMesh,
    organized_depth_mesh,
    unknown_frontier,
    voxel_centers_for_mask,
)
from rollout.semantic_scene_pipeline import (
    MaskObservation,
    exclusive_masks,
    load_mask,
    load_profile,
    sha256_file,
)
from src.build_semantic_scene import _run_sam
from src.reconstruct_scene_esdf import write_mesh_obj, write_mesh_ply


SCHEMA = "piper_robot.multiview_semantic_scene/v1"


def _parse_mask_specs(specs: list[str], profile: dict) -> dict[str, list[MaskObservation]]:
    by_view: dict[str, list[MaskObservation]] = {}
    counts: dict[tuple[str, str], int] = {}
    score = float(profile.get("accepted_mask_score", 1.0))
    for spec in specs:
        left, separator, path = spec.partition("=")
        view, colon, label = left.partition(":")
        if not separator or not colon or not view or not label or not path:
            raise ValueError(
                f"expected VIEW:LABEL=/absolute/mask.png, got {spec!r}"
            )
        key = (view, label)
        counts[key] = counts.get(key, 0) + 1
        by_view.setdefault(view, []).append(
            MaskObservation(
                instance_id=f"{label}-{counts[key]}",
                semantic_name=label,
                prompt="operator_accepted_mask",
                mask_path=str(Path(path).resolve()),
                sam_score=score,
                model="accepted_mask",
                inference_ms=0.0,
            )
        )
    return by_view


def _temporal_view(
    capture_dir: Path,
    view: dict,
    *,
    minimum_confidence: int,
) -> dict[str, Any]:
    records = list(view.get("frames", ()))
    if len(records) < 3:
        raise ValueError(f"{view.get('name')}: at least three frames are required")
    middle = records[len(records) // 2]
    rgb_path = capture_dir / middle["files"]["rgb_png"]["path"]
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    depths = []
    confidences = []
    for record in records:
        depth = np.load(capture_dir / record["files"]["depth_npy"]["path"])
        confidence = np.load(
            capture_dir / record["files"]["confidence_npy"]["path"]
        )
        if confidence.shape != depth.shape:
            raise ValueError(f"{view['name']}: confidence/depth shapes differ")
        depth = np.asarray(depth, dtype=float)
        depth[
            (~np.isfinite(depth))
            | (depth <= 0.0)
            | (confidence < minimum_confidence)
        ] = np.nan
        depths.append(depth)
        confidences.append(confidence)
    with np.errstate(all="ignore"):
        temporal_depth = np.nanmedian(np.stack(depths), axis=0)
    temporal_confidence = np.rint(
        np.median(np.stack(confidences), axis=0)
    ).astype(np.uint8)
    temporal_confidence[~np.isfinite(temporal_depth)] = 0
    raw_matrix = np.asarray(
        middle["intrinsics"]["K_raw_rgb"], dtype=float
    )
    depth_matrix = scaled_camera_matrix(
        raw_matrix,
        rgb.shape,
        temporal_depth.shape,
    )
    return {
        "name": str(view["name"]),
        "rgb_bgr": rgb,
        "depth_m": temporal_depth,
        "confidence": temporal_confidence,
        "camera_matrix": depth_matrix,
        "camera_pose": middle["camera_pose"],
        "pose_stability": dict(view["pose_stability"]),
        "source_frames": records,
        "representative_rgb": str(rgb_path.resolve()),
    }


def _semantic_labels(
    item: dict[str, Any],
    *,
    profile: dict,
    catalog: dict,
    label_ids: dict[str, int],
    label_colors: dict[int, tuple[int, int, int]],
    accepted_masks: list[MaskObservation] | None,
    sam_endpoint: str,
    output_dir: Path,
) -> tuple[np.ndarray, list[dict]]:
    rgb = item["rgb_bgr"]
    observations = (
        accepted_masks
        if accepted_masks is not None
        else _run_sam(rgb, profile, catalog, sam_endpoint, output_dir)
    )
    if not observations:
        raise RuntimeError(f"{item['name']}: SAM detected no configured objects")
    unknown_names = sorted(
        {observation.semantic_name for observation in observations}
        - set(label_ids)
    )
    if unknown_names:
        raise ValueError(f"{item['name']}: unknown mask labels {unknown_names}")
    owned = exclusive_masks(
        observations,
        rgb.shape[:2],
        transparent_semantics=(
            name
            for name, definition in catalog.items()
            if definition.transparent
        ),
    )
    labels = np.full(rgb.shape[:2], LABEL_BACKGROUND, dtype=np.uint8)
    overlay = rgb.copy()
    records = []
    for observation, mask in owned:
        label_id = label_ids[observation.semantic_name]
        labels[mask] = label_id
        color_rgb = label_colors[label_id]
        color_bgr = np.asarray(color_rgb[::-1], dtype=np.uint8)
        overlay[mask] = np.uint8(
            0.52 * overlay[mask].astype(np.float32)
            + 0.48 * color_bgr.astype(np.float32)
        )
        rows, columns = np.nonzero(mask)
        if len(rows):
            x0, x1 = int(columns.min()), int(columns.max()) + 1
            y0, y1 = int(rows.min()), int(rows.max()) + 1
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color_bgr.tolist(), 3)
            cv2.putText(
                overlay,
                observation.semantic_name,
                (x0, max(24, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color_bgr.tolist(),
                2,
                cv2.LINE_AA,
            )
        records.append(
            {
                **asdict(observation),
                "mask_sha256": sha256_file(observation.mask_path),
                "pixels": int(np.count_nonzero(mask)),
                "label_id": label_id,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "sam_overlay.png"
    labels_path = output_dir / "semantic_labels_rgb.npy"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"failed to write {overlay_path}")
    np.save(labels_path, labels)
    return labels, records


def _rotation_angle_deg(transform: np.ndarray) -> float:
    cosine = np.clip(
        (np.trace(transform[:3, :3]) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(cosine)))


def _tracked_pose_choice(
    seed: PoseRefinement,
    refined: PoseRefinement,
    *,
    minimum_overlap: float,
    maximum_translation_m: float,
    maximum_rotation_deg: float,
) -> tuple[PoseRefinement, dict]:
    """Keep Record3D tracking authoritative unless bounded ICP clearly helps."""

    seed_transform = seed.reference_from_camera
    refined_transform = refined.reference_from_camera
    delta = np.linalg.inv(seed_transform) @ refined_transform
    translation = float(np.linalg.norm(delta[:3, 3]))
    rotation = _rotation_angle_deg(delta)
    overlap_floor = max(minimum_overlap, seed.overlap_fraction * 0.85)
    residual_improved = (
        np.isfinite(refined.median_residual_m)
        and np.isfinite(seed.median_residual_m)
        and refined.median_residual_m <= seed.median_residual_m * 0.90
    )
    refinement_accepted = bool(
        residual_improved
        and refined.overlap_fraction >= overlap_floor
        and translation <= maximum_translation_m
        and rotation <= maximum_rotation_deg
    )
    chosen = refined if refinement_accepted else seed
    pose_accepted = bool(seed.overlap_fraction >= minimum_overlap)
    result = PoseRefinement(
        reference_from_camera=chosen.reference_from_camera,
        median_residual_m=chosen.median_residual_m,
        p90_residual_m=chosen.p90_residual_m,
        overlap_fraction=chosen.overlap_fraction,
        iterations=chosen.iterations,
        accepted=pose_accepted,
        reasons=(
            ()
            if pose_accepted
            else (
                f"tracked-pose neighbour overlap {seed.overlap_fraction:.3f} "
                f"< {minimum_overlap:.3f}",
            )
        ),
    )
    audit = {
        "pose_authority": (
            "bounded_icp_refinement"
            if refinement_accepted
            else "record3d_continuous_tracking"
        ),
        "seed_overlap_fraction": seed.overlap_fraction,
        "seed_median_residual_m": seed.median_residual_m,
        "seed_p90_residual_m": seed.p90_residual_m,
        "refined_overlap_fraction": refined.overlap_fraction,
        "refined_median_residual_m": refined.median_residual_m,
        "refined_p90_residual_m": refined.p90_residual_m,
        "refinement_delta_translation_m": translation,
        "refinement_delta_rotation_deg": rotation,
        "refinement_accepted": refinement_accepted,
    }
    return result, audit


def _match_support_modes(view_modes: list[list[dict]]) -> list[dict]:
    if not view_modes or not view_modes[0]:
        return []
    matched = []
    for reference in view_modes[0]:
        values = [float(reference["height_m"])]
        views = [0]
        for view_index, modes in enumerate(view_modes[1:], 1):
            if not modes:
                continue
            candidate = min(
                modes,
                key=lambda item: abs(
                    float(item["height_m"]) - float(reference["height_m"])
                ),
            )
            if abs(
                float(candidate["height_m"]) - float(reference["height_m"])
            ) <= 0.030:
                values.append(float(candidate["height_m"]))
                views.append(view_index)
        if len(values) >= 2:
            matched.append(
                {
                    "reference_height_m": float(reference["height_m"]),
                    "heights_m": values,
                    "view_indices": views,
                    "spread_m": float(max(values) - min(values)),
                }
            )
    return matched


def _semantic_vertex_colors(
    labels: np.ndarray,
    label_colors: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    colors = np.full((len(labels), 3), (156, 163, 175), dtype=np.uint8)
    for label, color in label_colors.items():
        colors[labels == label] = color
    return colors


def _make_mobile_viewer(
    path: Path,
    mesh: TriangleMesh,
    free_points: np.ndarray,
    free_distance_m: np.ndarray,
    frontier_points: np.ndarray,
    label_colors: dict[int, tuple[int, int, int]],
) -> None:
    import plotly.graph_objects as go

    maximum_faces = 180_000
    faces = mesh.faces
    if len(faces) > maximum_faces:
        faces = faces[
            np.linspace(0, len(faces) - 1, maximum_faces).astype(int)
        ]
    rgb = [f"rgb({r},{g},{b})" for r, g, b in mesh.colors_rgb.tolist()]
    semantic = _semantic_vertex_colors(
        mesh.semantic_labels, label_colors
    )
    semantic_rgb = [
        f"rgb({r},{g},{b})" for r, g, b in semantic.tolist()
    ]
    common = dict(
        x=mesh.vertices_xyz_m[:, 0],
        y=mesh.vertices_xyz_m[:, 1],
        z=mesh.vertices_xyz_m[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=0.88,
        lighting=dict(ambient=0.68, diffuse=0.78, roughness=0.9),
        hoverinfo="skip",
    )
    traces = [
        go.Mesh3d(
            **common,
            vertexcolor=rgb,
            name="multiview RGB-D polygons",
            visible=False,
        ),
        go.Mesh3d(
            **common,
            vertexcolor=semantic_rgb,
            name="multiview SAM labels",
        ),
        go.Scatter3d(
            x=free_points[:, 0],
            y=free_points[:, 1],
            z=free_points[:, 2],
            mode="markers",
            name="ESDF observed free",
            marker=dict(
                size=2.0,
                color=np.minimum(free_distance_m * 1000.0, 80.0),
                cmin=0,
                cmax=80,
                colorscale=[
                    [0.0, "#ef4444"],
                    [0.25, "#f97316"],
                    [0.50, "#facc15"],
                    [1.0, "#22c55e"],
                ],
                colorbar=dict(title="clearance mm"),
                opacity=0.65,
            ),
            hovertemplate="clearance %{marker.color:.1f} mm<extra></extra>",
        ),
        go.Scatter3d(
            x=frontier_points[:, 0],
            y=frontier_points[:, 1],
            z=frontier_points[:, 2],
            mode="markers",
            name="unknown boundary",
            marker=dict(size=2.0, color="#a855f7", opacity=0.58),
            hoverinfo="skip",
        ),
    ]
    figure = go.Figure(traces)
    figure.update_layout(
        title=(
            "Tag-free multiview RGB-D — zは重力で水平化"
            "<br><sup>SAM=物体ラベル、紫=未観測（自由空間ではない）</sup>"
        ),
        paper_bgcolor="#0b1020",
        plot_bgcolor="#0b1020",
        font=dict(color="#f8fafc"),
        margin=dict(l=0, r=0, t=95, b=0),
        legend=dict(orientation="h", y=1.01),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.12,
                buttons=[
                    dict(
                        label="SAM",
                        method="update",
                        args=[{"visible": [False, True, False, False]}],
                    ),
                    dict(
                        label="RGB",
                        method="update",
                        args=[{"visible": [True, False, False, False]}],
                    ),
                    dict(
                        label="ESDF",
                        method="update",
                        args=[{"visible": [False, False, True, False]}],
                    ),
                    dict(
                        label="全部",
                        method="update",
                        args=[{"visible": [False, True, True, True]}],
                    ),
                ],
            )
        ],
        scene=dict(
            xaxis=dict(title="right [m]", backgroundcolor="#0b1020"),
            yaxis=dict(title="forward [m]", backgroundcolor="#0b1020"),
            zaxis=dict(title="up [m]", backgroundcolor="#0b1020"),
            aspectmode="data",
            # Open from the arm side looking toward the wall.  In this view
            # microscope is on screen-left and incubator on screen-right.
            camera=dict(
                eye=dict(x=0.0, y=-1.80, z=0.72),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
    )
    figure.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )


def _make_index(
    path: Path,
    report: dict,
    view_names: list[str],
) -> None:
    rows = []
    for registration in report["registration"]["views"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(registration['name'])}</td>"
            f"<td>{'OK' if registration['accepted'] else 'NG'}</td>"
            f"<td>{registration['overlap_fraction']:.3f}</td>"
            f"<td>{registration['median_residual_m']*1000:.1f}</td>"
            f"<td>{registration['p90_residual_m']*1000:.1f}</td>"
            "</tr>"
        )
    thumbnails = "".join(
        f'<section><h3>{html.escape(name)}</h3>'
        f'<a href="views/{html.escape(name)}/sam_overlay.png">'
        f'<img src="views/{html.escape(name)}/sam_overlay.png"></a></section>'
        for name in view_names
    )
    readiness = report["readiness"]
    document = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tag-free multiview RGB-D</title>
<style>
body{{font-family:-apple-system,sans-serif;background:#0b1020;color:#e5e7eb;margin:16px}}
.gate{{padding:12px;border-radius:10px;background:#172033;margin:10px 0}}
table{{border-collapse:collapse;width:100%;max-width:760px}}td,th{{border:1px solid #475569;padding:7px}}
.views{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
img{{width:100%;border-radius:8px}}a{{color:#7dd3fc}}
</style></head><body>
<h1>タグなしマルチビューRGB-D</h1>
<div class="gate">display_ready={str(readiness['display_ready']).lower()} /
collision_ready=false / motion_ready=false<br>
理由: {html.escape(', '.join(readiness['reasons']) or 'none')}</div>
<p><a href="scene.html">インタラクティブ3Dを開く</a> /
<a href="multiview_report.json">詳細JSON</a></p>
<h2>位置合わせ</h2>
<table><tr><th>view</th><th>判定</th><th>overlap</th>
<th>median mm</th><th>p90 mm</th></tr>{''.join(rows)}</table>
<h2>SAM確認</h2><div class="views">{thumbnails}</div>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def build(args) -> dict:
    capture_dir = Path(args.capture).resolve()
    manifest_path = capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "piper_robot.rgbd_multiview_capture/v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("capture is not a complete multiview Record3D session")
    if manifest.get("device", {}).get("record3d_device_type") != 1:
        raise ValueError("multiview reconstruction requires head LiDAR depth")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile, catalog = load_profile(args.profile)
    accepted_masks = _parse_mask_specs(args.mask, profile)
    unknown_views = sorted(set(accepted_masks) - set(manifest["view_order"]))
    if unknown_views:
        raise ValueError(f"masks reference unknown views: {unknown_views}")

    object_names = [
        name for name in profile.get("objects", list(catalog)) if name in catalog
    ]
    label_ids = {name: index + 3 for index, name in enumerate(object_names)}
    label_colors = {
        LABEL_UNKNOWN: (168, 85, 247),
        LABEL_FREE: (34, 197, 94),
        LABEL_BACKGROUND: (156, 163, 175),
    }
    for name, label_id in label_ids.items():
        value = catalog[name].color.lstrip("#")
        label_colors[label_id] = tuple(
            int(value[index : index + 2], 16) for index in (0, 2, 4)
        )
    dynamic_names = tuple(args.dynamic_object)
    invalid_dynamic = sorted(set(dynamic_names) - set(label_ids))
    if invalid_dynamic:
        raise ValueError(f"dynamic objects are not in the profile: {invalid_dynamic}")
    dynamic_ids = tuple(label_ids[name] for name in dynamic_names)

    items = [
        _temporal_view(
            capture_dir,
            view,
            minimum_confidence=args.minimum_confidence,
        )
        for view in manifest["views"]
    ]
    for item in items:
        view_output = output / "views" / item["name"]
        labels_rgb, observations = _semantic_labels(
            item,
            profile=profile,
            catalog=catalog,
            label_ids=label_ids,
            label_colors=label_colors,
            accepted_masks=accepted_masks.get(item["name"]),
            sam_endpoint=args.sam_endpoint,
            output_dir=view_output,
        )
        item["observations"] = observations
        item["semantic_labels"] = cv2.resize(
            labels_rgb,
            (item["depth_m"].shape[1], item["depth_m"].shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        detected = {record["semantic_name"] for record in observations}
        item["semantic_gate"] = {
            "detected_objects": sorted(detected),
            "robot_mask_present": "robot" in detected,
            "accepted": "robot" in detected,
        }

    representative_poses = [item["camera_pose"] for item in items]
    seed_poses = normalize_record3d_poses(representative_poses)
    level_from_reference = gravity_level_transform(representative_poses[0])
    seed_poses = [level_from_reference @ pose for pose in seed_poses]
    baseline = []
    first_seed_inverse = np.linalg.inv(seed_poses[0])
    for item, pose in zip(items, seed_poses):
        relative_pose = first_seed_inverse @ pose
        baseline.append(
            {
                "name": item["name"],
                "translation_m": float(
                    np.linalg.norm(relative_pose[:3, 3])
                ),
                "rotation_deg": _rotation_angle_deg(relative_pose),
            }
        )
    baseline_sufficient = any(
        record["translation_m"] >= args.minimum_baseline_m
        or record["rotation_deg"] >= args.minimum_baseline_deg
        for record in baseline[1:]
    )
    continuous_tracked_session = bool(
        manifest.get("capture_mode") == "imported_record3d_exr_jpg_video"
        and manifest.get("pose_frame") == "single_record3d_session_local"
    )

    accepted_items = []
    target_points = []
    target_normals = []
    previous_points = None
    previous_normals = None
    registration_reports = []
    support_modes = []
    for index, (item, seed_pose) in enumerate(zip(items, seed_poses)):
        static = (
            item["confidence"] >= args.minimum_confidence
        ) & ~np.isin(item["semantic_labels"], dynamic_ids)
        points_camera, normals_camera = depth_points_and_normals(
            item["depth_m"],
            item["camera_matrix"],
            valid_mask=static,
            stride=args.registration_stride,
            min_depth_m=args.min_depth,
            max_depth_m=args.max_depth,
        )
        registration_audit = {
            "pose_authority": "rejected_before_registration",
            "refinement_accepted": False,
        }
        if not item["pose_stability"]["accepted"]:
            result = PoseRefinement(
                seed_pose,
                float("inf"),
                float("inf"),
                0.0,
                0,
                False,
                ("camera moved during stopped-view burst",),
            )
        elif not item["semantic_gate"]["accepted"]:
            result = PoseRefinement(
                seed_pose,
                float("inf"),
                float("inf"),
                0.0,
                0,
                False,
                ("SAM robot mask missing; static fusion would contain arm geometry",),
            )
        elif not target_points:
            result = PoseRefinement(seed_pose, 0.0, 0.0, 1.0, 0, True, ())
            registration_audit = {
                "pose_authority": "record3d_continuous_tracking",
                "refinement_accepted": False,
            }
        else:
            if continuous_tracked_session:
                seed_evaluation = refine_pose_point_to_plane(
                    points_camera,
                    seed_pose,
                    previous_points,
                    previous_normals,
                    maximum_correspondence_m=args.maximum_correspondence_m,
                    maximum_iterations=0,
                    acceptance_median_m=args.acceptance_median_m,
                    acceptance_p90_m=args.acceptance_p90_m,
                    acceptance_overlap=args.acceptance_overlap,
                )
                refined = refine_pose_point_to_plane(
                    points_camera,
                    seed_pose,
                    previous_points,
                    previous_normals,
                    maximum_correspondence_m=args.maximum_correspondence_m,
                    acceptance_median_m=args.acceptance_median_m,
                    acceptance_p90_m=args.acceptance_p90_m,
                    acceptance_overlap=args.acceptance_overlap,
                )
                result, registration_audit = _tracked_pose_choice(
                    seed_evaluation,
                    refined,
                    minimum_overlap=float(
                        getattr(args, "tracked_pose_minimum_overlap", 0.15)
                    ),
                    maximum_translation_m=float(
                        getattr(
                            args,
                            "maximum_refinement_translation_m",
                            0.03,
                        )
                    ),
                    maximum_rotation_deg=float(
                        getattr(
                            args,
                            "maximum_refinement_rotation_deg",
                            3.0,
                        )
                    ),
                )
            else:
                result = refine_pose_point_to_plane(
                    points_camera,
                    seed_pose,
                    np.vstack(target_points),
                    np.vstack(target_normals),
                    maximum_correspondence_m=args.maximum_correspondence_m,
                    acceptance_median_m=args.acceptance_median_m,
                    acceptance_p90_m=args.acceptance_p90_m,
                    acceptance_overlap=args.acceptance_overlap,
                )
                registration_audit = {
                    "pose_authority": "static_scene_icp",
                    "refinement_accepted": result.accepted,
                }
        registration_reports.append(
            {
                "name": item["name"],
                **asdict(result),
                "reference_from_camera": result.reference_from_camera.tolist(),
                "reasons": list(result.reasons),
                **registration_audit,
            }
        )
        if not result.accepted:
            continue
        item["reference_from_camera"] = result.reference_from_camera
        transformed_points, transformed_normals = transform_points_normals(
            points_camera,
            normals_camera,
            result.reference_from_camera,
        )
        target_points.append(transformed_points)
        target_normals.append(transformed_normals)
        previous_points = transformed_points
        previous_normals = transformed_normals
        support_modes.append(
            support_height_modes(transformed_points, transformed_normals)
        )
        accepted_items.append(item)

    if not accepted_items:
        raise RuntimeError("all multiview frames failed registration gates")
    support_matches = _match_support_modes(support_modes)
    maximum_support_spread = (
        max(record["spread_m"] for record in support_matches)
        if support_matches
        else None
    )
    support_gate = bool(
        support_matches
        and maximum_support_spread is not None
        and maximum_support_spread <= args.acceptance_support_spread_m
    )

    multiview_frames = []
    measured_meshes = []
    world_points = []
    for item in accepted_items:
        valid_depth = (
            np.isfinite(item["depth_m"])
            & (item["depth_m"] >= args.min_depth)
            & (item["depth_m"] <= args.max_depth)
            & (item["confidence"] >= args.minimum_confidence)
        )
        points = backproject(item["depth_m"], item["camera_matrix"])
        transformed, _ = transform_points_normals(
            points[valid_depth],
            np.tile([0.0, 0.0, 1.0], (np.count_nonzero(valid_depth), 1)),
            item["reference_from_camera"],
        )
        world_points.append(transformed)
        multiview_frames.append(
            MultiviewFrame(
                name=item["name"],
                rgb_bgr=item["rgb_bgr"],
                depth_m=item["depth_m"],
                confidence=item["confidence"],
                camera_matrix=item["camera_matrix"],
                reference_from_camera=item["reference_from_camera"],
                semantic_labels=item["semantic_labels"],
            )
        )
        mesh_depth = item["depth_m"].copy()
        mesh_depth[~valid_depth] = np.nan
        mesh = organized_depth_mesh(
            mesh_depth,
            item["camera_matrix"],
            rgb=item["rgb_bgr"],
            stride=args.mesh_stride,
            min_depth_m=args.min_depth,
            max_depth_m=args.max_depth,
            semantic_labels=item["semantic_labels"],
        )
        vertices, _ = transform_points_normals(
            mesh.vertices_xyz_m,
            np.tile([0.0, 0.0, 1.0], (len(mesh.vertices_xyz_m), 1)),
            item["reference_from_camera"],
        )
        measured_meshes.append(
            TriangleMesh(
                vertices,
                mesh.faces,
                mesh.colors_rgb,
                mesh.semantic_labels,
            )
        )

    grid = automatic_world_grid(
        world_points,
        voxel_size_m=args.voxel_size,
        truncation_m=args.truncation,
        maximum_voxels=args.maximum_voxels,
    )
    volume = integrate_multiview_projective_depth(
        multiview_frames,
        grid,
        truncation_m=args.truncation,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        minimum_confidence=args.minimum_confidence,
        dynamic_label_ids=dynamic_ids,
    )
    merged_mesh = merge_triangle_meshes(measured_meshes)
    write_mesh_ply(
        output / "scene_mesh_multiview.ply",
        merged_mesh.vertices_xyz_m,
        merged_mesh.faces,
        merged_mesh.colors_rgb,
        merged_mesh.semantic_labels,
    )
    dynamic_faces = np.any(
        np.isin(merged_mesh.semantic_labels[merged_mesh.faces], dynamic_ids),
        axis=1,
    )
    write_mesh_obj(
        output / "scene_static_mesh_multiview.obj",
        merged_mesh.vertices_xyz_m,
        merged_mesh.faces[~dynamic_faces],
    )
    np.savez_compressed(
        output / "scene_mesh_multiview.npz",
        vertices_xyz_m=merged_mesh.vertices_xyz_m,
        faces=merged_mesh.faces,
        colors_rgb=merged_mesh.colors_rgb,
        semantic_labels=merged_mesh.semantic_labels,
    )
    np.savez_compressed(
        output / "scene_esdf_multiview.npz",
        tsdf=volume.tsdf,
        observed=volume.observed,
        esdf_m=volume.esdf_m,
        semantic_labels=volume.semantic_labels,
        origin_xyz_m=grid.origin_xyz_m,
        voxel_size_m=grid.voxel_size_m,
        shape_zyx=grid.shape_zyx,
    )
    esdf_mask = volume.free & np.isfinite(volume.esdf_m)
    esdf_mask &= volume.esdf_m <= 0.080
    free_points, free_indices = voxel_centers_for_mask(
        grid,
        esdf_mask,
        maximum_points=args.maximum_esdf_points,
        seed=17,
    )
    free_distance = volume.esdf_m[tuple(free_indices.T)]
    frontier_points, _ = voxel_centers_for_mask(
        grid,
        unknown_frontier(volume),
        maximum_points=args.maximum_frontier_points,
        seed=19,
    )
    _make_mobile_viewer(
        output / "scene.html",
        merged_mesh,
        free_points,
        free_distance,
        frontier_points,
        label_colors,
    )

    readiness_reasons = []
    if len(accepted_items) < 2:
        readiness_reasons.append("fewer_than_two_registered_views")
    if not baseline_sufficient:
        readiness_reasons.append("insufficient_camera_baseline")
    if not support_gate:
        readiness_reasons.append("support_height_multiview_disagreement")
    display_ready = not readiness_reasons
    next_stage = None
    if not display_ready:
        next_stage = (
            "repeat_multiview_capture_with_revised_angles"
            if args.attempt < 2
            else "stage_4_exact_robot_cad_alignment"
        )
    report = {
        "schema": SCHEMA,
        "capture": str(capture_dir),
        "capture_manifest": str(manifest_path),
        "capture_manifest_sha256": sha256_file(manifest_path),
        "profile": str(Path(args.profile).resolve()),
        "attempt": args.attempt,
        "coordinate_frame": {
            "name": "gravity_levelled_first_record3d_camera",
            "z_up": True,
            "right_handed": True,
            "camera_to_robot_extrinsic_used": False,
            "T_level_first_camera": level_from_reference.tolist(),
            "transform_convention": "p_level = T_level_first_camera @ p_camera",
        },
        "robot_state": {
            "capture_mode": manifest.get("capture_mode"),
            "operator_action": manifest.get("operator_action"),
            "commands_sent": False,
            "per_view": {
                view["name"]: view.get("robot_state")
                for view in manifest.get("views", ())
                if view.get("robot_state") is not None
            },
        },
        "readiness": {
            "display_ready": display_ready,
            "collision_ready": False,
            "motion_ready": False,
            "reasons": readiness_reasons
            + [
                "camera_to_robot_extrinsic_intentionally_not_used",
                "robot_cad_alignment_not_yet_accepted",
            ],
            "next_stage": next_stage,
        },
        "registration": {
            "method": (
                "continuous_record3d_pose_with_bounded_adjacent_view_icp"
                if continuous_tracked_session
                else "record3d_relative_pose_plus_static_sam_point_to_plane"
            ),
            "continuous_tracked_session": continuous_tracked_session,
            "baseline": baseline,
            "baseline_sufficient": baseline_sufficient,
            "views": registration_reports,
            "accepted_view_names": [item["name"] for item in accepted_items],
        },
        "support_consistency": {
            "per_view_modes": support_modes,
            "matched_levels": support_matches,
            "maximum_spread_m": maximum_support_spread,
            "threshold_m": args.acceptance_support_spread_m,
            "accepted": support_gate,
        },
        "semantics": {
            "label_ids": label_ids,
            "label_colors_rgb": {
                str(key): list(value) for key, value in label_colors.items()
            },
            "dynamic_objects_excluded_from_static_esdf": list(dynamic_names),
            "views": {
                item["name"]: item["observations"] for item in items
            },
        },
        "fusion": {
            "method": "confidence_and_view_angle_weighted_projective_tsdf",
            "voxel_size_m": grid.voxel_size_m,
            "truncation_m": args.truncation,
            "shape_zyx": list(grid.shape_zyx),
            "observed_voxels": int(np.count_nonzero(volume.observed)),
            "unknown_voxels": int(np.count_nonzero(volume.unknown)),
            "mesh_vertices": int(len(merged_mesh.vertices_xyz_m)),
            "mesh_triangles": int(len(merged_mesh.faces)),
        },
        "artifacts": {
            "viewer": str((output / "scene.html").resolve()),
            "index": str((output / "index.html").resolve()),
            "mesh_ply": str((output / "scene_mesh_multiview.ply").resolve()),
            "static_mesh_obj": str(
                (output / "scene_static_mesh_multiview.obj").resolve()
            ),
            "esdf": str((output / "scene_esdf_multiview.npz").resolve()),
        },
        "limitations": [
            "wrist cameras are RGB-only and are not used for metric fusion",
            "transparent dish and lid depth may be the support surface",
            "measured polygons and inferred object completion remain separate",
            "tag-free relative geometry does not authorize robot motion",
        ],
    }
    report_path = output / "multiview_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _make_index(output / "index.html", report, [item["name"] for item in items])
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument(
        "--mask",
        action="append",
        default=[],
        help="accepted VIEW:LABEL=/absolute/mask.png; masks replace live SAM for that view",
    )
    parser.add_argument(
        "--dynamic-object",
        action="append",
        default=["robot", "culture_media_bottle", "petri_dish", "petri_lid"],
    )
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--truncation", type=float, default=0.020)
    parser.add_argument("--minimum-confidence", type=int, default=1)
    parser.add_argument("--min-depth", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=2.00)
    parser.add_argument("--registration-stride", type=int, default=4)
    parser.add_argument("--mesh-stride", type=int, default=2)
    parser.add_argument("--maximum-correspondence-m", type=float, default=0.060)
    parser.add_argument("--acceptance-median-m", type=float, default=0.010)
    parser.add_argument("--acceptance-p90-m", type=float, default=0.025)
    parser.add_argument("--acceptance-overlap", type=float, default=0.30)
    parser.add_argument("--tracked-pose-minimum-overlap", type=float, default=0.15)
    parser.add_argument(
        "--maximum-refinement-translation-m", type=float, default=0.030
    )
    parser.add_argument(
        "--maximum-refinement-rotation-deg", type=float, default=3.0
    )
    parser.add_argument(
        "--acceptance-support-spread-m", type=float, default=0.008
    )
    parser.add_argument("--minimum-baseline-m", type=float, default=0.040)
    parser.add_argument("--minimum-baseline-deg", type=float, default=5.0)
    parser.add_argument("--maximum-voxels", type=int, default=24_000_000)
    parser.add_argument("--maximum-esdf-points", type=int, default=18_000)
    parser.add_argument("--maximum-frontier-points", type=int, default=9_000)
    args = parser.parse_args(argv)
    if (
        args.voxel_size <= 0.0
        or args.truncation <= args.voxel_size
        or args.registration_stride < 1
        or args.mesh_stride < 1
        or args.minimum_confidence not in (0, 1, 2)
        or not 0.0 <= args.tracked_pose_minimum_overlap <= 1.0
        or args.maximum_refinement_translation_m <= 0.0
        or args.maximum_refinement_rotation_deg <= 0.0
    ):
        parser.error("invalid fusion, stride, or confidence configuration")
    build(args)


if __name__ == "__main__":
    main()
