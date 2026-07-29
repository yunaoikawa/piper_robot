#!/usr/bin/env python3
"""Recover distinct lab support heights from a saved head RGB-D capture.

This is a display/calibration diagnostic, not an accepted camera-to-robot
extrinsic.  It deliberately keeps the black work table, aluminium platform,
microscope stage, and labware supports as separate planes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull


DEFAULT_REGIONS = {
    "left_table": [0.00, 0.53, 0.57, 0.96],
    "right_platform": [0.47, 0.99, 0.66, 0.98],
    "white_wall": [0.69, 0.99, 0.00, 0.61],
}

DEFAULT_ANCHORS = {
    "left_piper_base": {"pixel_uv": [0.18, 0.48], "support": "left_table"},
    "right_piper_base": {"pixel_uv": [0.30, 0.80], "support": "right_platform"},
    "petri_dish": {"pixel_uv": [0.70, 0.83], "support": "right_platform"},
    "microscope": {"pixel_uv": [0.44, 0.51], "support": "left_table"},
    "incubator": {"pixel_uv": [0.89, 0.78], "support": "right_platform"},
}

DEFAULT_SEMANTIC_OBJECTS = {}


def _fit_plane(points, *, threshold_m=0.006, iterations=2500, seed=7):
    points = np.asarray(points, dtype=float)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        offset = -float(normal @ sample[0])
        residual = np.abs(points @ normal + offset)
        count = int(np.count_nonzero(residual < threshold_m))
        if best is None or count > best[0]:
            best = (count, residual)
    if best is None:
        raise ValueError("plane RANSAC failed")
    inliers = points[best[1] < threshold_m]
    center = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    offset = -float(normal @ center)
    residual = inliers @ normal + offset
    return {
        "normal": normal,
        "offset": offset,
        "center": center,
        "inliers": inliers,
        "rms_m": float(np.sqrt(np.mean(residual**2))),
        "inlier_fraction": float(len(inliers) / len(points)),
    }


def _region_points(points, valid, region):
    height, width = valid.shape
    x0, x1, y0, y1 = region
    x0, x1 = int(x0 * width), int(x1 * width)
    y0, y1 = int(y0 * height), int(y1 * height)
    mask = valid[y0:y1, x0:x1]
    return points[y0:y1, x0:x1][mask]


def _ray_plane(pixel_xy, matrix, plane):
    ray = np.linalg.inv(matrix) @ np.array([*pixel_xy, 1.0])
    distance = -plane["offset"] / float(plane["normal"] @ ray)
    return ray * distance


def _box_trace(center, size, color, name):
    half = np.asarray(size, dtype=float) / 2
    vertices = np.asarray(
        [
            [sx * half[0], sy * half[1], sz * half[2]]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    ) + np.asarray(center)
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ]
    )
    trace = go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=0.45, name=name, showlegend=True,
        hovertemplate=name + "<extra></extra>",
    )
    trace.visible = "legendonly"
    return trace


def _cylinder_trace(center, size, color, name, *, visible=True):
    radius = float(size[0]) / 2
    half_height = float(size[2]) / 2
    angles = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    vertices = np.vstack(
        [
            np.c_[ring, np.full(len(ring), -half_height)],
            np.c_[ring, np.full(len(ring), half_height)],
            [0, 0, -half_height],
            [0, 0, half_height],
        ]
    ) + np.asarray(center)
    faces = []
    count = len(ring)
    for index in range(count):
        following = (index + 1) % count
        faces.extend(
            [
                [index, following, count + following],
                [index, count + following, count + index],
                [2 * count, following, index],
                [2 * count + 1, count + index, count + following],
            ]
        )
    faces = np.asarray(faces)
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=0.42, name=name, visible=visible,
        hovertemplate=name + " (hidden-body proxy)<extra></extra>",
    )


def _discover_horizontal_surfaces(
    points, level_points, valid, up_camera, *, bin_m=0.004, tolerance_m=0.008
):
    """Find connected, nearly-horizontal surfaces without semantic labels."""
    horizontal = np.zeros(valid.shape, dtype=bool)
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals = np.cross(dx, dy)
    lengths = np.linalg.norm(normals, axis=-1)
    usable = lengths > 1e-7
    normals[usable] /= lengths[usable, None]
    alignment = np.zeros_like(lengths)
    alignment[usable] = np.abs(normals[usable] @ up_camera)
    horizontal[1:-1, 1:-1] = alignment > np.cos(np.deg2rad(13.0))
    horizontal &= valid

    heights = level_points[..., 2][horizontal]
    low, high = np.quantile(heights, [0.01, 0.99])
    edges = np.arange(np.floor(low / bin_m) * bin_m, high + 2 * bin_m, bin_m)
    counts, _ = np.histogram(heights, bins=edges)
    smoothed = gaussian_filter1d(counts.astype(float), 1.25)
    minimum_prominence = max(12.0, float(np.max(smoothed)) * 0.025)
    padded = np.r_[0.0, smoothed, 0.0]
    peaks, properties = find_peaks(
        padded,
        distance=max(1, int(0.018 / bin_m)),
        prominence=minimum_prominence,
    )
    peaks = peaks - 1
    centers = (edges[:-1] + edges[1:]) / 2
    order = sorted(
        range(len(peaks)),
        key=lambda index: properties["prominences"][index],
        reverse=True,
    )
    surfaces = []
    palette = [
        (239, 68, 68), (34, 197, 94), (59, 130, 246), (234, 179, 8),
        (168, 85, 247), (20, 184, 166), (249, 115, 22), (236, 72, 153),
    ]
    kernel = np.ones((3, 3), np.uint8)
    for rank, peak_index in enumerate(order[:12]):
        height = float(centers[peaks[peak_index]])
        mask = horizontal & (
            np.abs(level_points[..., 2] - height) <= tolerance_m
        )
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for component in range(1, component_count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < 22:
                continue
            component_mask = labels == component
            component_heights = level_points[..., 2][component_mask]
            surfaces.append(
                {
                    "height_m": float(np.median(component_heights)),
                    "height_mad_m": float(
                        np.median(
                            np.abs(component_heights - np.median(component_heights))
                        )
                    ),
                    "area_pixels_lowres": area,
                    "bbox_lowres": [
                        int(stats[component, cv2.CC_STAT_LEFT]),
                        int(stats[component, cv2.CC_STAT_TOP]),
                        int(stats[component, cv2.CC_STAT_WIDTH]),
                        int(stats[component, cv2.CC_STAT_HEIGHT]),
                    ],
                    "mask": component_mask,
                    "color_rgb": palette[rank % len(palette)],
                    "peak_prominence": float(properties["prominences"][peak_index]),
                }
            )
    surfaces.sort(key=lambda item: item["area_pixels_lowres"], reverse=True)
    return horizontal, surfaces[:10]


def _draw_validation_overlay(
    rgb, regions, anchors, surfaces, depth_shape, output_path,
    semantic_observations=None,
):
    overlay = rgb.copy()
    height, width = overlay.shape[:2]
    low_height, low_width = depth_shape
    layer = overlay.copy()
    for surface in surfaces:
        mask = cv2.resize(
            surface["mask"].astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        color_bgr = tuple(reversed(surface["color_rgb"]))
        layer[mask] = color_bgr
    cv2.addWeighted(layer, 0.38, overlay, 0.62, 0, overlay)
    for name, region in regions.items():
        x0, x1, y0, y1 = region
        p0 = (int(x0 * width), int(y0 * height))
        p1 = (int(x1 * width), int(y1 * height))
        cv2.rectangle(overlay, p0, p1, (0, 255, 255), 3)
        cv2.putText(
            overlay, name, (p0[0] + 5, p0[1] + 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
        )
    for name, definition in anchors.items():
        if not definition.get("confirmed", False):
            continue
        u, v = np.asarray(definition["pixel_uv"]) * [width, height]
        center = (int(u), int(v))
        cv2.drawMarker(
            overlay, center, (255, 255, 255), cv2.MARKER_CROSS, 24, 3
        )
        cv2.putText(
            overlay, name, (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
    for name, observation in (semantic_observations or {}).items():
        x0, x1, y0, y1 = observation["roi_normalized"]
        p0 = (int(x0 * width), int(y0 * height))
        p1 = (int(x1 * width), int(y1 * height))
        cv2.rectangle(overlay, p0, p1, (255, 0, 255), 3)
        cv2.putText(
            overlay, name, (p0[0] + 5, p1[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA,
        )
    for index, surface in enumerate(surfaces):
        x, y, w, h = surface["bbox_lowres"]
        center = (
            int((x + w / 2) * width / low_width),
            int((y + h / 2) * height / low_height),
        )
        color_bgr = tuple(reversed(surface["color_rgb"]))
        cv2.putText(
            overlay, f"S{index + 1} {surface['height_m']:+.3f}m", center,
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, color_bgr, 2, cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), overlay)


def _write_support_prism(path, points_level, *, thickness_m=0.025):
    """Write a watertight convex support prism suitable for MuJoCo collision."""
    xy = np.asarray(points_level, dtype=float)[:, :2]
    hull = ConvexHull(xy)
    polygon = xy[hull.vertices]
    top = float(np.median(np.asarray(points_level)[:, 2]))
    bottom = top - thickness_m
    vertices = np.vstack(
        [
            np.c_[polygon, np.full(len(polygon), bottom)],
            np.c_[polygon, np.full(len(polygon), top)],
        ]
    )
    count = len(polygon)
    faces = []
    for index in range(1, count - 1):
        faces.append((count, count + index, count + index + 1))
        faces.append((0, index + 1, index))
    for index in range(count):
        following = (index + 1) % count
        faces.append((index, following, count + following))
        faces.append((index, count + following, count + index))
    lines = [f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices]
    lines.extend(
        f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces
    )
    Path(path).write_text("\n".join(lines) + "\n")
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "height_m": top,
        "thickness_m": thickness_m,
        "footprint_area_m2": float(hull.volume),
    }


def _triangulate_depth_grid(points, valid, *, maximum_edge_m=0.035):
    """Triangulate locally continuous RGB-D pixels without bridging gaps."""
    height, width = valid.shape
    grid = np.arange(height * width, dtype=np.int32).reshape(height, width)
    candidates = np.concatenate(
        [
            np.stack(
                [grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1]], axis=-1
            ).reshape(-1, 3),
            np.stack(
                [grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=-1
            ).reshape(-1, 3),
        ],
        axis=0,
    )
    flat_points = np.asarray(points).reshape(-1, 3)
    flat_valid = np.asarray(valid).reshape(-1)
    keep = np.all(flat_valid[candidates], axis=1)
    triangle_points = flat_points[candidates]
    edge_lengths = np.stack(
        [
            np.linalg.norm(triangle_points[:, 0] - triangle_points[:, 1], axis=1),
            np.linalg.norm(triangle_points[:, 1] - triangle_points[:, 2], axis=1),
            np.linalg.norm(triangle_points[:, 2] - triangle_points[:, 0], axis=1),
        ],
        axis=1,
    )
    keep &= np.max(edge_lengths, axis=1) <= maximum_edge_m
    return candidates[keep]


def _write_measured_support_mjcf(
    output_dir, level_points, surfaces, semantic_observations=None
):
    """Export only large measured support planes; minor blue patches are omitted."""
    accepted = [
        (source_index, surface)
        for source_index, surface in enumerate(surfaces, 1)
        if surface["area_pixels_lowres"] >= 500
    ]
    assets = []
    geoms = []
    exported = []
    for index, (source_index, surface) in enumerate(accepted, 1):
        name = f"measured_support_{index}"
        filename = f"{name}.obj"
        metadata = _write_support_prism(
            output_dir / filename, level_points[surface["mask"]]
        )
        red, green, blue = np.asarray(surface["color_rgb"], dtype=float) / 255.0
        assets.append(f'    <mesh name="{name}" file="{filename}"/>')
        geoms.append(
            f'    <geom name="{name}" type="mesh" mesh="{name}" '
            f'rgba="{red:.4f} {green:.4f} {blue:.4f} 0.82" '
            'contype="1" conaffinity="1" friction="1 0.01 0.01"/>'
        )
        exported.append(
            {
                "name": name,
                "source_surface": f"S{source_index}",
                "obj": filename,
                **metadata,
            }
        )
    for name, observation in (semantic_observations or {}).items():
        center = observation.get("proxy_center_level_m")
        size = observation.get("proxy_size_m")
        if center is None or size is None:
            continue
        geoms.append(
            f'    <geom name="{name}_occlusion_proxy" type="cylinder" '
            f'pos="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}" '
            f'size="{0.5 * size[0]:.6f} {0.5 * size[2]:.6f}" '
            'rgba="0.72 0.80 0.90 0.45" contype="1" conaffinity="1"/>'
        )
    xml_path = output_dir / "measured_supports.mjcf"
    xml_path.write_text(
        """<mujoco model="measured_multilevel_lab_supports">
  <compiler angle="radian" meshdir="."/>
  <option gravity="0 0 -9.81"/>
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7"/></visual>
  <asset>
"""
        + "\n".join(assets)
        + """
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
"""
        + "\n".join(geoms)
        + """
    <camera name="measured_overview" pos="0 -1.8 1.25"
            xyaxes="1 0 0 0 0.55 0.84" fovy="55"/>
  </worldbody>
</mujoco>
"""
    )
    return xml_path, exported


def calibrate(
    capture_dir,
    output_dir,
    *,
    regions=None,
    anchors=None,
    semantic_objects=None,
):
    capture_dir = Path(capture_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((capture_dir / "manifest.json").read_text())
    sequence = int(manifest["derived_preview"]["selected_sequence"])
    frame_dir = capture_dir / "raw" / manifest["camera_label"] / f"{sequence:06d}"
    meta = json.loads((frame_dir / "meta.json").read_text())
    rgb = cv2.rotate(cv2.imread(str(frame_dir / "rgb.png")), cv2.ROTATE_90_CLOCKWISE)
    depth = cv2.rotate(np.load(frame_dir / "depth.npy"), cv2.ROTATE_90_CLOCKWISE).astype(float)
    confidence = cv2.rotate(np.load(frame_dir / "confidence.npy"), cv2.ROTATE_90_CLOCKWISE)
    scale_x = depth.shape[1] / rgb.shape[1]
    scale_y = depth.shape[0] / rgb.shape[0]
    matrix = np.asarray(meta["intrinsics"]["K_rgb_rotated_clockwise"], dtype=float)
    matrix[0] *= scale_x
    matrix[1] *= scale_y
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    points = np.stack(
        [
            (xx - matrix[0, 2]) * depth / matrix[0, 0],
            (yy - matrix[1, 2]) * depth / matrix[1, 1],
            depth,
        ],
        axis=-1,
    )
    valid = (
        np.isfinite(depth)
        & (depth > 0.30)
        & (depth < 1.80)
        & (confidence >= 1)
    )
    regions = DEFAULT_REGIONS if regions is None else regions
    fits = {
        name: _fit_plane(_region_points(points, valid, region), seed=7 + index)
        for index, (name, region) in enumerate(regions.items())
    }
    up = fits["left_table"]["normal"]
    if up[1] < 0:
        up = -up
    right_normal = fits["right_platform"]["normal"]
    if right_normal @ up < 0:
        right_normal = -right_normal
    up = up + right_normal
    up /= np.linalg.norm(up)
    wall_normal = fits["white_wall"]["normal"]
    wall_horizontal = wall_normal - up * float(wall_normal @ up)
    x_axis = wall_horizontal / np.linalg.norm(wall_horizontal)
    y_axis = np.cross(up, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    anchors = DEFAULT_ANCHORS if anchors is None else anchors
    right_uv = np.asarray(anchors["right_piper_base"]["pixel_uv"]) * np.array(
        [depth.shape[1], depth.shape[0]]
    )
    origin_camera = _ray_plane(right_uv, matrix, fits["right_platform"])
    rotation = np.stack([x_axis, y_axis, up])
    level_points = (rotation @ (points.reshape(-1, 3) - origin_camera).T).T
    level_points = level_points.reshape(points.shape)
    horizontal_mask, discovered_surfaces = _discover_horizontal_surfaces(
        points, level_points, valid, up
    )

    level_heights = {}
    for name in ("left_table", "right_platform"):
        level_heights[name] = float(
            np.median(
                (rotation @ (fits[name]["inliers"] - origin_camera).T).T[:, 2]
            )
        )

    anchor_xyz = {}
    for name, definition in anchors.items():
        uv = np.asarray(definition["pixel_uv"]) * np.array(
            [depth.shape[1], depth.shape[0]]
        )
        point_camera = _ray_plane(uv, matrix, fits[definition["support"]])
        point_level = rotation @ (point_camera - origin_camera)
        point_level[2] = level_heights[definition["support"]]
        anchor_xyz[name] = point_level

    semantic_objects = (
        DEFAULT_SEMANTIC_OBJECTS
        if semantic_objects is None
        else semantic_objects
    )
    semantic_observations = {}
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    blue_full = cv2.inRange(hsv, (75, 70, 70), (115, 255, 255))
    blue_low = cv2.resize(
        blue_full,
        (depth.shape[1], depth.shape[0]),
        interpolation=cv2.INTER_AREA,
    ) > 100
    for name, definition in semantic_objects.items():
        x0, x1, y0, y1 = definition["blue_label_roi"]
        roi = np.zeros(depth.shape, dtype=bool)
        roi[
            int(y0 * depth.shape[0]) : int(y1 * depth.shape[0]),
            int(x0 * depth.shape[1]) : int(x1 * depth.shape[1]),
        ] = True
        support = blue_low & roi & valid
        observation = {
            "type": definition["type"],
            "roi_normalized": definition["blue_label_roi"],
            "occlusion": definition.get("occlusion"),
            "blue_label_pixels_lowres": int(np.count_nonzero(support)),
            "pose_authority": "blue_label_depth_only_hidden_body_proxy",
        }
        if np.count_nonzero(support) >= 8:
            label_points = level_points[support]
            label_center = np.median(label_points, axis=0)
            proxy_size = np.asarray(definition["proxy_size_m"], dtype=float)
            # The label constrains XY well but not the hidden bottle bottom.
            # Keep its observed label height as the proxy centre and expose
            # the full-height uncertainty in the report.
            proxy_center = label_center.copy()
            observation.update(
                {
                    "blue_label_center_level_m": label_center.tolist(),
                    "blue_label_mad_m": np.median(
                        np.abs(label_points - label_center), axis=0
                    ).tolist(),
                    "proxy_center_level_m": proxy_center.tolist(),
                    "proxy_size_m": proxy_size.tolist(),
                    "vertical_uncertainty_m": float(proxy_size[2] / 2),
                }
            )
        semantic_observations[name] = observation

    overlay_path = output_dir / "surface_validation.png"
    _draw_validation_overlay(
        rgb,
        regions,
        anchors,
        discovered_surfaces,
        depth.shape,
        overlay_path,
        semantic_observations,
    )
    measured_mjcf_path, exported_collision_surfaces = (
        _write_measured_support_mjcf(
            output_dir,
            level_points,
            discovered_surfaces,
            semantic_observations,
        )
    )

    small_rgb = cv2.resize(
        rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA
    )
    scene_faces = _triangulate_depth_grid(level_points, valid)
    scene_vertices = level_points.reshape(-1, 3)
    scene_colors = small_rgb.reshape(-1, 3)[:, ::-1]
    np.savez_compressed(
        output_dir / "measured_scene_mesh_levelled.npz",
        vertices_xyz_m=scene_vertices.astype(np.float32),
        faces=scene_faces.astype(np.int32),
        vertex_rgb=scene_colors.astype(np.uint8),
        valid_vertex_mask=valid.reshape(-1),
    )
    color_strings = [
        f"rgb({red},{green},{blue})"
        for red, green, blue in scene_colors.tolist()
    ]
    traces = [
        go.Mesh3d(
            x=scene_vertices[:, 0],
            y=scene_vertices[:, 1],
            z=scene_vertices[:, 2],
            i=scene_faces[:, 0],
            j=scene_faces[:, 1],
            k=scene_faces[:, 2],
            vertexcolor=color_strings,
            opacity=0.90,
            flatshading=False,
            name="measured RGB-D polygon mesh",
            hoverinfo="skip",
        )
    ]
    for name, height in level_heights.items():
        traces.append(
            go.Scatter3d(
                x=[anchor_xyz["right_piper_base"][0]],
                y=[anchor_xyz["right_piper_base"][1]],
                z=[height],
                mode="markers+text",
                marker=dict(size=7),
                text=[f"{name}: {height:+.3f} m"],
                textposition="top center",
                name=name,
            )
        )
    for index, surface in enumerate(discovered_surfaces):
        surface_cloud = level_points[surface["mask"]]
        red, green, blue = surface["color_rgb"]
        traces.append(
            go.Scatter3d(
                x=surface_cloud[:, 0],
                y=surface_cloud[:, 1],
                z=surface_cloud[:, 2],
                mode="markers",
                marker=dict(
                    size=3.5,
                    color=f"rgb({red},{green},{blue})",
                    opacity=0.9,
                ),
                name=f"S{index + 1} support {surface['height_m']:+.3f} m",
                hovertemplate=(
                    f"S{index + 1}<br>height={surface['height_m']:+.4f} m"
                    "<extra></extra>"
                ),
            )
        )
    left = anchor_xyz["left_piper_base"]
    right = anchor_xyz["right_piper_base"]
    dish = anchor_xyz["petri_dish"]
    microscope = anchor_xyz["microscope"]
    incubator = anchor_xyz["incubator"]
    traces.extend(
        [
            _box_trace(left + [0, 0, 0.08], [0.14, 0.14, 0.16], "#64748b", "left Piper base proxy"),
            _box_trace(right + [0, 0, 0.08], [0.14, 0.14, 0.16], "#475569", "right Piper base proxy"),
            _box_trace(microscope + [0, 0, 0.15], [0.24, 0.32, 0.30], "#e5e7eb", "microscope proxy"),
            _box_trace(incubator + [0, 0, 0.18], [0.28, 0.34, 0.36], "#f8fafc", "incubator proxy"),
            _box_trace(dish + [0, 0, 0.008], [0.09, 0.09, 0.016], "#60a5fa", "petri dish proxy"),
        ]
    )
    for name, observation in semantic_observations.items():
        if "proxy_center_level_m" not in observation:
            continue
        traces.append(
            _cylinder_trace(
                observation["proxy_center_level_m"],
                observation["proxy_size_m"],
                "#93c5fd",
                f"{name} (occluded proxy)",
            )
        )
    figure = go.Figure(traces)
    figure.update_layout(
        title="Measured multi-level lab calibration",
        paper_bgcolor="#f8fafc",
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            aspectmode="data",
            bgcolor="#eef2f7",
            xaxis_title="level X [m]",
            yaxis_title="level Y [m]",
            zaxis_title="height relative to right platform [m]",
            camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0)),
        ),
    )
    plot_html = figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "scrollZoom": True, "displaylogo": False},
    )
    surface_rows = "\n".join(
        (
            f"<tr><td>S{index + 1}</td>"
            f"<td>{surface['height_m']:+.3f} m</td>"
            f"<td>{surface['height_mad_m'] * 1000:.1f} mm</td>"
            f"<td>{surface['area_pixels_lowres']}</td></tr>"
        )
        for index, surface in enumerate(discovered_surfaces)
    )
    html_path = output_dir / "index.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>実測・多段ラボ形状</title>
<style>
body{{margin:0;background:#eef2f7;color:#172033;font-family:-apple-system,sans-serif}}
.card{{margin:10px;background:white;border-radius:12px;padding:12px;box-shadow:0 2px 12px #0002}}
img{{width:100%;height:auto;border-radius:8px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:6px;border-bottom:1px solid #ddd;text-align:left}}
.warn{{color:#9a3412}}.plot{{height:72vh}}
</style></head><body>
<div class="card"><h2>画像・深度による多段面校正</h2>
<p>色付き部分が深度から自動抽出した水平な連結面、黄枠が姿勢推定用ROI、
白十字が未承認の物体アンカーです。</p>
<img src="surface_validation.png" alt="surface validation">
<table><thead><tr><th>面</th><th>右台からの高さ</th><th>MAD</th><th>低解像度面積</th></tr></thead>
<tbody>{surface_rows}</tbody></table>
<p><a href="{measured_mjcf_path.name}">実測支持面MuJoCo (MJCF)</a> —
面積500px未満の青い小領域は衝突形状から除外しました。</p>
<p><a href="mujoco_supports.html">実測MuJoCo形状だけを操作して見る</a></p>
<p>紫枠はユーザー確認済みの培養液ボトル青シール領域です。
ボトル本体は遮蔽中のため、3Dでは半透明円柱と高さ不確実性で表します。</p>
<p class="warn">箱形プロキシは位置未承認なので初期非表示です。凡例タップでのみ表示します。</p>
</div>
<div class="card plot">{plot_html}</div>
</body></html>
""",
        encoding="utf-8",
    )
    report = {
        "schema": "piper_robot.multilevel_lab_calibration/v1",
        "capture": str(capture_dir),
        "sequence": sequence,
        "status": "DISPLAY_ONLY_UNTIL_ROBOT_EXTRINSIC_ACCEPTED",
        "height_reference": "right_platform",
        "level_heights_m": level_heights,
        "height_differences_m": {
            "left_table_minus_right_platform": (
                level_heights["left_table"] - level_heights["right_platform"]
            ),
        },
        "anchor_xyz_level_m": {
            name: value.tolist() for name, value in anchor_xyz.items()
        },
        "plane_quality": {
            name: {
                "rms_m": fit["rms_m"],
                "inlier_fraction": fit["inlier_fraction"],
            }
            for name, fit in fits.items()
        },
        "discovered_horizontal_surfaces": [
            {
                key: value
                for key, value in surface.items()
                if key not in {"mask", "color_rgb"}
            }
            | {"color_rgb": list(surface["color_rgb"])}
            for surface in discovered_surfaces
        ],
        "horizontal_pixel_fraction": float(
            np.count_nonzero(horizontal_mask) / np.count_nonzero(valid)
        ),
        "measured_support_mjcf": str(measured_mjcf_path),
        "measured_scene_mesh": str(
            output_dir / "measured_scene_mesh_levelled.npz"
        ),
        "measured_scene_triangle_count": int(len(scene_faces)),
        "exported_collision_surfaces": exported_collision_surfaces,
        "semantic_observations": semantic_observations,
        "T_level_camera": np.block(
            [
                [rotation, (-rotation @ origin_camera).reshape(3, 1)],
                [np.array([[0.0, 0.0, 0.0, 1.0]])],
            ]
        ).tolist(),
        "limitations": [
            "object XY anchors are image-guided and require operator confirmation",
            "camera-to-robot transform remains unaccepted",
            "transparent dish depth is replaced by its support-plane intersection",
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    config = {} if args.config is None else json.loads(Path(args.config).read_text())
    report = calibrate(
        args.capture,
        args.output_dir,
        regions=config.get("regions"),
        anchors=config.get("anchors"),
        semantic_objects=config.get("semantic_objects"),
    )
    print(json.dumps(report["height_differences_m"], indent=2))


if __name__ == "__main__":
    main()
