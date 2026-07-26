#!/usr/bin/env python3
"""Build a conservative ESDF and mobile polygon viewer from saved RGB-D."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.scene_volume import (
    automatic_grid,
    integrate_projective_depth,
    level_transform,
    organized_depth_mesh,
    transform_points,
    unknown_frontier,
    voxel_centers_for_mask,
)
from rollout.scene_semantics import (
    LABEL_BACKGROUND,
    LABEL_COLORS_RGB,
    LABEL_LID,
    LABEL_NAMES,
    LABEL_ROBOT,
    compose_surface_labels,
    estimate_image_homography,
    largest_components,
    recover_blended_sam_mask,
    render_semantic_labels,
    warp_mask,
)
from src.reconstruct_head_pointcloud import (
    align_depth,
    camera_matrix_from_profile,
)
from rollout.scene_3d import (
    estimate_target_on_support_plane,
    scaled_camera_matrix,
)


def write_mesh_ply(path: Path, vertices, faces, colors, labels=None):
    with path.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(vertices)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        if labels is not None:
            stream.write("property uchar semantic_label\n")
        stream.write(f"element face {len(faces)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        for index, (point, color) in enumerate(zip(vertices, colors)):
            line = (
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}"
            )
            if labels is not None:
                line += f" {int(labels[index])}"
            stream.write(line + "\n")
        for face in faces:
            stream.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def write_mesh_obj(path: Path, vertices, faces):
    with path.open("w") as stream:
        stream.write("# visual-only measured surface; not a collision mesh\n")
        for point in vertices:
            stream.write(
                f"v {point[0]:.7f} {point[1]:.7f} {point[2]:.7f}\n"
            )
        for face in faces:
            stream.write(
                f"f {int(face[0]) + 1} {int(face[1]) + 1} "
                f"{int(face[2]) + 1}\n"
            )


def semantic_vertex_colors(labels):
    labels = np.asarray(labels, dtype=np.uint8)
    colors = np.full((len(labels), 3), 156, dtype=np.uint8)
    for label, color in LABEL_COLORS_RGB.items():
        colors[labels == label] = color
    return colors


def make_mobile_html(
    path: Path,
    mesh,
    free_points,
    free_distance_m,
    frontier_points,
    *,
    title: str,
):
    try:
        import plotly.graph_objects as go
    except ImportError as error:
        raise RuntimeError(
            "mobile HTML requires plotly (pip install plotly)"
        ) from error

    vertices = mesh.vertices_xyz_m
    faces = mesh.faces
    colors = [
        f"rgb({r},{g},{b})" for r, g, b in mesh.colors_rgb.tolist()
    ]
    semantic_colors = [
        f"rgb({r},{g},{b})"
        for r, g, b in semantic_vertex_colors(mesh.semantic_labels).tolist()
    ]
    traces = [
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            vertexcolor=colors,
            name="RGB-D polygon surface",
            opacity=0.82,
            lighting=dict(ambient=0.65, diffuse=0.8, roughness=0.9),
            hoverinfo="skip",
            visible=False,
        ),
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            vertexcolor=semantic_colors,
            name="SAM semantic surface",
            opacity=0.88,
            lighting=dict(ambient=0.72, diffuse=0.75, roughness=0.9),
            hoverinfo="skip",
        ),
        go.Scatter3d(
            x=free_points[:, 0],
            y=free_points[:, 1],
            z=free_points[:, 2],
            mode="markers",
            name="ESDF observed free",
            marker=dict(
                size=2.2,
                color=np.minimum(free_distance_m * 1000.0, 80.0),
                cmin=0,
                cmax=80,
                colorscale=[
                    [0.0, "#ef4444"],
                    [0.20, "#f97316"],
                    [0.45, "#facc15"],
                    [1.0, "#22c55e"],
                ],
                colorbar=dict(title="clearance mm", thickness=14),
                opacity=0.68,
            ),
            hovertemplate="clearance %{marker.color:.1f} mm<extra></extra>",
        ),
        go.Scatter3d(
            x=frontier_points[:, 0],
            y=frontier_points[:, 1],
            z=frontier_points[:, 2],
            mode="markers",
            name="unknown boundary (collision)",
            marker=dict(size=2.0, color="#a855f7", opacity=0.65),
            hoverinfo="skip",
        ),
    ]
    buttons = [
        dict(
            label="全部",
            method="update",
            args=[{"visible": [False, True, True, True]}],
        ),
        dict(
            label="SAMラベル",
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
            label="未知境界",
            method="update",
            args=[{"visible": [False, True, False, True]}],
        ),
    ]
    figure = go.Figure(traces)
    figure.update_layout(
        title=title,
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#f9fafb"),
        margin=dict(l=0, r=0, t=90, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0,
                y=1.12,
                buttons=buttons,
                bgcolor="#374151",
                font=dict(color="white"),
            )
        ],
        scene=dict(
            xaxis=dict(title="right [m]", backgroundcolor="#111827"),
            yaxis=dict(title="bench forward [m]", backgroundcolor="#111827"),
            zaxis=dict(title="up [m]", backgroundcolor="#111827"),
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=-1.55, z=0.9)),
        ),
    )
    figure.write_html(
        str(path),
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--truncation", type=float, default=0.03)
    parser.add_argument("--min-depth", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=2.00)
    parser.add_argument("--mesh-stride", type=int, default=2)
    parser.add_argument("--support-normal", type=float, nargs=3)
    parser.add_argument("--support-normal-file")
    parser.add_argument("--support-offset", type=float)
    parser.add_argument(
        "--sam-source-rgb",
        help="RGB used to create the saved lossless SAM diagnostic overlays",
    )
    parser.add_argument("--sam-lid-overlay")
    parser.add_argument("--sam-robot-overlay")
    parser.add_argument(
        "--sam-extra-robot-overlay",
        help="optional 0.30-source-weight cyan SAM overlay (e.g. blue clamps)",
    )
    parser.add_argument("--lid-mask")
    parser.add_argument("--robot-mask")
    parser.add_argument("--max-esdf-points", type=int, default=18000)
    parser.add_argument("--max-frontier-points", type=int, default=9000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_bgr = cv2.imread(args.rgb)
    if rgb_bgr is None:
        raise RuntimeError(f"could not read RGB image: {args.rgb}")
    depth = align_depth(np.load(args.depth), rgb_bgr.shape)
    full_matrix = camera_matrix_from_profile(args.profile)
    depth_matrix = scaled_camera_matrix(full_matrix, rgb_bgr.shape, depth.shape)

    lid_mask = None
    robot_mask = None
    sam_registration = None
    if args.lid_mask:
        loaded = cv2.imread(args.lid_mask, cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            raise RuntimeError(f"could not read {args.lid_mask}")
        lid_mask = loaded > 0
    if args.robot_mask:
        loaded = cv2.imread(args.robot_mask, cv2.IMREAD_GRAYSCALE)
        if loaded is None:
            raise RuntimeError(f"could not read {args.robot_mask}")
        robot_mask = loaded > 0
    overlay_requested = any(
        (
            args.sam_lid_overlay,
            args.sam_robot_overlay,
            args.sam_extra_robot_overlay,
        )
    )
    if overlay_requested:
        if not args.sam_source_rgb:
            raise ValueError("SAM overlays require --sam-source-rgb")
        sam_source = cv2.imread(args.sam_source_rgb)
        if sam_source is None:
            raise RuntimeError(f"could not read {args.sam_source_rgb}")
        homography, sam_registration = estimate_image_homography(
            sam_source, rgb_bgr
        )
        if args.sam_lid_overlay:
            overlay = cv2.imread(args.sam_lid_overlay)
            recovered = recover_blended_sam_mask(
                sam_source,
                overlay,
                source_weight=0.55,
                tint_bgr=(20, 150, 60),
            )
            lid_mask = warp_mask(
                largest_components(recovered, count=1),
                homography,
                rgb_bgr.shape,
            )
        if args.sam_robot_overlay:
            overlay = cv2.imread(args.sam_robot_overlay)
            recovered = recover_blended_sam_mask(
                sam_source,
                overlay,
                source_weight=0.55,
                tint_bgr=(0, 255, 255),
            )
            robot_mask = warp_mask(
                largest_components(recovered, count=1),
                homography,
                rgb_bgr.shape,
            )
        if args.sam_extra_robot_overlay:
            overlay = cv2.imread(args.sam_extra_robot_overlay)
            recovered = recover_blended_sam_mask(
                sam_source,
                overlay,
                source_weight=0.30,
                tint_bgr=(0, 255, 255),
            )
            extra = warp_mask(
                largest_components(recovered, count=4),
                homography,
                rgb_bgr.shape,
            )
            robot_mask = extra if robot_mask is None else robot_mask | extra
    for name, mask in (("lid", lid_mask), ("robot", robot_mask)):
        if mask is not None and mask.shape != rgb_bgr.shape[:2]:
            raise ValueError(f"{name} mask and RGB shapes differ")
    surface_labels_rgb = compose_surface_labels(
        rgb_bgr.shape, robot_mask=robot_mask, lid_mask=lid_mask
    )
    surface_labels_depth = cv2.resize(
        surface_labels_rgb,
        (depth.shape[1], depth.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    support_normal = args.support_normal
    support_offset = args.support_offset
    if args.support_normal_file:
        support_normal = np.load(args.support_normal_file).tolist()
    fitted_support = None
    if support_normal is None and lid_mask is not None:
        lid_depth_mask = cv2.resize(
            lid_mask.astype(np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        yy, xx = np.where(lid_depth_mask)
        target_pixel = np.array([np.median(xx), np.median(yy)])
        target = estimate_target_on_support_plane(
            depth,
            depth_matrix,
            lid_depth_mask,
            target_pixel,
            ring_margin_px=max(18, int(70 * depth.shape[1] / rgb_bgr.shape[1])),
        )
        support_normal = target.plane.normal.tolist()
        support_offset = target.plane.offset
        fitted_support = {
            "normal": target.plane.normal.tolist(),
            "offset": target.plane.offset,
            "rms_m": target.plane.rms_m,
            "inlier_fraction": target.plane.inlier_fraction,
        }

    grid = automatic_grid(
        depth,
        depth_matrix,
        voxel_size_m=args.voxel_size,
        truncation_m=args.truncation,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
    )
    volume = integrate_projective_depth(
        depth,
        depth_matrix,
        grid,
        truncation_m=args.truncation,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        surface_labels=surface_labels_depth,
    )
    mesh = organized_depth_mesh(
        depth,
        depth_matrix,
        rgb=rgb_bgr,
        stride=args.mesh_stride,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        semantic_labels=surface_labels_depth,
    )

    if support_normal is not None and support_offset is not None:
        rotation, translation = level_transform(
            support_normal, support_offset
        )
    else:
        rotation, translation = np.eye(3), np.zeros(3)

    mesh = type(mesh)(
        transform_points(mesh.vertices_xyz_m, rotation, translation),
        mesh.faces,
        mesh.colors_rgb,
        mesh.semantic_labels,
    )
    esdf_mask = volume.free & np.isfinite(volume.esdf_m)
    esdf_mask &= volume.esdf_m <= 0.080
    free_points, free_indices = voxel_centers_for_mask(
        grid, esdf_mask, maximum_points=args.max_esdf_points, seed=5
    )
    free_distances = volume.esdf_m[tuple(free_indices.T)]
    frontier_points, _ = voxel_centers_for_mask(
        grid,
        unknown_frontier(volume),
        maximum_points=args.max_frontier_points,
        seed=8,
    )
    free_points = transform_points(free_points, rotation, translation)
    frontier_points = transform_points(
        frontier_points, rotation, translation
    )

    write_mesh_ply(
        output_dir / "scene_mesh_levelled.ply",
        mesh.vertices_xyz_m,
        mesh.faces,
        mesh.colors_rgb,
        mesh.semantic_labels,
    )
    static_face_mask = ~np.any(
        mesh.semantic_labels[mesh.faces] == LABEL_ROBOT, axis=1
    )
    robot_face_mask = np.any(
        mesh.semantic_labels[mesh.faces] == LABEL_ROBOT, axis=1
    )
    write_mesh_ply(
        output_dir / "scene_static_mesh_levelled.ply",
        mesh.vertices_xyz_m,
        mesh.faces[static_face_mask],
        semantic_vertex_colors(mesh.semantic_labels),
        mesh.semantic_labels,
    )
    write_mesh_obj(
        output_dir / "scene_static_mesh_levelled.obj",
        mesh.vertices_xyz_m,
        mesh.faces[static_face_mask],
    )
    write_mesh_obj(
        output_dir / "scene_robot_observation_levelled.obj",
        mesh.vertices_xyz_m,
        mesh.faces[robot_face_mask],
    )
    np.save(output_dir / "semantic_labels_rgb.npy", surface_labels_rgb)
    np.savez_compressed(
        output_dir / "scene_mesh_levelled.npz",
        vertices_xyz_m=mesh.vertices_xyz_m,
        faces=mesh.faces,
        colors_rgb=mesh.colors_rgb,
        semantic_labels=mesh.semantic_labels,
    )
    cv2.imwrite(
        str(output_dir / "semantic_labels_overlay.png"),
        render_semantic_labels(rgb_bgr, surface_labels_rgb),
    )
    np.savez_compressed(
        output_dir / "scene_esdf.npz",
        tsdf=volume.tsdf,
        observed=volume.observed,
        esdf_m=volume.esdf_m,
        semantic_labels=volume.semantic_labels,
        origin_xyz_m=grid.origin_xyz_m,
        voxel_size_m=grid.voxel_size_m,
        shape_zyx=grid.shape_zyx,
        camera_to_level_rotation=rotation,
        camera_to_level_translation=translation,
    )
    make_mobile_html(
        output_dir / "esdf.html",
        mesh,
        free_points,
        free_distances,
        frontier_points,
        title=(
            "Offline ESDF — 赤:近い / 緑:遠い / 紫:未知"
            "<br><sup>紫の未知領域は自由空間として扱わない</sup>"
        ),
    )
    report = {
        "offline_only": True,
        "rgb_source": str(Path(args.rgb)),
        "depth_source": str(Path(args.depth)),
        "rgb_depth_file_mtime_delta_s": abs(
            Path(args.rgb).stat().st_mtime - Path(args.depth).stat().st_mtime
        ),
        "confidence_available": False,
        "pose_available": False,
        "sam_semantics": bool(
            overlay_requested or args.lid_mask or args.robot_mask
        ),
        "sam_registration": sam_registration,
        "support_plane_fit": fitted_support,
        "coordinate_frame": (
            "support-plane-levelled" if support_normal is not None else "camera"
        ),
        "voxel_size_m": grid.voxel_size_m,
        "truncation_m": args.truncation,
        "shape_zyx": list(grid.shape_zyx),
        "observed_voxels": int(np.count_nonzero(volume.observed)),
        "free_voxels": int(np.count_nonzero(volume.free)),
        "occupied_band_voxels": int(np.count_nonzero(volume.occupied)),
        "static_occupied_band_voxels": int(
            np.count_nonzero(
                volume.occupied
                & (volume.semantic_labels != LABEL_ROBOT)
            )
        ),
        "robot_band_voxels": int(
            np.count_nonzero(volume.semantic_labels == LABEL_ROBOT)
        ),
        "unknown_voxels": int(np.count_nonzero(volume.unknown)),
        "mesh_vertices": int(len(mesh.vertices_xyz_m)),
        "mesh_triangles": int(len(mesh.faces)),
        "static_mesh_triangles": int(np.count_nonzero(static_face_mask)),
        "robot_observation_triangles": int(
            np.count_nonzero(robot_face_mask)
        ),
        "semantic_surface_pixels": {
            LABEL_NAMES[label]: int(np.count_nonzero(surface_labels_rgb == label))
            for label in (LABEL_BACKGROUND, LABEL_ROBOT, LABEL_LID)
        },
        "viewer_esdf_samples": int(len(free_points)),
        "viewer_unknown_frontier_samples": int(len(frontier_points)),
        "limitations": [
            "single saved view; occluded space remains unknown",
            "saved frame has no Record3D confidence map or camera pose",
            "transparent lid may return the support surface rather than lid",
            "camera-to-robot calibration is not available in this artifact",
        ],
    }
    (output_dir / "esdf_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
