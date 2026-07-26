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
from src.reconstruct_head_pointcloud import (
    align_depth,
    camera_matrix_from_profile,
)
from rollout.scene_3d import scaled_camera_matrix


def write_mesh_ply(path: Path, vertices, faces, colors):
    with path.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(vertices)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        stream.write(f"element face {len(faces)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        for point, color in zip(vertices, colors):
            stream.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            stream.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


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
            args=[{"visible": [True, True, True]}],
        ),
        dict(
            label="ポリゴン",
            method="update",
            args=[{"visible": [True, False, False]}],
        ),
        dict(
            label="ESDF",
            method="update",
            args=[{"visible": [False, True, False]}],
        ),
        dict(
            label="未知境界",
            method="update",
            args=[{"visible": [True, False, True]}],
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
    )
    mesh = organized_depth_mesh(
        depth,
        depth_matrix,
        rgb=rgb_bgr,
        stride=args.mesh_stride,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
    )

    support_normal = args.support_normal
    if args.support_normal_file:
        support_normal = np.load(args.support_normal_file).tolist()
    if support_normal is not None and args.support_offset is not None:
        rotation, translation = level_transform(
            support_normal, args.support_offset
        )
    else:
        rotation, translation = np.eye(3), np.zeros(3)

    mesh = type(mesh)(
        transform_points(mesh.vertices_xyz_m, rotation, translation),
        mesh.faces,
        mesh.colors_rgb,
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
    )
    np.savez_compressed(
        output_dir / "scene_esdf.npz",
        tsdf=volume.tsdf,
        observed=volume.observed,
        esdf_m=volume.esdf_m,
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
        "confidence_available": False,
        "pose_available": False,
        "coordinate_frame": (
            "support-plane-levelled" if support_normal is not None else "camera"
        ),
        "voxel_size_m": grid.voxel_size_m,
        "truncation_m": args.truncation,
        "shape_zyx": list(grid.shape_zyx),
        "observed_voxels": int(np.count_nonzero(volume.observed)),
        "free_voxels": int(np.count_nonzero(volume.free)),
        "occupied_band_voxels": int(np.count_nonzero(volume.occupied)),
        "unknown_voxels": int(np.count_nonzero(volume.unknown)),
        "mesh_vertices": int(len(mesh.vertices_xyz_m)),
        "mesh_triangles": int(len(mesh.faces)),
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
