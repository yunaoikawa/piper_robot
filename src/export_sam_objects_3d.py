#!/usr/bin/env python3
"""Export only accepted SAM-labelled RGB-D surfaces as interactive 3D.

This deliberately performs no CAD fitting, object completion, or fixture
alignment.  Each output surface is a subset of the synchronized RGB-D mesh.
Later masks have priority when labels overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go


COLORS = {
    "robot": "#12c7c9",
    "microscope": "#e333d1",
    "incubator": "#f2d72f",
    "culture_media_bottle_label": "#ff8c1a",
}


def _compact_mesh(vertices, faces):
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], remap[faces]


def _write_obj(path, vertices, faces):
    lines = [f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices]
    lines.extend(
        f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces
    )
    Path(path).write_text("\n".join(lines) + "\n")


def _load_masks(mask_specs, shape):
    masks = []
    for spec in mask_specs:
        label, separator, path = spec.partition("=")
        if not separator:
            raise ValueError(f"expected LABEL=PATH, got {spec!r}")
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        mask = cv2.resize(
            image,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_AREA,
        ) > 100
        masks.append((label, Path(path), mask))
    return masks


def export(mesh_path, mask_specs, output_dir, *, image_shape=(192, 256)):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(mesh_path)
    vertices = np.asarray(archive["vertices_xyz_m"], dtype=float)
    faces = np.asarray(archive["faces"], dtype=np.int32)
    valid = np.asarray(archive["valid_vertex_mask"], dtype=bool)
    if len(vertices) != image_shape[0] * image_shape[1]:
        raise ValueError(
            f"mesh has {len(vertices)} vertices, expected "
            f"{image_shape[0]}x{image_shape[1]}"
        )

    masks = _load_masks(mask_specs, image_shape)
    # Later, more-specific labels win (e.g. bottle label over robot gripper).
    owned = []
    claimed = np.zeros(image_shape, dtype=bool)
    for label, path, mask in reversed(masks):
        exclusive = mask & ~claimed
        owned.append((label, path, exclusive))
        claimed |= mask
    owned.reverse()

    traces = []
    report_objects = {}
    all_points = []
    for label, source_path, mask in owned:
        vertex_mask = mask.reshape(-1) & valid
        # Keep triangles whose complete surface belongs to this semantic label.
        selected_faces = faces[np.all(vertex_mask[faces], axis=1)]
        if len(selected_faces):
            object_vertices, object_faces = _compact_mesh(
                vertices, selected_faces
            )
            _write_obj(
                output_dir / f"{label}.obj",
                object_vertices,
                object_faces,
            )
            traces.append(
                go.Mesh3d(
                    x=object_vertices[:, 0],
                    y=object_vertices[:, 1],
                    z=object_vertices[:, 2],
                    i=object_faces[:, 0],
                    j=object_faces[:, 1],
                    k=object_faces[:, 2],
                    name=label,
                    color=COLORS.get(label, "#8b9bb4"),
                    opacity=1.0,
                    flatshading=False,
                    hovertemplate=label + "<extra></extra>",
                    showlegend=True,
                )
            )
        else:
            object_vertices = vertices[vertex_mask]
            object_faces = np.empty((0, 3), dtype=np.int32)
        # Preserve sparse labels even when downsampling leaves no full triangle.
        unused = vertex_mask.copy()
        if len(selected_faces):
            unused[np.unique(selected_faces)] = False
        sparse = vertices[unused]
        if len(sparse):
            traces.append(
                go.Scatter3d(
                    x=sparse[:, 0],
                    y=sparse[:, 1],
                    z=sparse[:, 2],
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": COLORS.get(label, "#8b9bb4"),
                    },
                    name=f"{label} (points)",
                    hovertemplate=label + "<extra></extra>",
                    showlegend=not len(selected_faces),
                )
            )
        points = vertices[vertex_mask]
        if not len(points):
            continue
        all_points.append(points)
        center = np.median(points, axis=0)
        traces.append(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="text",
                text=[label],
                textfont={"size": 13, "color": "#111827"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        report_objects[label] = {
            "source_mask": str(source_path.resolve()),
            "mask_pixels": int(mask.sum()),
            "valid_depth_points": int(len(points)),
            "vertices_in_obj": int(len(object_vertices)),
            "triangles_in_obj": int(len(object_faces)),
            "median_xyz_level_m": center.tolist(),
            "lower_xyz_level_m": np.quantile(points, 0.02, axis=0).tolist(),
            "upper_xyz_level_m": np.quantile(points, 0.98, axis=0).tolist(),
        }

    if not all_points:
        raise RuntimeError("accepted masks contain no valid RGB-D points")
    combined = np.concatenate(all_points)
    scene_center = np.median(combined, axis=0)
    figure = go.Figure(traces)
    figure.update_layout(
        title="SAM-labelled RGB-D surfaces only",
        paper_bgcolor="#f8fafc",
        margin={"l": 0, "r": 0, "t": 52, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
        },
        scene={
            "aspectmode": "data",
            "bgcolor": "#eef2f7",
            "xaxis": {"title": "X (m)", "gridcolor": "#cbd5e1"},
            "yaxis": {"title": "Y (m)", "gridcolor": "#cbd5e1"},
            "zaxis": {"title": "Z (m)", "gridcolor": "#cbd5e1"},
            "camera": {"eye": {"x": 1.35, "y": -1.55, "z": 1.05}},
        },
    )
    html_path = output_dir / "index.html"
    figure.write_html(
        html_path,
        include_plotlyjs=True,
        full_html=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
    )
    report = {
        "schema": "piper_robot.sam_objects_3d/v1",
        "method": "accepted_sam_masks_intersect_synchronized_rgbd_mesh",
        "frame": "levelled_rgbd",
        "completion_or_cad_fitting": False,
        "mesh_source": str(Path(mesh_path).resolve()),
        "scene_center_xyz_m": scene_center.tolist(),
        "objects": report_objects,
        "html": str(html_path.resolve()),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument(
        "--mask",
        action="append",
        required=True,
        help="accepted semantic mask as LABEL=PATH; later masks win overlap",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-height", type=int, default=192)
    parser.add_argument("--image-width", type=int, default=256)
    args = parser.parse_args(argv)
    report = export(
        args.mesh,
        args.mask,
        args.output_dir,
        image_shape=(args.image_height, args.image_width),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
