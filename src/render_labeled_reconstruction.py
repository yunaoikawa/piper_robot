#!/usr/bin/env python3
"""Render a measured RGB-D mesh with named SAM and Qwen-3D overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def _rgb(values: np.ndarray) -> list[str]:
    return [f"rgb({r},{g},{b})" for r, g, b in values.astype(np.uint8)]


def _compact(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return vertices[used], remap[faces]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-predictions", type=Path)
    parser.add_argument("--maximum-faces", type=int, default=120_000)
    args = parser.parse_args()

    mesh = np.load(args.mesh)
    vertices = np.asarray(mesh["vertices_xyz_m"], dtype=float)
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    colors = np.asarray(mesh["colors_rgb"], dtype=np.uint8)
    labels = np.asarray(mesh["semantic_labels"], dtype=np.int32)
    report = json.loads(args.report.read_text())
    semantics = report["semantics"]
    label_ids = {int(value): name for name, value in semantics["label_ids"].items()}
    label_colors = {
        int(key): value for key, value in semantics["label_colors_rgb"].items()
    }

    if len(faces) > args.maximum_faces:
        indices = np.linspace(0, len(faces) - 1, args.maximum_faces).astype(int)
        display_faces = faces[indices]
    else:
        display_faces = faces
    traces: list[go.BaseTraceType] = [
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=display_faces[:, 0], j=display_faces[:, 1], k=display_faces[:, 2],
            vertexcolor=_rgb(colors), name="Measured RGB-D reconstruction",
            opacity=0.72, hoverinfo="skip", lighting={"ambient": 0.72},
        )
    ]

    for label_id, name in sorted(label_ids.items()):
        selected = display_faces[np.all(labels[display_faces] == label_id, axis=1)]
        if not len(selected):
            continue
        object_vertices, object_faces = _compact(vertices, selected)
        color = label_colors[label_id]
        css_color = f"rgb({color[0]},{color[1]},{color[2]})"
        traces.append(
            go.Mesh3d(
                x=object_vertices[:, 0], y=object_vertices[:, 1],
                z=object_vertices[:, 2], i=object_faces[:, 0],
                j=object_faces[:, 1], k=object_faces[:, 2],
                color=css_color, opacity=0.88, name=f"SAM: {name}",
                hovertemplate=name + "<extra></extra>", showlegend=True,
            )
        )
        points = vertices[labels == label_id]
        center = np.median(points, axis=0)
        traces.append(
            go.Scatter3d(
                x=[center[0]], y=[center[1]], z=[center[2]], mode="text",
                text=[name], textfont={"size": 15, "color": css_color},
                hoverinfo="skip", showlegend=False,
            )
        )

    if args.qwen_predictions:
        predictions = json.loads(args.qwen_predictions.read_text())
        for record in predictions["predictions"]:
            artifact = Path(record["artifact"])
            if not artifact.exists():
                artifact = args.qwen_predictions.parent / artifact.name
            result = np.load(artifact)
            points = np.asarray(result["points_xyz_m"])[
                np.asarray(result["mask"], dtype=bool)
            ]
            traces.append(
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode="markers", marker={"size": 4, "color": "#ef233c"},
                    name=f"Qwen: {record['query']} ({record['score']:.2f})",
                    visible="legendonly",
                    hovertemplate=record["query"] + "<extra></extra>",
                )
            )

    figure = go.Figure(traces)
    figure.update_layout(
        title="Measured reconstruction + named semantic overlays",
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        paper_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01},
        scene={
            "aspectmode": "data", "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)", "zaxis_title": "Z (m)",
            "bgcolor": "rgb(235,240,247)",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        args.output, include_plotlyjs=True, full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
