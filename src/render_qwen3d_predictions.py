#!/usr/bin/env python3
"""Render language-grounded Qwen-3D point masks as an interactive viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def _sample_indices(count: int, maximum: int, seed: int = 0) -> np.ndarray:
    if count <= maximum:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(count, maximum, False))


def _rgb_strings(colors: np.ndarray) -> list[str]:
    return [f"rgb({r},{g},{b})" for r, g, b in colors.astype(np.uint8)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-background-points", type=int, default=45000)
    parser.add_argument("--maximum-mask-points", type=int, default=25000)
    args = parser.parse_args()

    report = json.loads(args.predictions.read_text())
    prediction_dir = args.predictions.parent
    traces: list[go.Scatter3d] = []
    labels: list[str] = []
    base_points = None
    base_colors = None
    for index, record in enumerate(report["predictions"]):
        artifact = Path(record["artifact"])
        if not artifact.exists():
            artifact = prediction_dir / artifact.name
        data = np.load(artifact)
        points = np.asarray(data["points_xyz_m"])
        colors = np.asarray(data["colors_rgb"])
        mask = np.asarray(data["mask"], dtype=bool)
        if base_points is None:
            base_points, base_colors = points, colors
        elif len(points) != len(base_points):
            raise ValueError("prediction point orders are inconsistent")
        selected = np.flatnonzero(mask)
        selected = selected[
            _sample_indices(len(selected), args.maximum_mask_points, index + 1)
        ]
        query = str(record["query"])
        labels.append(query)
        traces.append(
            go.Scatter3d(
                x=points[selected, 0],
                y=points[selected, 1],
                z=points[selected, 2],
                mode="markers",
                marker={"size": 3.2, "color": "rgb(255,55,40)", "opacity": 0.96},
                name=f"Qwen-3D: {query}",
                hovertemplate=(
                    f"{query}<br>score={record['score']:.3f}<extra></extra>"
                ),
                visible=index == 0,
            )
        )

    if base_points is None or base_colors is None:
        raise ValueError("no predictions found")
    background = _sample_indices(
        len(base_points), args.maximum_background_points
    )
    traces.insert(
        0,
        go.Scatter3d(
            x=base_points[background, 0],
            y=base_points[background, 1],
            z=base_points[background, 2],
            mode="markers",
            marker={
                "size": 1.5,
                "color": _rgb_strings(base_colors[background]),
                "opacity": 0.38,
            },
            name="posed RGB-D",
            hoverinfo="skip",
            visible=True,
        ),
    )
    buttons = []
    for index, label in enumerate(labels, start=1):
        visibility = [True] + [False] * len(labels)
        visibility[index] = True
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visibility}],
            }
        )
    figure = go.Figure(traces)
    figure.update_layout(
        title=(
            "Qwen-3D language-grounded masks — semantic output only; "
            "not motion authority"
        ),
        margin={"l": 0, "r": 0, "t": 74, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Z (m)",
            "bgcolor": "rgb(235,240,247)",
        },
        paper_bgcolor="white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.01,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        args.output,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
