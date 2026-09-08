#!/usr/bin/env python3
"""Create a small, phone-friendly copy of a self-contained Plotly 3D page.

The source page remains the lossless engineering artifact.  This command only
reduces display geometry and shares one Plotly runtime between all mobile
pages; it never changes reconstruction, MuJoCo, ESDF, or collision inputs.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def _new_plot_arguments(page: str) -> tuple[list[dict], dict]:
    marker = "Plotly.newPlot("
    start = page.rfind(marker)
    if start < 0:
        raise ValueError("Plotly.newPlot call not found")
    decoder = json.JSONDecoder()
    first_comma = page.index(",", start + len(marker))
    data_start = page.index("[", first_comma)
    data, data_end = decoder.raw_decode(page[data_start:])
    layout_start = data_start + data_end
    layout_start = page.index("{", layout_start)
    layout, _ = decoder.raw_decode(page[layout_start:])
    return data, layout


def _array(value) -> np.ndarray:
    if isinstance(value, dict) and "bdata" in value:
        array = np.frombuffer(
            base64.b64decode(value["bdata"]),
            dtype=np.dtype(value["dtype"]),
        )
        shape = value.get("shape")
        if shape:
            if isinstance(shape, str):
                shape = tuple(int(part) for part in shape.split(","))
            array = array.reshape(shape)
        return array
    return np.asarray(value)


def _plain(array: np.ndarray, *, decimals: int = 5) -> list:
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.floating):
        array = np.round(array.astype(float), decimals)
    return array.tolist()


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=int)
    return np.linspace(0, count - 1, maximum, dtype=int)


def _simplify_mesh(trace: dict, maximum_faces: int) -> None:
    faces = np.column_stack(
        [_array(trace[name]).astype(int) for name in ("i", "j", "k")]
    )
    selected = faces[_sample_indices(len(faces), maximum_faces)]
    used = np.unique(selected)
    remap = np.full(int(used.max()) + 1, -1, dtype=int)
    remap[used] = np.arange(len(used))
    selected = remap[selected]
    for name in ("x", "y", "z"):
        trace[name] = _plain(_array(trace[name])[used])
    for column, name in enumerate(("i", "j", "k")):
        trace[name] = _plain(selected[:, column])


def _simplify_scatter(trace: dict, maximum_points: int) -> None:
    coordinates = [
        _array(trace[name]) for name in ("x", "y", "z") if name in trace
    ]
    if not coordinates:
        return
    selected = _sample_indices(len(coordinates[0]), maximum_points)
    for name in ("x", "y", "z"):
        if name in trace:
            trace[name] = _plain(_array(trace[name])[selected])
    marker = trace.get("marker")
    if isinstance(marker, dict):
        for name, value in tuple(marker.items()):
            if isinstance(value, dict) and "bdata" in value:
                values = _array(value)
                if len(values) == len(coordinates[0]):
                    marker[name] = _plain(values[selected])


def optimize(
    source: Path,
    output: Path,
    *,
    maximum_faces: int,
    maximum_points: int,
) -> dict:
    data, layout = _new_plot_arguments(source.read_text(encoding="utf-8"))
    original_bytes = source.stat().st_size
    original_faces = 0
    displayed_faces = 0
    original_points = 0
    displayed_points = 0
    for trace in data:
        trace_type = trace.get("type")
        if trace_type == "mesh3d":
            count = len(_array(trace["i"]))
            original_faces += count
            _simplify_mesh(trace, maximum_faces)
            displayed_faces += len(trace["i"])
        elif trace_type in {"scatter3d", "scattergl", "scatter"}:
            count = len(_array(trace.get("x", [])))
            original_points += count
            _simplify_scatter(trace, maximum_points)
            displayed_points += len(trace.get("x", []))
    layout.pop("template", None)
    layout["autosize"] = True
    layout["margin"] = {"l": 0, "r": 0, "t": 48, "b": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    go.Figure(data=data, layout=layout).write_html(
        output,
        include_plotlyjs="directory",
        full_html=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
    )
    return {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "original_bytes": original_bytes,
        "mobile_bytes": output.stat().st_size,
        "original_faces": original_faces,
        "displayed_faces": displayed_faces,
        "original_points": original_points,
        "displayed_points": displayed_points,
        "display_only": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-faces", type=int, default=1800)
    parser.add_argument("--maximum-points", type=int, default=3500)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = optimize(
        args.source,
        args.output,
        maximum_faces=args.maximum_faces,
        maximum_points=args.maximum_points,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
