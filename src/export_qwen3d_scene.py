#!/usr/bin/env python3
"""Export a Piper multiview scene as a Qwen-3D input/preview bundle.

This adapter is observation-only.  It preserves the posed RGB-D capture and
metric fused point cloud, and makes the semantic source explicit.  Existing
SAM labels are shown as bootstrap evidence; they are never misrepresented as
Qwen-3D predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go


SCHEMA = "piper_robot.qwen3d_scene_input/v1"
QWEN3D_UPSTREAM_IMAGE_SIZE = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgb_strings(colors: np.ndarray) -> list[str]:
    return [f"rgb({r},{g},{b})" for r, g, b in colors.astype(np.uint8)]


def _stratified_indices(labels: np.ndarray, maximum: int) -> np.ndarray:
    if len(labels) <= maximum:
        return np.arange(len(labels))
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    quota = max(64, maximum // max(1, len(groups)))
    chosen = []
    for group in groups:
        count = min(len(group), quota)
        chosen.append(rng.choice(group, count, replace=False))
    selected = np.unique(np.concatenate(chosen))
    remaining = maximum - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(len(labels)), selected, assume_unique=True)
        selected = np.r_[
            selected,
            rng.choice(pool, min(remaining, len(pool)), replace=False),
        ]
    return np.sort(selected[:maximum])


def _resize_rgbd_and_intrinsics(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    output_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize an aligned RGB-D frame while preserving its pinhole geometry."""
    source_height, source_width = depth.shape
    output_height, output_width = output_hw
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and depth must already be pixel-aligned")
    resized_rgb = cv2.resize(
        rgb, (output_width, output_height), interpolation=cv2.INTER_AREA
    )
    resized_depth = cv2.resize(
        depth, (output_width, output_height), interpolation=cv2.INTER_NEAREST
    )
    resized_intrinsics = np.asarray(intrinsics, dtype=float).copy()
    resized_intrinsics[0] *= output_width / source_width
    resized_intrinsics[1] *= output_height / source_height
    return resized_rgb, resized_depth, resized_intrinsics


def _view_records(report: dict, output_dir: Path) -> list[dict]:
    capture = Path(report["capture"])
    manifest = json.loads((capture / "manifest.json").read_text())
    registration = {
        item["name"]: item["reference_from_camera"]
        for item in report["registration"]["views"]
    }
    records = []
    frames_root = output_dir / "posed_rgbd"
    frames_root.mkdir(parents=True, exist_ok=True)
    upstream_root = output_dir / "qwen3d_upstream_rgbd" / "pasteur_scene"
    for subdirectory in ("color", "depth", "intrinsic", "pose"):
        (upstream_root / subdirectory).mkdir(parents=True, exist_ok=True)
    for output_index, view in enumerate(manifest["views"]):
        frames = view["frames"]
        frame = frames[len(frames) // 2]
        files = {
            key: str((capture / value["path"]).resolve())
            for key, value in frame["files"].items()
        }
        rgb = cv2.imread(files["rgb_png"], cv2.IMREAD_COLOR)
        depth = np.load(files["depth_npy"]).astype(np.float32)
        if rgb is None or depth.ndim != 2:
            raise ValueError(f"invalid RGB-D view {view['name']}")
        height, width = depth.shape
        source_height, source_width = rgb.shape[:2]
        aligned_rgb = cv2.resize(
            rgb, (width, height), interpolation=cv2.INTER_AREA
        )
        intrinsics = np.asarray(frame["intrinsics"]["K_raw_rgb"], dtype=float)
        intrinsics[0] *= width / source_width
        intrinsics[1] *= height / source_height
        pose = np.asarray(registration[view["name"]], dtype=float)
        frame_dir = frames_root / f"{output_index:03d}_{view['name']}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = frame_dir / "color.png"
        depth_path = frame_dir / "depth.png"
        intrinsic_path = frame_dir / "intrinsic.txt"
        pose_path = frame_dir / "pose.txt"
        cv2.imwrite(str(rgb_path), aligned_rgb)
        cv2.imwrite(
            str(depth_path),
            np.clip(np.rint(depth * 1000.0), 0, 65535).astype(np.uint16),
        )
        np.savetxt(intrinsic_path, intrinsics, fmt="%.10f")
        np.savetxt(pose_path, pose, fmt="%.10f")

        upstream_rgb, upstream_depth, upstream_intrinsics = (
            _resize_rgbd_and_intrinsics(
                aligned_rgb,
                depth,
                intrinsics,
                (QWEN3D_UPSTREAM_IMAGE_SIZE, QWEN3D_UPSTREAM_IMAGE_SIZE),
            )
        )
        upstream_stem = f"{output_index:06d}"
        upstream_paths = {
            "rgb": upstream_root / "color" / f"{upstream_stem}.png",
            "depth": upstream_root / "depth" / f"{upstream_stem}.png",
            "intrinsics": upstream_root
            / "intrinsic"
            / f"{upstream_stem}.txt",
            "pose": upstream_root / "pose" / f"{upstream_stem}.txt",
        }
        cv2.imwrite(str(upstream_paths["rgb"]), upstream_rgb)
        cv2.imwrite(
            str(upstream_paths["depth"]),
            np.clip(np.rint(upstream_depth * 1000.0), 0, 65535).astype(
                np.uint16
            ),
        )
        np.savetxt(
            upstream_paths["intrinsics"], upstream_intrinsics, fmt="%.10f"
        )
        np.savetxt(upstream_paths["pose"], pose, fmt="%.10f")
        records.append(
            {
                "name": view["name"],
                "rgb": str(rgb_path.resolve()),
                "depth": str(depth_path.resolve()),
                "intrinsics": intrinsics.tolist(),
                "intrinsics_path": str(intrinsic_path.resolve()),
                "T_scene_camera": pose.tolist(),
                "pose_path": str(pose_path.resolve()),
                "image_shape_hw": [height, width],
                "depth_unit": "mm_uint16",
                "source_frame_index": frame["source_frame_index"],
                "source_rgb": files["rgb_png"],
                "source_depth": files["depth_npy"],
                "qwen3d_upstream": {
                    key: str(path.resolve())
                    for key, path in upstream_paths.items()
                },
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiview-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-display-points", type=int, default=45000)
    args = parser.parse_args()

    report = json.loads(args.multiview_report.read_text())
    mesh_path = Path(report["artifacts"]["mesh_ply"]).with_suffix(".npz")
    if not mesh_path.exists():
        mesh_path = args.multiview_report.parent / "scene_mesh_multiview.npz"
    mesh = np.load(mesh_path)
    points = np.asarray(mesh["vertices_xyz_m"], dtype=np.float32)
    colors = np.asarray(mesh["colors_rgb"], dtype=np.uint8)
    labels = np.asarray(mesh["semantic_labels"], dtype=np.uint16)
    semantics = report["semantics"]
    id_to_name = {
        int(value): name for name, value in semantics["label_ids"].items()
    }
    id_to_name.setdefault(2, "measured_background")
    indices = _stratified_indices(labels, args.maximum_display_points)
    display_points = points[indices]
    display_colors = colors[indices]
    display_labels = labels[indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "qwen3d_input_points.npz",
        points_xyz_m=points,
        colors_rgb=colors,
        bootstrap_semantic_labels=labels,
    )

    traces = [
        go.Scatter3d(
            x=display_points[:, 0],
            y=display_points[:, 1],
            z=display_points[:, 2],
            mode="markers",
            marker={
                "size": 1.7,
                "color": _rgb_strings(display_colors),
                "opacity": 0.68,
            },
            name="RGB-D geometry",
            hoverinfo="skip",
            visible=True,
        )
    ]
    semantic_trace_names = []
    for label in sorted(np.unique(display_labels)):
        mask = display_labels == label
        name = id_to_name.get(int(label), f"label_{int(label)}")
        semantic_trace_names.append(name)
        color = np.asarray(
            semantics["label_colors_rgb"].get(str(int(label)), [180, 180, 180])
        )
        traces.append(
            go.Scatter3d(
                x=display_points[mask, 0],
                y=display_points[mask, 1],
                z=display_points[mask, 2],
                mode="markers",
                marker={
                    "size": 2.5,
                    "color": f"rgb({color[0]},{color[1]},{color[2]})",
                    "opacity": 0.92,
                },
                name=f"bootstrap: {name}",
                hovertemplate=f"{name}<extra></extra>",
                visible=False,
            )
        )

    count = len(traces)
    buttons = [
        {
            "label": "RGB-D",
            "method": "update",
            "args": [{"visible": [True] + [False] * (count - 1)}],
        },
        {
            "label": "All semantic regions",
            "method": "update",
            "args": [{"visible": [False] + [True] * (count - 1)}],
        },
    ]
    for index, name in enumerate(semantic_trace_names, start=1):
        visible = [False] * count
        visible[0] = True
        visible[index] = True
        buttons.append(
            {
                "label": name,
                "method": "update",
                "args": [{"visible": visible}],
            }
        )
    figure = go.Figure(data=traces)
    figure.update_layout(
        title=(
            "Qwen-3D posed RGB-D input preview — semantic colors are SAM "
            "bootstrap, not Qwen-3D predictions"
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
        legend={"orientation": "h", "y": 0.01},
    )
    html_path = args.output_dir / "index.html"
    figure.write_html(
        html_path,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    manifest = {
        "schema": SCHEMA,
        "source_repository": "https://github.com/ll220/qwen3d",
        "source_revision_reviewed": "7ef6d01e495290639884878a06e43ba2905e0ef5",
        "multiview_report": str(args.multiview_report.resolve()),
        "multiview_report_sha256": _sha256(args.multiview_report),
        "coordinate_frame": report["coordinate_frame"],
        "posed_rgbd_views": _view_records(report, args.output_dir),
        "point_cloud": {
            "path": str((args.output_dir / "qwen3d_input_points.npz").resolve()),
            "point_count": int(len(points)),
            "units": "m",
        },
        "queries": [
            "the incubator on the raised rear platform",
            "the microscope to the left of the incubator",
            "the two robot arms",
            "the transparent petri dish and its lid",
            "the culture media bottle between the platforms",
        ],
        "bootstrap_semantics": {
            "source": "existing_reviewed_sam3_multiview_masks",
            "qwen3d_inference_completed": False,
            "label_ids": semantics["label_ids"],
        },
        "readiness": {
            "qwen3d_input_ready": True,
            "qwen3d_inference_ready": False,
            "motion_authority": False,
            "reason": "Qwen-3D checkpoint/runtime requires a working CUDA GPU",
        },
        "viewer": str(html_path.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
