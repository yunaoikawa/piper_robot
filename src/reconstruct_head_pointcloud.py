#!/usr/bin/env python3
"""Back-project a Record3D head depth frame into a colored point cloud."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.scene_3d import (
    backproject,
    register_point_clouds,
    scaled_camera_matrix,
)


def camera_calibration_from_profile(
    path: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    profile = json.loads(Path(path).read_text())
    matrix = np.asarray(
        profile["head_camera_matrix_rotated"], dtype=float
    )
    if matrix.shape != (3, 3):
        raise ValueError("invalid rotated head camera matrix")
    shape = profile.get("head_camera_reference_shape_hw")
    if (
        shape is None
        or len(shape) != 2
        or any(int(value) <= 0 for value in shape)
    ):
        raise ValueError(
            "profile requires head_camera_reference_shape_hw"
        )
    return matrix, (int(shape[0]), int(shape[1]))


def camera_matrix_from_profile(path: str) -> np.ndarray:
    """Backward-compatible matrix-only accessor."""

    return camera_calibration_from_profile(path)[0]


def align_depth(depth: np.ndarray, rgb_shape) -> np.ndarray:
    """Rotate portrait Record3D depth to match the displayed head RGB."""

    depth = np.asarray(depth, dtype=float)
    rgb_height, rgb_width = rgb_shape[:2]
    if depth.shape[1] / depth.shape[0] < 1.0:
        depth = np.rot90(depth, k=3)
    if abs(depth.shape[1] / depth.shape[0] - rgb_width / rgb_height) > 0.02:
        raise ValueError(
            f"depth {depth.shape} does not match RGB {rgb_shape[:2]}"
        )
    return depth


def bbox_mask(shape, bbox_xyxy):
    height, width = shape
    x0, y0, x1, y1 = np.asarray(bbox_xyxy, dtype=float)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    rx = max(1.0, 0.5 * (x1 - x0))
    ry = max(1.0, 0.5 * (y1 - y0))
    yy, xx = np.mgrid[:height, :width]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def write_ascii_ply(path: Path, points, colors, target):
    with path.open("w") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write(
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        )
        stream.write("property uchar target\nend_header\n")
        for point, color, selected in zip(points, colors, target):
            stream.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[2])} {int(color[1])} {int(color[0])} "
                f"{int(selected)}\n"
            )


def render_projection(points, colors, target, axes, label, size=(720, 520)):
    width, height = size
    values = points[:, axes]
    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    draw_colors = colors[finite].copy()
    draw_target = target[finite]
    lower = np.percentile(values, 1.0, axis=0)
    upper = np.percentile(values, 99.0, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    uv = (values - lower) / span
    px = np.clip((uv[:, 0] * (width - 40) + 20).astype(int), 0, width - 1)
    py = np.clip(
        ((1.0 - uv[:, 1]) * (height - 50) + 35).astype(int), 0, height - 1
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = 18
    canvas[py, px] = draw_colors
    canvas[
        py[draw_target],
        px[draw_target],
    ] = (0, 0, 255)
    cv2.putText(
        canvas,
        label,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
    )
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-bbox", type=float, nargs=4)
    parser.add_argument(
        "--reference-points",
        help="optional reference .npy cloud for AprilTag-free camera-shift check",
    )
    parser.add_argument("--min-depth", type=float, default=0.20)
    parser.add_argument("--max-depth", type=float, default=2.00)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise RuntimeError(f"could not read RGB image: {args.rgb}")
    depth = align_depth(np.load(args.depth), rgb.shape)
    full_matrix, matrix_reference_shape = camera_calibration_from_profile(
        args.profile
    )
    depth_matrix = scaled_camera_matrix(
        full_matrix, matrix_reference_shape, depth.shape
    )
    xyz = backproject(depth, depth_matrix)
    valid = (
        np.isfinite(depth)
        & (depth >= args.min_depth)
        & (depth <= args.max_depth)
    )
    target = np.zeros(depth.shape, dtype=bool)
    if args.target_bbox is not None:
        scale_x = depth.shape[1] / rgb.shape[1]
        scale_y = depth.shape[0] / rgb.shape[0]
        box = np.asarray(args.target_bbox, dtype=float)
        box[[0, 2]] *= scale_x
        box[[1, 3]] *= scale_y
        target = bbox_mask(depth.shape, box) & valid

    color_small = cv2.resize(
        rgb,
        (depth.shape[1], depth.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    points = xyz[valid]
    colors = color_small[valid]
    selected = target[valid]
    write_ascii_ply(
        output_dir / "head_scene.ply", points, colors, selected
    )
    np.save(output_dir / "head_scene_points.npy", points)
    registration = None
    if args.reference_points:
        result = register_point_clouds(
            points, np.load(args.reference_points)
        )
        registration = {
            "accepted": result.accepted,
            "rmse_m": result.rmse_m,
            "inlier_fraction": result.inlier_fraction,
            "iterations": result.iterations,
            "live_to_reference": result.live_to_reference.tolist(),
        }

    if np.any(selected):
        target_points = points[selected]
        target_center = np.median(target_points, axis=0)
        target_spread = np.percentile(
            target_points, [10, 90], axis=0
        )
    else:
        target_center = np.full(3, np.nan)
        target_spread = np.full((2, 3), np.nan)

    top = render_projection(
        points, colors, selected, (0, 2), "TOP: camera X-Z"
    )
    side = render_projection(
        points, colors, selected, (2, 1), "SIDE: camera Z-Y"
    )
    rotation = np.array(
        [
            [0.82, 0.0, -0.57],
            [-0.22, 0.92, -0.32],
            [0.52, 0.39, 0.76],
        ]
    )
    iso_points = points @ rotation.T
    iso = render_projection(
        iso_points,
        colors,
        selected,
        (0, 1),
        "ISOMETRIC POINT CLOUD",
    )
    depth_vis = np.zeros_like(depth, dtype=np.uint8)
    lo, hi = np.percentile(depth[valid], [2, 98])
    depth_vis[valid] = np.clip(
        (depth[valid] - lo) / max(hi - lo, 1e-6) * 255,
        0,
        255,
    ).astype(np.uint8)
    depth_vis = cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_TURBO)
    depth_vis[~valid] = 0
    depth_vis[target] = (0, 0, 255)
    depth_vis = cv2.resize(
        depth_vis, (720, 520), interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(
        depth_vis,
        "FRONT: Record3D depth",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
    )
    overview = np.vstack(
        (np.hstack((depth_vis, top)), np.hstack((side, iso)))
    )
    cv2.imwrite(str(output_dir / "head_scene_views.png"), overview)
    report = {
        "point_count": int(len(points)),
        "depth_range_m": [float(lo), float(hi)],
        "camera_matrix_full": full_matrix.tolist(),
        "camera_matrix_depth": depth_matrix.tolist(),
        "target_point_count": int(np.count_nonzero(selected)),
        "target_center_camera_xyz_m": target_center.tolist(),
        "target_10_90_percentile_camera_xyz_m": target_spread.tolist(),
        "rgb_source": str(Path(args.rgb)),
        "depth_source": str(Path(args.depth)),
        "registration": registration,
    }
    (output_dir / "head_scene.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
