#!/usr/bin/env python3

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.scene_3d import (
    assess_target_geometry,
    backproject,
    estimate_target_on_support_plane,
    nearest_scene_distance,
    register_point_clouds,
    temporal_median_depth,
)


def test_temporal_median():
    frames = [np.ones((4, 5)), np.ones((4, 5)) * 1.02, np.ones((4, 5))]
    frames[1][1, 1] = 0.0
    result = temporal_median_depth(frames)
    assert np.allclose(result, 1.0)


def test_transparent_target_uses_support_plane():
    height, width = 180, 240
    matrix = np.array(
        [[220.0, 0.0, width / 2], [0.0, 220.0, height / 2], [0, 0, 1]]
    )
    yy, xx = np.mgrid[:height, :width]
    # Slightly tilted bench.
    depth = 0.86 + 0.00012 * xx - 0.00008 * yy
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (128, 96), 22, 1, -1)
    # Transparent/reflective returns inside the object are deliberately wrong.
    depth[mask.astype(bool)] = 1.24
    estimate = estimate_target_on_support_plane(
        depth, matrix, mask, np.array([108.0, 96.0]), ring_margin_px=35
    )
    expected_z = 0.86 + 0.00012 * 108 - 0.00008 * 96
    assert abs(estimate.point_camera_xyz_m[2] - expected_z) < 0.006
    assert estimate.plane.inlier_fraction > 0.90
    assert estimate.confidence > 0.50
    quality = assess_target_geometry(
        estimate,
        matrix,
        native_pixel_stride_xy=(4.0, 4.0),
        maximum_view_angle_deg=60.0,
        maximum_native_footprint_m=0.030,
    )
    assert quality.accepted
    assert quality.native_pixel_footprint_x_m > 0


def test_small_scene_registration():
    rng = np.random.default_rng(4)
    reference = rng.uniform([-0.25, -0.15, 0.65], [0.25, 0.15, 1.05], (900, 3))
    angle = np.deg2rad(1.2)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.008, -0.006, 0.004])
    # live -> reference is the transform under test.
    live = (reference - translation) @ rotation
    result = register_point_clouds(
        live,
        reference,
        max_correspondence_m=0.05,
        acceptance_rmse_m=0.01,
        acceptance_inlier_fraction=0.80,
    )
    moved = live @ result.live_to_reference[:3, :3].T
    moved += result.live_to_reference[:3, 3]
    assert result.accepted
    assert np.median(np.linalg.norm(moved - reference, axis=1)) < 0.003


def test_proximity_is_warning_metric_only():
    scene = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    assert np.isclose(nearest_scene_distance([0.03, 0.0, 0.0], scene), 0.03)


if __name__ == "__main__":
    test_temporal_median()
    test_transparent_target_uses_support_plane()
    test_small_scene_registration()
    test_proximity_is_warning_metric_only()
    print("AprilTag-free 3D scene checks passed")
