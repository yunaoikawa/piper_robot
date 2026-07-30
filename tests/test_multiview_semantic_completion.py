#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_multiview_semantic_scene import (
    _support_for,
    discover_multilevel_supports,
    transform_points,
    voxel_components,
)
from src.calibrate_head_robot_from_cad import fit_transform
from src.capture_record3d_multiview import _robot_state_stability


def _plane(x0, x1, y0, y1, z, rows=14, columns=18):
    x = np.linspace(x0, x1, columns)
    y = np.linspace(y0, y1, rows)
    xx, yy = np.meshgrid(x, y)
    vertices = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z)]
    grid = np.arange(len(vertices)).reshape(rows, columns)
    faces = np.concatenate(
        (
            np.stack(
                (grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1]), axis=-1
            ).reshape(-1, 3),
            np.stack(
                (grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]), axis=-1
            ).reshape(-1, 3),
        )
    )
    return vertices, faces


def test_voxel_components_separate_two_arms():
    rng = np.random.default_rng(7)
    left = rng.normal([-0.25, 0.0, 0.3], 0.015, (120, 3))
    right = rng.normal([0.25, 0.0, 0.3], 0.015, (100, 3))
    components = voxel_components(np.vstack((left, right)), voxel_size_m=0.05)
    assert [len(item) for item in components[:2]] == [120, 100]


def test_multilevel_supports_keep_two_raised_platforms_and_bench():
    surfaces = [
        _plane(-0.7, 0.7, -0.3, 0.6, 0.0),
        _plane(-0.55, -0.10, 0.0, 0.35, 0.16),
        _plane(0.10, 0.55, 0.0, 0.35, 0.16),
    ]
    vertices = []
    faces = []
    offset = 0
    for surface_vertices, surface_faces in surfaces:
        vertices.append(surface_vertices)
        faces.append(surface_faces + offset)
        offset += len(surface_vertices)
    vertices = np.vstack(vertices)
    faces = np.vstack(faces)
    supports = discover_multilevel_supports(
        vertices,
        faces,
        np.full(len(vertices), 2, dtype=np.int32),
        minimum_area_m2=0.015,
    )
    assert len(supports) == 3
    assert supports[0]["support_id"] == "support-bench"
    assert np.isclose(supports[0]["height_m"], 0.0, atol=0.01)
    assert np.isclose(supports[1]["height_m"], 0.16, atol=0.01)
    assert np.isclose(supports[2]["height_m"], 0.16, atol=0.01)
    assert all(item["holes_preserved"] for item in supports)


def test_per_view_robot_state_gate_is_read_only_and_strict():
    before = {
        "left_joint_positions_rad": [0.0] * 6,
        "right_joint_positions_rad": [0.0] * 6,
    }
    after = {
        "left_joint_positions_rad": [0.001] * 6,
        "right_joint_positions_rad": [0.002] * 6,
    }
    accepted = _robot_state_stability(
        before, after, maximum_joint_delta_rad=0.005
    )
    rejected = _robot_state_stability(
        before,
        {
            **after,
            "right_joint_positions_rad": [0.010] * 6,
        },
        maximum_joint_delta_rad=0.005,
    )
    assert accepted["accepted"]
    assert not rejected["accepted"]


def test_semantic_support_roles_select_front_and_rear_without_pixel_rules():
    supports = [
        {
            "support_id": "bench",
            "height_m": 0.0,
            "bounds_xy_m": [[-1.0, -1.0], [1.0, 1.0]],
        },
        {
            "support_id": "front",
            "height_m": 0.16,
            "bounds_xy_m": [[-0.5, 0.1], [0.5, 0.5]],
        },
        {
            "support_id": "rear",
            "height_m": 0.16,
            "bounds_xy_m": [[-0.5, 0.6], [0.5, 1.0]],
        },
    ]
    profile = {
        "support_assignment": {
            "depth_axis": 1,
            "depth_sign": 1,
            "semantic_roles": {
                "incubator": "rear_elevated",
                "petri_lid": "front_elevated",
            },
        }
    }
    points = np.array([[0.0, 0.55, 0.2], [0.1, 0.55, 0.3], [-0.1, 0.55, 0.4]])
    assert (
        _support_for(
            points,
            supports,
            semantic_name="incubator",
            profile=profile,
        )["support_id"]
        == "rear"
    )
    assert (
        _support_for(
            points,
            supports,
            semantic_name="petri_lid",
            profile=profile,
        )["support_id"]
        == "front"
    )


def test_cad_fit_recovers_synthetic_camera_to_robot_transform():
    rng = np.random.default_rng(11)
    base = np.vstack(
        (
            rng.normal([-0.25, 0.05, 0.25], [0.04, 0.02, 0.10], (400, 3)),
            rng.normal([0.22, -0.03, 0.42], [0.03, 0.06, 0.04], (350, 3)),
        )
    )
    cad = [
        base,
        base + [0.02, 0.0, 0.04],
        base + [-0.01, 0.03, -0.02],
    ]
    truth = np.eye(4)
    truth[:3, :3] = Rotation.from_euler(
        "xyz", [0.07, -0.04, 0.18]
    ).as_matrix()
    truth[:3, 3] = [0.13, -0.08, 0.21]
    camera_from_robot = np.linalg.inv(truth)
    observed = [
        transform_points(points, camera_from_robot)
        + rng.normal(0.0, 0.0005, points.shape)
        for points in cad
    ]
    initial = truth.copy()
    initial[:3, 3] += [0.01, -0.01, 0.008]
    fitted = fit_transform(observed, cad, initial, maximum_evaluations=80)
    assert np.linalg.norm(fitted[:3, 3] - truth[:3, 3]) < 0.003
    rotation_error = Rotation.from_matrix(
        fitted[:3, :3] @ truth[:3, :3].T
    ).magnitude()
    assert np.degrees(rotation_error) < 0.3


if __name__ == "__main__":
    test_voxel_components_separate_two_arms()
    test_multilevel_supports_keep_two_raised_platforms_and_bench()
    test_per_view_robot_state_gate_is_read_only_and_strict()
    test_semantic_support_roles_select_front_and_rear_without_pixel_rules()
    test_cad_fit_recovers_synthetic_camera_to_robot_transform()
    print("multiview semantic completion checks passed")
