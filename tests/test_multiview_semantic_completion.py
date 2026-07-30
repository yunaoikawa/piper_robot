#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.semantic_scene_pipeline import ObjectGeometry
from src.build_multiview_semantic_scene import (
    _support_for,
    discover_multilevel_supports,
    fit_box_to_semantic_volume,
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


def test_semantic_volume_fit_rejects_free_space_and_recovers_box_pose():
    rng = np.random.default_rng(23)
    center = np.array([0.18, 0.42])
    yaw = 0.48
    size = np.array([0.40, 0.22, 0.30])
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    )
    values = np.linspace(-1.0, 1.0, 24)
    first = np.array(
        [[-size[0] / 2, value * size[1] / 2] for value in values]
    )
    second = np.array(
        [[value * size[0] / 2, -size[1] / 2] for value in values]
    )
    surface_xy = np.vstack((first, second)) @ rotation.T + center
    surface = np.vstack(
        [
            np.c_[
                surface_xy,
                np.full(len(surface_xy), z),
            ]
            for z in np.linspace(0.03, size[2] - 0.03, 8)
        ]
    )
    local_semantic = rng.uniform(-0.5, 0.5, (1600, 3)) * size
    semantic = local_semantic.copy()
    semantic[:, :2] = semantic[:, :2] @ rotation.T + center
    semantic[:, 2] += size[2] / 2
    free = rng.uniform([-0.25, 0.0, 0.0], [0.65, 0.85, 0.32], (9000, 3))
    delta = free[:, :2] - center
    local_xy = delta @ rotation
    inside = (
        (np.abs(local_xy[:, 0]) <= size[0] / 2)
        & (np.abs(local_xy[:, 1]) <= size[1] / 2)
        & (free[:, 2] <= size[2])
    )
    free = free[~inside]
    initial = ObjectGeometry(
        kind="box",
        center_xyz_m=(0.02, 0.28, size[2] / 2),
        size_xyz_m=tuple(size),
        yaw_rad=-0.25,
    )
    fitted, report = fit_box_to_semantic_volume(
        surface,
        semantic,
        free,
        initial,
        support_height_m=0.0,
        voxel_size_m=0.01,
        configuration={
            "minimum_observed_points": 100,
            "maximum_surface_points": 1000,
            "maximum_semantic_voxels": 1200,
            "maximum_free_voxels": 3500,
            "maximum_iterations": 22,
            "population_size": 7,
            "minimum_improvement_fraction": 0.10,
            "seed": 9,
        },
    )
    assert report["accepted"]
    assert np.linalg.norm(np.asarray(fitted.center_xyz_m[:2]) - center) < 0.04
    yaw_error = abs(
        np.arctan2(
            np.sin(fitted.yaw_rad - yaw),
            np.cos(fitted.yaw_rad - yaw),
        )
    )
    yaw_error = min(yaw_error, abs(np.pi - yaw_error))
    assert np.degrees(yaw_error) < 8.0
    assert (
        report["optimized"]["known_free_intrusion_fraction"]
        < report["initial"]["known_free_intrusion_fraction"]
    )


if __name__ == "__main__":
    test_voxel_components_separate_two_arms()
    test_multilevel_supports_keep_two_raised_platforms_and_bench()
    test_per_view_robot_state_gate_is_read_only_and_strict()
    test_semantic_support_roles_select_front_and_rear_without_pixel_rules()
    test_cad_fit_recovers_synthetic_camera_to_robot_transform()
    test_semantic_volume_fit_rejects_free_space_and_recovers_box_pose()
    print("multiview semantic completion checks passed")
