#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.multiview_scene import (
    MultiviewFrame,
    automatic_world_grid,
    camera_pose_stability,
    gravity_level_transform,
    integrate_multiview_projective_depth,
    normalize_record3d_poses,
    record3d_pose_matrix,
    refine_pose_point_to_plane,
)
from rollout.scene_semantics import LABEL_BACKGROUND
from rollout.scene_volume import VoxelGrid


def _pose(translation=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0)):
    return {
        "translation_xyz_m": list(translation),
        "quaternion_xyzw": list(quaternion),
    }


def test_record3d_poses_are_normalized_to_first_camera():
    poses = normalize_record3d_poses(
        [_pose((1.0, 2.0, 3.0)), _pose((1.1, 2.0, 3.0))]
    )
    assert np.allclose(poses[0], np.eye(4))
    assert np.allclose(poses[1][:3, 3], [0.1, 0.0, 0.0])


def test_record3d_pose_converts_opencv_depth_axes_to_arkit_axes():
    pose = record3d_pose_matrix(_pose())
    assert np.allclose(pose[:3, :3] @ [1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert np.allclose(pose[:3, :3] @ [0.0, 1.0, 0.0], [0.0, -1.0, 0.0])
    assert np.allclose(pose[:3, :3] @ [0.0, 0.0, 1.0], [0.0, 0.0, -1.0])


def test_stopped_view_pose_stability_gate():
    accepted = camera_pose_stability(
        [_pose(), _pose((0.001, 0.0, 0.0)), _pose((0.002, 0.0, 0.0))]
    )
    rejected = camera_pose_stability(
        [_pose(), _pose((0.010, 0.0, 0.0))]
    )
    assert accepted["accepted"]
    assert not rejected["accepted"]


def test_gravity_level_transform_is_upright_and_not_mirrored():
    half = np.deg2rad(25.0) / 2.0
    first = _pose(
        quaternion=(np.sin(half), 0.0, 0.0, np.cos(half))
    )
    level = gravity_level_transform(first)
    reference_from_session = np.linalg.inv(record3d_pose_matrix(first))
    up_reference = reference_from_session[:3, :3] @ [0.0, 1.0, 0.0]
    assert np.isclose(np.linalg.det(level[:3, :3]), 1.0)
    assert np.allclose(level[:3, :3] @ level[:3, :3].T, np.eye(3))
    assert np.allclose(level[:3, :3] @ up_reference, [0.0, 0.0, 1.0])


def test_point_to_plane_refinement_recovers_small_translation():
    values = np.linspace(-0.25, 0.25, 18)
    yy, zz = np.meshgrid(values, values)
    plane_x = np.column_stack(
        (np.zeros(yy.size), yy.ravel(), zz.ravel())
    )
    xx, zz = np.meshgrid(values, values)
    plane_y = np.column_stack(
        (xx.ravel(), np.zeros(xx.size), zz.ravel())
    )
    xx, yy = np.meshgrid(values, values)
    plane_z = np.column_stack(
        (xx.ravel(), yy.ravel(), np.full(xx.size, 0.35))
    )
    target = np.vstack((plane_x, plane_y, plane_z))
    normals = np.vstack(
        (
            np.tile([1.0, 0.0, 0.0], (len(plane_x), 1)),
            np.tile([0.0, 1.0, 0.0], (len(plane_y), 1)),
            np.tile([0.0, 0.0, 1.0], (len(plane_z), 1)),
        )
    )
    true_pose = np.eye(4)
    true_pose[:3, 3] = [0.012, -0.009, 0.006]
    source = target - true_pose[:3, 3]
    initial = np.eye(4)
    result = refine_pose_point_to_plane(
        source,
        initial,
        target,
        normals,
        maximum_correspondence_m=0.05,
        acceptance_overlap=0.8,
    )
    assert result.accepted
    assert result.median_residual_m < 0.002
    assert np.allclose(
        result.reference_from_camera[:3, 3],
        true_pose[:3, 3],
        atol=0.002,
    )


def test_multiview_tsdf_preserves_unknown_and_excludes_dynamic_esdf():
    height, width = 30, 40
    matrix = np.array(
        [[45.0, 0.0, width / 2], [0.0, 45.0, height / 2], [0.0, 0.0, 1.0]]
    )
    depth = np.ones((height, width), dtype=float)
    confidence = np.full(depth.shape, 2, dtype=np.uint8)
    labels = np.full(depth.shape, LABEL_BACKGROUND, dtype=np.uint8)
    dynamic_label = 7
    labels[:, width // 2 :] = dynamic_label
    frames = [
        MultiviewFrame(
            name="center",
            rgb_bgr=np.zeros((height, width, 3), dtype=np.uint8),
            depth_m=depth,
            confidence=confidence,
            camera_matrix=matrix,
            reference_from_camera=np.eye(4),
            semantic_labels=labels,
        )
    ]
    grid = VoxelGrid(
        origin_xyz_m=np.array([-0.3, -0.25, 0.75]),
        voxel_size_m=0.02,
        shape_zyx=(20, 25, 30),
    )
    volume = integrate_multiview_projective_depth(
        frames,
        grid,
        truncation_m=0.04,
        dynamic_label_ids=(dynamic_label,),
    )
    assert np.any(volume.free)
    assert np.any(volume.unknown)
    dynamic = volume.semantic_labels == dynamic_label
    assert np.any(dynamic)
    assert np.all(np.isnan(volume.esdf_m[dynamic]))
    static = volume.occupied & ~dynamic
    assert np.any(static)
    assert np.all(volume.esdf_m[static] < 0.0)


def test_world_grid_rejects_unbounded_capture():
    points = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    try:
        automatic_world_grid(
            [points],
            voxel_size_m=0.005,
            percentile=0.0,
            maximum_voxels=10_000,
        )
    except ValueError as error:
        assert "voxels" in str(error)
    else:
        raise AssertionError("oversized multiview grid was accepted")


if __name__ == "__main__":
    test_record3d_poses_are_normalized_to_first_camera()
    test_record3d_pose_converts_opencv_depth_axes_to_arkit_axes()
    test_stopped_view_pose_stability_gate()
    test_gravity_level_transform_is_upright_and_not_mirrored()
    test_point_to_plane_refinement_recovers_small_translation()
    test_multiview_tsdf_preserves_unknown_and_excludes_dynamic_esdf()
    test_world_grid_rejects_unbounded_capture()
    print("multiview scene checks passed")
