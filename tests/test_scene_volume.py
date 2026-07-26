#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.scene_volume import (
    VoxelGrid,
    integrate_projective_depth,
    level_transform,
    organized_depth_mesh,
    transform_points,
)


def test_projective_esdf_preserves_unknown_space():
    height, width = 80, 100
    matrix = np.array(
        [[90.0, 0.0, width / 2], [0.0, 90.0, height / 2], [0, 0, 1]]
    )
    depth = np.ones((height, width), dtype=float)
    grid = VoxelGrid(
        origin_xyz_m=np.array([-0.20, -0.15, 0.70]),
        voxel_size_m=0.01,
        shape_zyx=(45, 40, 50),
    )
    volume = integrate_projective_depth(
        depth, matrix, grid, truncation_m=0.03
    )
    x, y, z = grid.centers()
    ix = int(np.argmin(abs(x)))
    iy = int(np.argmin(abs(y)))
    free_z = int(np.argmin(abs(z - 0.90)))
    surface_z = int(np.argmin(abs(z - 1.005)))
    unknown_z = int(np.argmin(abs(z - 1.10)))
    assert volume.free[free_z, iy, ix]
    assert volume.esdf_m[free_z, iy, ix] > 0
    assert volume.occupied[surface_z, iy, ix]
    assert volume.esdf_m[surface_z, iy, ix] < 0
    assert volume.unknown[unknown_z, iy, ix]
    assert np.isnan(volume.esdf_m[unknown_z, iy, ix])


def test_confidence_zero_is_not_ray_carved():
    depth = np.ones((20, 20), dtype=float)
    confidence = np.full_like(depth, 2, dtype=np.uint8)
    confidence[:, :10] = 0
    matrix = np.array([[30.0, 0, 10.0], [0, 30.0, 10.0], [0, 0, 1]])
    grid = VoxelGrid(np.array([-0.2, -0.2, 0.8]), 0.02, (12, 20, 20))
    volume = integrate_projective_depth(
        depth,
        matrix,
        grid,
        confidence=confidence,
        minimum_confidence=1,
    )
    assert np.any(volume.observed[:, :, 10:])
    assert not np.any(volume.observed[:, :, :4])


def test_depth_mesh_rejects_discontinuity():
    depth = np.ones((12, 16), dtype=float)
    depth[:, 8:] = 1.4
    matrix = np.array([[20.0, 0, 8.0], [0, 20.0, 6.0], [0, 0, 1]])
    mesh = organized_depth_mesh(
        depth, matrix, maximum_edge_m=0.10
    )
    assert len(mesh.vertices_xyz_m) == depth.size
    for face in mesh.faces:
        z = mesh.vertices_xyz_m[face, 2]
        assert np.ptp(z) < 0.1


def test_support_plane_is_levelled():
    normal = np.array([0.25, -0.75, -0.61])
    normal /= np.linalg.norm(normal)
    offset = 0.62
    rotation, translation = level_transform(normal, offset)
    point = -offset * normal
    tangent_a = np.cross(normal, [1.0, 0.0, 0.0])
    tangent_b = np.cross(normal, tangent_a)
    points = np.stack((point, point + tangent_a, point + tangent_b))
    levelled = transform_points(points, rotation, translation)
    assert np.max(np.abs(levelled[:, 2])) < 1e-9
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)


if __name__ == "__main__":
    test_projective_esdf_preserves_unknown_space()
    test_confidence_zero_is_not_ray_carved()
    test_depth_mesh_rejects_discontinuity()
    test_support_plane_is_levelled()
    print("scene volume checks passed")
