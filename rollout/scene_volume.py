"""Conservative projective TSDF/ESDF maps for offline RGB-D scenes.

The map keeps unknown space separate from observed free space.  This matters
for manipulation: space hidden behind the first depth return must never become
collision-free merely because no point was measured there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from rollout.scene_3d import backproject
from rollout.scene_semantics import (
    LABEL_BACKGROUND,
    LABEL_FREE,
    LABEL_ROBOT,
    LABEL_UNKNOWN,
)


@dataclass(frozen=True)
class VoxelGrid:
    """Axis-aligned camera-frame voxel grid with arrays stored as (z, y, x)."""

    origin_xyz_m: np.ndarray
    voxel_size_m: float
    shape_zyx: tuple[int, int, int]

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nz, ny, nx = self.shape_zyx
        origin = np.asarray(self.origin_xyz_m, dtype=float)
        x = origin[0] + (np.arange(nx) + 0.5) * self.voxel_size_m
        y = origin[1] + (np.arange(ny) + 0.5) * self.voxel_size_m
        z = origin[2] + (np.arange(nz) + 0.5) * self.voxel_size_m
        return x, y, z


@dataclass(frozen=True)
class SceneVolume:
    grid: VoxelGrid
    tsdf: np.ndarray
    observed: np.ndarray
    esdf_m: np.ndarray
    semantic_labels: np.ndarray

    @property
    def free(self) -> np.ndarray:
        return self.observed & (self.tsdf > 0.0)

    @property
    def occupied(self) -> np.ndarray:
        return self.observed & (self.tsdf <= 0.0)

    @property
    def unknown(self) -> np.ndarray:
        return ~self.observed


@dataclass(frozen=True)
class TriangleMesh:
    vertices_xyz_m: np.ndarray
    faces: np.ndarray
    colors_rgb: np.ndarray
    semantic_labels: np.ndarray | None = None


def automatic_grid(
    depth_m,
    camera_matrix,
    *,
    voxel_size_m: float = 0.01,
    truncation_m: float = 0.03,
    min_depth_m: float = 0.20,
    max_depth_m: float = 2.00,
    percentile: float = 0.5,
    lateral_margin_m: float = 0.025,
) -> VoxelGrid:
    """Choose robust scene bounds without allowing isolated depth outliers."""

    depth = np.asarray(depth_m, dtype=float)
    valid = (
        np.isfinite(depth)
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
    )
    points = backproject(depth, camera_matrix)[valid]
    if len(points) < 100:
        raise ValueError("too few valid depth pixels for a scene volume")
    lower = np.percentile(points, percentile, axis=0)
    upper = np.percentile(points, 100.0 - percentile, axis=0)
    lower[:2] -= lateral_margin_m
    upper[:2] += lateral_margin_m
    lower[2] = max(min_depth_m, lower[2] - lateral_margin_m)
    upper[2] = min(
        max_depth_m, upper[2] + truncation_m + lateral_margin_m
    )
    origin = np.floor(lower / voxel_size_m) * voxel_size_m
    shape_xyz = np.ceil((upper - origin) / voxel_size_m).astype(int)
    if np.any(shape_xyz < 2):
        raise ValueError("automatic volume bounds are degenerate")
    return VoxelGrid(
        origin_xyz_m=origin,
        voxel_size_m=float(voxel_size_m),
        shape_zyx=tuple(int(v) for v in shape_xyz[::-1]),
    )


def integrate_projective_depth(
    depth_m,
    camera_matrix,
    grid: VoxelGrid,
    *,
    truncation_m: float = 0.03,
    min_depth_m: float = 0.20,
    max_depth_m: float = 2.00,
    confidence=None,
    minimum_confidence: int = 1,
    surface_labels=None,
) -> SceneVolume:
    """Integrate one depth image while ray-carving only measured free space.

    Voxels in front of a valid return and within ``truncation_m`` behind it are
    observed.  Everything farther behind the return or outside the image
    remains unknown.
    """

    depth = np.asarray(depth_m, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    valid_depth = (
        np.isfinite(depth)
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
    )
    if confidence is not None:
        confidence = np.asarray(confidence)
        if confidence.shape != depth.shape:
            raise ValueError("confidence and depth shapes differ")
        valid_depth &= confidence >= minimum_confidence
    if surface_labels is not None:
        surface_labels = np.asarray(surface_labels, dtype=np.uint8)
        if surface_labels.shape != depth.shape:
            raise ValueError("surface labels and depth shapes differ")

    nz, ny, nx = grid.shape_zyx
    x, y, z = grid.centers()
    xx, yy = np.meshgrid(x, y)
    tsdf = np.ones((nz, ny, nx), dtype=np.float32)
    observed = np.zeros((nz, ny, nx), dtype=bool)
    semantics = np.full((nz, ny, nx), LABEL_UNKNOWN, dtype=np.uint8)
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    for iz, z_value in enumerate(z):
        if z_value <= 0:
            continue
        u = np.rint(fx * xx / z_value + cx).astype(np.int32)
        v = np.rint(fy * yy / z_value + cy).astype(np.int32)
        inside = (
            (u >= 0)
            & (u < depth.shape[1])
            & (v >= 0)
            & (v < depth.shape[0])
        )
        sampled = np.full((ny, nx), np.nan, dtype=float)
        sampled_valid = np.zeros((ny, nx), dtype=bool)
        sampled[inside] = depth[v[inside], u[inside]]
        sampled_valid[inside] = valid_depth[v[inside], u[inside]]
        signed = sampled - z_value
        layer_observed = sampled_valid & (signed >= -truncation_m)
        observed[iz] = layer_observed
        tsdf[iz, layer_observed] = np.clip(
            signed[layer_observed] / truncation_m, -1.0, 1.0
        )
        layer_free = layer_observed & (signed > 0.0)
        semantics[iz, layer_free] = LABEL_FREE
        layer_surface = layer_observed & ~layer_free
        if surface_labels is None:
            semantics[iz, layer_surface] = LABEL_BACKGROUND
        else:
            sampled_labels = np.full((ny, nx), LABEL_BACKGROUND, dtype=np.uint8)
            sampled_labels[inside] = surface_labels[v[inside], u[inside]]
            semantics[iz, layer_surface] = sampled_labels[layer_surface]

    free = observed & (tsdf > 0.0)
    occupied = observed & ~free
    # Robot returns are a dynamic semantic layer.  They remain labelled and
    # visible, but are not baked into the static-environment ESDF because the
    # complete posed robot CAD is responsible for robot geometry.
    static_occupied = occupied & (semantics != LABEL_ROBOT)
    esdf = np.full(tsdf.shape, np.nan, dtype=np.float32)
    if np.any(static_occupied):
        distance_to_occupied = distance_transform_edt(
            ~static_occupied, sampling=grid.voxel_size_m
        )
        esdf[free] = distance_to_occupied[free]
    if np.any(free):
        distance_to_free = distance_transform_edt(
            ~free, sampling=grid.voxel_size_m
        )
        esdf[static_occupied] = -distance_to_free[static_occupied]
    return SceneVolume(
        grid=grid,
        tsdf=tsdf,
        observed=observed,
        esdf_m=esdf,
        semantic_labels=semantics,
    )


def level_transform(support_normal, support_offset: float):
    """Return a rigid camera-to-level transform with the support plane at z=0."""

    up = np.asarray(support_normal, dtype=float).reshape(3)
    up /= np.linalg.norm(up)
    if up[2] > 0:
        up = -up
        support_offset = -support_offset
    right_hint = np.array([1.0, 0.0, 0.0])
    right = right_hint - up * float(right_hint @ up)
    right /= np.linalg.norm(right)
    forward = np.cross(up, right)
    forward /= np.linalg.norm(forward)
    rotation = np.stack((right, forward, up), axis=0)
    plane_origin = -float(support_offset) * up
    translation = -rotation @ plane_origin
    return rotation, translation


def transform_points(points, rotation, translation) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points @ np.asarray(rotation, dtype=float).T + np.asarray(
        translation, dtype=float
    )


def organized_depth_mesh(
    depth_m,
    camera_matrix,
    *,
    rgb=None,
    stride: int = 1,
    min_depth_m: float = 0.20,
    max_depth_m: float = 2.00,
    maximum_edge_m: float = 0.045,
    semantic_labels=None,
) -> TriangleMesh:
    """Triangulate adjacent depth samples and reject depth discontinuities."""

    depth = np.asarray(depth_m, dtype=float)[::stride, ::stride]
    matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3).copy()
    matrix[0, :] /= stride
    matrix[1, :] /= stride
    matrix[2, :] = (0.0, 0.0, 1.0)
    points = backproject(depth, matrix)
    valid = (
        np.isfinite(depth)
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
    )
    index = np.full(depth.shape, -1, dtype=np.int32)
    index[valid] = np.arange(np.count_nonzero(valid), dtype=np.int32)
    vertices = points[valid]
    if rgb is None:
        colors = np.full((len(vertices), 3), 190, dtype=np.uint8)
    else:
        import cv2

        image = cv2.resize(
            np.asarray(rgb),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        colors = image[valid][:, ::-1].copy()
    vertex_labels = None
    if semantic_labels is not None:
        import cv2

        label_image = cv2.resize(
            np.asarray(semantic_labels, dtype=np.uint8),
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        vertex_labels = label_image[valid]

    faces = []
    h, w = depth.shape
    candidates = (
        ((0, 0), (0, 1), (1, 0)),
        ((0, 1), (1, 1), (1, 0)),
    )
    for dyx in candidates:
        offsets = np.asarray(dyx)
        y0, x0 = offsets.min(axis=0)
        y1, x1 = offsets.max(axis=0)
        ys = np.arange(h - (y1 - y0))
        xs = np.arange(w - (x1 - x0))
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        triangle_indices = np.stack(
            [index[gy + dy, gx + dx] for dy, dx in dyx], axis=-1
        )
        candidate_valid = np.all(triangle_indices >= 0, axis=-1)
        flat = triangle_indices[candidate_valid]
        if len(flat):
            tri_points = vertices[flat]
            edges = np.stack(
                (
                    np.linalg.norm(tri_points[:, 1] - tri_points[:, 0], axis=1),
                    np.linalg.norm(tri_points[:, 2] - tri_points[:, 1], axis=1),
                    np.linalg.norm(tri_points[:, 0] - tri_points[:, 2], axis=1),
                ),
                axis=1,
            )
            faces.append(flat[np.max(edges, axis=1) <= maximum_edge_m])
    face_array = (
        np.concatenate(faces, axis=0)
        if faces
        else np.empty((0, 3), dtype=np.int32)
    )
    return TriangleMesh(vertices, face_array, colors, vertex_labels)


def unknown_frontier(volume: SceneVolume) -> np.ndarray:
    """Unknown voxels immediately adjacent to observed free space."""

    return volume.unknown & binary_dilation(volume.free, iterations=1)


def voxel_centers_for_mask(
    grid: VoxelGrid, mask, *, maximum_points: int | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argwhere(np.asarray(mask, dtype=bool))
    if maximum_points and len(indices) > maximum_points:
        rng = np.random.default_rng(seed)
        indices = indices[
            np.sort(rng.choice(len(indices), maximum_points, replace=False))
        ]
    x, y, z = grid.centers()
    points = np.column_stack(
        (x[indices[:, 2]], y[indices[:, 1]], z[indices[:, 0]])
    )
    return points, indices
