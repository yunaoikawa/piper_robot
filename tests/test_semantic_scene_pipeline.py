from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from rollout.semantic_scene_pipeline import (
    CatalogObject,
    MaskObservation,
    aabb_intersections,
    choose_support,
    discover_supports,
    exclusive_masks,
    quality_score,
    robust_oriented_geometry,
    support_collision_boxes,
)


def organized_plane(height=32, width=48):
    x = np.linspace(-0.4, 0.4, width)
    y = np.linspace(-0.25, 0.25, height)
    xx, yy = np.meshgrid(x, y)
    vertices = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
    grid = np.arange(height * width).reshape(height, width)
    faces = np.concatenate(
        [
            np.stack(
                [grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1]], axis=-1
            ).reshape(-1, 3),
            np.stack(
                [grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=-1
            ).reshape(-1, 3),
        ]
    )
    return vertices, faces


def definition(transparent=False):
    return CatalogObject.from_dict(
        {
            "name": "sample",
            "prompts": ["sample object"],
            "completion": "primitive",
            "primitive": "cylinder",
            "nominal_size_m": [0.10, 0.10, 0.04],
            "size_range_m": [[0.06, 0.06, 0.02], [0.14, 0.14, 0.08]],
            "transparent": transparent,
            "support": "horizontal_surface",
            "minimum_confidence": 0.7,
            "color": "#fff000",
        }
    )


def test_support_discovery_and_supported_completion():
    points, _ = organized_plane()
    valid = np.ones(points.shape[:2], dtype=bool)
    supports = discover_supports(
        points.reshape(-1, 3),
        valid.reshape(-1),
        valid.shape,
        height_tolerance_m=0.004,
        minimum_area_fraction=0.01,
    )
    assert supports
    object_points = np.array(
        [
            [-0.04, -0.03, 0.02],
            [0.04, -0.03, 0.02],
            [-0.04, 0.03, 0.04],
            [0.04, 0.03, 0.04],
        ]
    )
    support = choose_support(object_points, supports)
    assert support is not None
    geometry = robust_oriented_geometry(
        object_points,
        catalog=definition(transparent=True),
        support_height_m=support["height_m"],
    )
    assert np.allclose(geometry.size_xyz_m, [0.10, 0.10, 0.04])
    assert np.isclose(geometry.center_xyz_m[2], 0.02, atol=0.005)


def test_support_collision_boxes_preserve_an_occluded_arm_hole():
    points, _ = organized_plane(height=32, width=48)
    mask = np.ones((32, 48), dtype=bool)
    mask[8:24, 18:30] = False
    boxes = support_collision_boxes(
        {"mask": mask},
        vertices=points.reshape(-1, 3),
        valid=np.ones(mask.size, dtype=bool),
        shape_hw=mask.shape,
        tile_size_px=4,
        minimum_points=2,
    )
    assert boxes
    for box in boxes:
        left, top, right, bottom = box["source_pixel_bounds_xyxy"]
        assert not (
            left >= 18 and right <= 30 and top >= 8 and bottom <= 24
        )


def test_quality_is_dimensionless_and_rejects_missing_support():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:15, 10:20] = True
    geometry = robust_oriented_geometry(
        np.array(
            [[-0.03, -0.03, 0.01], [0.03, 0.03, 0.05], [0.03, -0.03, 0.03]]
        ),
        catalog=definition(),
        support_height_m=0.0,
    )
    accepted, terms = quality_score(
        sam_score=0.95,
        mask=mask,
        valid_depth=np.ones_like(mask),
        geometry=geometry,
        catalog=definition(),
        support_found=True,
    )
    unsupported, _ = quality_score(
        sam_score=0.95,
        mask=mask,
        valid_depth=np.ones_like(mask),
        geometry=geometry,
        catalog=definition(),
        support_found=False,
    )
    assert accepted > unsupported
    assert accepted > 0.7
    assert terms["valid_depth_fraction"] == 1.0


def test_mask_overlap_uses_confidence_not_argument_order(tmp_path):
    low = np.zeros((12, 12), np.uint8)
    high = np.zeros((12, 12), np.uint8)
    low[2:10, 2:10] = 255
    high[5:11, 5:11] = 255
    low_path = tmp_path / "low.png"
    high_path = tmp_path / "high.png"
    assert cv2.imwrite(str(low_path), low)
    assert cv2.imwrite(str(high_path), high)
    observations = [
        MaskObservation("low", "low", "low", str(low_path), 0.5, "sam", 1),
        MaskObservation("high", "high", "high", str(high_path), 0.9, "sam", 1),
    ]
    owned = {item.instance_id: mask for item, mask in exclusive_masks(observations, low.shape)}
    assert not np.any(owned["low"] & owned["high"])
    assert owned["high"][6, 6]


def test_nested_opaque_false_positive_is_suppressed_but_transparent_pair_is_not(
    tmp_path,
):
    candidate = np.zeros((20, 20), np.uint8)
    owner = np.zeros((20, 20), np.uint8)
    candidate[4:15, 4:15] = 255
    candidate[3, 4:15] = 255
    owner[4:15, 4:15] = 255
    candidate_path = tmp_path / "candidate.png"
    owner_path = tmp_path / "owner.png"
    assert cv2.imwrite(str(candidate_path), candidate)
    assert cv2.imwrite(str(owner_path), owner)
    observations = [
        MaskObservation(
            "arm-fragment",
            "robot",
            "robot",
            str(candidate_path),
            0.65,
            "sam",
            1,
        ),
        MaskObservation(
            "microscope",
            "microscope",
            "microscope",
            str(owner_path),
            0.90,
            "sam",
            1,
        ),
    ]
    opaque = {
        item.instance_id: mask
        for item, mask in exclusive_masks(observations, candidate.shape)
    }
    assert "arm-fragment" not in opaque
    transparent = {
        item.instance_id: mask
        for item, mask in exclusive_masks(
            observations,
            candidate.shape,
            transparent_semantics={"robot", "microscope"},
        )
    }
    assert np.count_nonzero(transparent["arm-fragment"]) == 11


def test_aabb_intersection_gate():
    geometry = {
        "kind": "box",
        "center_xyz_m": [0, 0, 0.05],
        "size_xyz_m": [0.1, 0.1, 0.1],
        "yaw_rad": 0,
    }
    objects = [
        {"instance_id": "a", "geometry": geometry},
        {
            "instance_id": "b",
            "geometry": {**geometry, "center_xyz_m": [0.04, 0, 0.05]},
        },
    ]
    assert aabb_intersections(objects)[0]["a"] == "a"
