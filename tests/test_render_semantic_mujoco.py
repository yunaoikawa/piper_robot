#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.scene_semantics import LABEL_BACKGROUND, LABEL_LID, LABEL_ROBOT
from src.render_semantic_mujoco import (
    classify_faces,
    compact_mesh,
    compute_rough_alignment,
    validate_transform,
)


def test_face_semantics_use_majority_and_robot_foreground_priority():
    labels = np.array(
        [LABEL_BACKGROUND, LABEL_ROBOT, LABEL_ROBOT, LABEL_LID, LABEL_LID]
    )
    faces = np.array(
        [[0, 1, 2], [0, 3, 4], [1, 3, 4], [1, 3, 2]]
    )
    result = classify_faces(labels, faces)
    assert np.array_equal(
        result, [LABEL_ROBOT, LABEL_LID, LABEL_LID, LABEL_ROBOT]
    )


def test_compact_mesh_reindexes_vertices():
    vertices = np.arange(18).reshape(6, 3)
    faces = np.array([[5, 2, 4], [2, 1, 4]])
    compact_vertices, compact_faces = compact_mesh(vertices, faces)
    assert len(compact_vertices) == 4
    assert np.max(compact_faces) == 3
    assert np.array_equal(
        compact_vertices[compact_faces[0]], vertices[faces[0]]
    )


def test_only_rigid_registration_transforms_are_accepted():
    transform = np.eye(4)
    transform[:3, 3] = [0.1, -0.2, 0.3]
    assert np.array_equal(
        validate_transform(transform, name="test"), transform
    )
    invalid = transform.copy()
    invalid[0, 0] = 2.0
    try:
        validate_transform(invalid, name="test")
    except ValueError:
        pass
    else:
        raise AssertionError("non-rigid transform should be rejected")


def test_display_only_fixture_alignment_maps_anchor_and_heading():
    payload = {
        "display_only": True,
        "anchor": {
            "source_xyz_m": [0.397, 0.751, 0.0],
            "target_xyz_m": [0.35, 0.0, 0.0],
        },
        "source_heading_xy": [0.0, 1.0],
        "target_heading_xy": [1.0, 0.0],
    }
    transform = compute_rough_alignment(payload)
    mapped_anchor = transform @ np.array([0.397, 0.751, 0.0, 1.0])
    mapped_heading = transform[:3, :3] @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(mapped_anchor[:3], [0.35, 0.0, 0.0])
    assert np.allclose(mapped_heading, [1.0, 0.0, 0.0])
    assert np.allclose(transform[2], [0.0, 0.0, 1.0, 0.0])


def test_rough_alignment_requires_explicit_display_only_acknowledgement():
    payload = {
        "anchor": {
            "source_xyz_m": [0.0, 0.0, 0.0],
            "target_xyz_m": [0.0, 0.0, 0.0],
        },
        "source_heading_xy": [1.0, 0.0],
        "target_heading_xy": [1.0, 0.0],
    }
    try:
        compute_rough_alignment(payload)
    except ValueError as error:
        assert "display_only=true" in str(error)
    else:
        raise AssertionError("rough alignment must remain explicitly display-only")


if __name__ == "__main__":
    test_face_semantics_use_majority_and_robot_foreground_priority()
    test_compact_mesh_reindexes_vertices()
    test_only_rigid_registration_transforms_are_accepted()
    test_display_only_fixture_alignment_maps_anchor_and_heading()
    test_rough_alignment_requires_explicit_display_only_acknowledgement()
    print("semantic MuJoCo rendering checks passed")
