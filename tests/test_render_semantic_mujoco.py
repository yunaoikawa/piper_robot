#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.scene_semantics import LABEL_BACKGROUND, LABEL_LID, LABEL_ROBOT
from src.render_semantic_mujoco import classify_faces, compact_mesh


def test_face_semantics_use_majority_and_lid_priority():
    labels = np.array(
        [LABEL_BACKGROUND, LABEL_ROBOT, LABEL_ROBOT, LABEL_LID, LABEL_LID]
    )
    faces = np.array([[0, 1, 2], [0, 3, 4], [1, 3, 4]])
    result = classify_faces(labels, faces)
    assert np.array_equal(
        result, [LABEL_ROBOT, LABEL_LID, LABEL_LID]
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


if __name__ == "__main__":
    test_face_semantics_use_majority_and_lid_priority()
    test_compact_mesh_reindexes_vertices()
    print("semantic MuJoCo rendering checks passed")
