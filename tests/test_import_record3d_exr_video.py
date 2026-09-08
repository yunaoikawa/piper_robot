#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.import_record3d_exr_video import (
    path_fraction_targets,
    select_view_centers,
)


def _poses(count: int) -> list[list[float]]:
    return [
        [0.0, 0.0, 0.0, 1.0, 0.002 * index, 0.0, 0.0]
        for index in range(count)
    ]


def test_targets_are_spaced_by_camera_path_distance():
    poses = _poses(101)
    assert path_fraction_targets(poses, 3) == [5, 50, 95]


def test_selection_avoids_blurry_targets_and_keeps_stable_bursts():
    with tempfile.TemporaryDirectory() as temporary:
        rgb = Path(temporary)
        poses = _poses(101)
        targets = path_fraction_targets(poses, 3)
        checker = (
            (np.indices((96, 128)).sum(axis=0) % 2) * 255
        ).astype(np.uint8)
        sharp = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)
        blurry = np.full((96, 128, 3), 127, dtype=np.uint8)
        for index in range(len(poses)):
            image = blurry if index in targets else sharp
            assert cv2.imwrite(str(rgb / f"{index}.jpg"), image)
        selected = select_view_centers(
            poses,
            rgb,
            view_count=3,
            frames_per_view=3,
            candidate_radius=4,
        )
        assert len(selected) == 3
        assert all(item["pose_stability"]["accepted"] for item in selected)
        assert all(item["center_index"] not in targets for item in selected)
        assert all(len(item["frame_indices"]) == 3 for item in selected)


if __name__ == "__main__":
    test_targets_are_spaced_by_camera_path_distance()
    test_selection_avoids_blurry_targets_and_keeps_stable_bursts()
    print("Record3D EXR importer checks passed")
