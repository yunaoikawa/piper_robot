"""Offline unit tests; no RPC connection or hardware commands."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "docs/evaluate_door_approach_distance.py"
spec = importlib.util.spec_from_file_location("door_distance_evaluation", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_distance_is_mm_not_image_error():
    result = module.pose_difference([1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, .003, .004, 0])
    assert result["distance_mm"] == pytest.approx(5)
    assert result["delta_xyz_mm"] == pytest.approx([3, 4, 0])
    assert result["orientation_difference_deg"] == pytest.approx(0)


def test_rotation_is_separate_from_distance():
    result = module.pose_difference([1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0])
    assert result["distance_mm"] == 0
    assert result["orientation_difference_deg"] == pytest.approx(180)


def test_quaternion_sign_does_not_change_pose():
    assert module.pose_difference([1, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0])["orientation_difference_deg"] == pytest.approx(0)


@pytest.mark.parametrize("pose", [[1, 0, 0], [1, 0, 0, 0, np.nan, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
def test_invalid_pose_rejected(pose):
    with pytest.raises(ValueError):
        module.pose_difference(pose, [1, 0, 0, 0, 0, 0, 0])
