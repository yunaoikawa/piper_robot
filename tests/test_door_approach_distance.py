"""Offline unit tests; no RPC connection or hardware commands."""
import importlib.util
import json
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


def test_tracked_four_stage_snapshot_preserves_continuation():
    report_path = path.parent / "assets/code_as_learning_machine/door_approach_distance_report.json"
    report = json.loads(report_path.read_text())
    rows = report["configurations"]
    assert [r["stage"] for r in rows] == ["D1", "D2", "D3", "D4"]
    assert rows[2]["historical_stage"] == "D4_pre_parser_fix"
    assert rows[3]["historical_stage"] == "D4"
    assert rows[2]["distance_mm"] == rows[3]["distance_mm"]
    assert rows[2]["actual_ee_pose_wxyz_xyz"] == rows[3]["stage_before_ee_pose_wxyz_xyz"]
    assert report["continuation_audit"]["identical_recorded_after_pose"]
    assert report["fully_registered_3d_distance_available"] is False
