import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.save_jaw_side_view_reference import _pose_difference


def _pose(rotation_deg=0.0, translation=(0.0, 0.0, 0.0)):
    xyzw = Rotation.from_euler("z", rotation_deg, degrees=True).as_quat()
    return np.r_[xyzw[[3, 0, 1, 2]], translation]


def test_physical_level_consensus_pose_match_is_sign_invariant():
    pose = _pose(12.0, (0.1, 0.2, 0.3))
    opposite_quaternion = pose.copy()
    opposite_quaternion[:4] *= -1.0
    translation_m, rotation_deg = _pose_difference(pose, opposite_quaternion)
    assert translation_m == pytest.approx(0.0)
    assert rotation_deg == pytest.approx(0.0)


def test_physical_level_consensus_pose_match_reports_motion():
    translation_m, rotation_deg = _pose_difference(
        _pose(), _pose(2.0, (0.003, 0.0, 0.0))
    )
    assert translation_m == pytest.approx(0.003)
    assert rotation_deg == pytest.approx(2.0)
