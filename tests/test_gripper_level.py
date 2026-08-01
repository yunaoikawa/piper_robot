from pathlib import Path

import numpy as np

from robot.arm.home import physical_home_q
from rollout.gripper_level import (
    JawLevelReference,
    RightJawLevelCheckpoint,
    assess_jaw_level,
    leveled_pose,
)
from rollout.teleop_trajectory_stream import ProductionRightFK


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "robot/cone-e-description/robot-welded-base-and-lift.mjcf"
SUCCESS_PRECLOSE_Q = np.asarray(
    [-0.2213601023, 0.3046472073, -0.0692023039, -0.5872684121, 0.0, 2.9294652939]
)
BAD_LOW_Q = np.asarray(
    [0.4228234589, 1.0378774405, -0.1852143407, 0.8782147765, -0.9789202809, 1.9821355343]
)


def test_saved_success_passes_and_bad_low_pose_fails():
    fk = ProductionRightFK(MODEL)
    reference = JawLevelReference()
    home = assess_jaw_level(fk.pose(physical_home_q("right")).parameters(), reference)
    success = assess_jaw_level(fk.pose(SUCCESS_PRECLOSE_Q).parameters(), reference)
    bad = assess_jaw_level(fk.pose(BAD_LOW_Q).parameters(), reference)
    assert home.accepted
    assert success.accepted
    assert success.tip_height_difference_m < 0.0017
    assert not bad.accepted
    assert bad.tip_height_difference_m > 0.025


def test_leveled_pose_preserves_position_and_removes_tip_height_difference():
    fk = ProductionRightFK(MODEL)
    reference = JawLevelReference()
    source = np.asarray(fk.pose(BAD_LOW_Q).parameters())
    result = leveled_pose(source, reference)
    assessment = assess_jaw_level(result, reference, planned=True)
    assert assessment.accepted
    assert assessment.combined_tilt_deg < 1e-6
    assert assessment.tip_height_difference_m < 1e-9
    assert np.allclose(result[4:], source[4:])


class _PoseRPC:
    def __init__(self, pose):
        self.pose = pose
        self.calls = 0

    def get_right_ee_pose(self):
        self.calls += 1
        return self.pose


def test_checkpoint_reads_exactly_one_pose_per_named_gate():
    fk = ProductionRightFK(MODEL)
    rpc = _PoseRPC(fk.pose(SUCCESS_PRECLOSE_Q))
    checkpoint = RightJawLevelCheckpoint(rpc, JawLevelReference())
    assessment = checkpoint.require("before_close")
    assert assessment.accepted
    assert rpc.calls == 1
    assert checkpoint.records[0]["checkpoint"] == "before_close"
