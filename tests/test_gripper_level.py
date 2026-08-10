import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot.arm.home import physical_home_q
from rollout.gripper_level import (
    JawLevelReference,
    RightJawLevelCheckpoint,
    assess_jaw_level,
    leveled_pose,
    signed_outward_tip_pitch_deg,
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


def test_signed_tip_pitch_uses_physical_outward_not_approach_axis():
    reference = JawLevelReference()
    angle = math.radians(10.0)
    approach = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    up = np.array([math.sin(angle), 0.0, math.cos(angle)])
    baseline = np.array([0.0, -1.0, 0.0])
    rotation = Rotation.from_matrix(np.column_stack((approach, up, baseline)))
    xyzw = rotation.as_quat()
    pose = np.r_[xyzw[[3, 0, 1, 2]], np.zeros(3)]
    assert signed_outward_tip_pitch_deg(pose, reference) == pytest.approx(10.0)


def test_calibrated_physical_approach_offset_corrects_attachment_pitch():
    offset_deg = 2.8
    reference = JawLevelReference(
        approach_axis_ee=(
            math.cos(math.radians(offset_deg)),
            math.sin(math.radians(offset_deg)),
            0.0,
        )
    )
    pose_pitch_deg = 5.0
    angle = math.radians(pose_pitch_deg)
    approach = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    up = np.array([math.sin(angle), 0.0, math.cos(angle)])
    baseline = np.array([0.0, -1.0, 0.0])
    rotation = Rotation.from_matrix(np.column_stack((approach, up, baseline)))
    xyzw = rotation.as_quat()
    pose = np.r_[xyzw[[3, 0, 1, 2]], np.zeros(3)]
    assert signed_outward_tip_pitch_deg(pose, reference) == pytest.approx(
        2.2, abs=0.02
    )
    leveled = leveled_pose(pose, reference)
    assessment = assess_jaw_level(leveled, reference)
    assert assessment.accepted
    assert assessment.combined_tilt_deg < 1e-6
    assert abs(signed_outward_tip_pitch_deg(leveled, reference)) < 1e-6


def test_saved_pasteur_attachment_axis_matches_rgbd_consensus():
    level_config = json.loads(
        (ROOT / "src/configs/pasteur_fast_lid_grasp_level.json").read_text()
    )
    calibration = json.loads(
        (ROOT / "src/configs/pasteur_right_physical_gripper_level.json").read_text()
    )
    reference = JawLevelReference(
        support_up_robot=level_config["support_up_robot"],
        tip_baseline_ee=level_config["tip_baseline_ee"],
        approach_axis_ee=level_config["approach_axis_ee"],
        open_tip_span_m=level_config["open_tip_span_m"],
        maximum_checkpoint_tilt_deg=level_config[
            "maximum_checkpoint_tilt_deg"
        ],
        maximum_planned_tilt_deg=level_config["maximum_planned_tilt_deg"],
        maximum_tip_height_difference_m=level_config[
            "maximum_tip_height_difference_m"
        ],
    )
    pose = calibration["calibrated_pose_wxyz_xyz_audit_only"]
    physical_pitch = signed_outward_tip_pitch_deg(pose, reference)
    assert physical_pitch == pytest.approx(
        calibration["level_consensus"]["median_angle_deg"], abs=1e-6
    )
    assert assess_jaw_level(pose, reference).accepted


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
