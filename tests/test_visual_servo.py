#!/usr/bin/env python3
"""Hardware-free checks for demo-relative visual servoing."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.arm.ik_continuity import joint_target_is_continuous
from rollout.visual_servo import (
    DemoRelativeServo,
    FineObservation,
    ManipulationTemplate,
    MeasuredImageJacobian,
    ObjectEstimate,
    ServoAction,
    ServoConfig,
    ServoPhase,
    StampedObservation,
    trust_region_step,
)


class FakeAdapter:
    def __init__(self):
        self.position = np.array([0.10, 0.20, 0.0])
        self.feature = np.array([50.0, 60.0])
        self.success = False

    def detect_object(self, observation):
        return ObjectEstimate(self.position, observation.timestamp, 0.95, source="fake")

    def fine_observation(self, observation):
        return FineObservation(self.feature, 0.95)

    def check_success(self, observation):
        return self.success, {"fake": True}


def observation(pose, ratio=1.0, timestamp=None):
    return StampedObservation(
        timestamp=time.time() if timestamp is None else timestamp,
        ee_pose=np.asarray(pose, float),
        joint_positions=np.zeros(6),
        gripper_ratio=ratio,
    )


def test_template_tracks_only_selected_axes():
    template = ManipulationTemplate(
        reference_object_position_m=[0.10, 0.20, 0.0],
        goal_ee_pose=[1, 0, 0, 0, 0.30, 0.40, 0.80],
        tracked_translation_axes=[True, True, False],
    )
    estimate = ObjectEstimate([0.12, 0.17, 0.30], time.time(), 1.0)
    target = template.target_pose(estimate)
    assert np.allclose(target[4:7], [0.32, 0.37, 0.80])


def test_measured_jacobian_uses_actual_motion():
    estimator = MeasuredImageJacobian()
    assert not estimator.update([0.002, 0.0], [4.0, 1.0])
    assert estimator.update([0.0, 0.003], [0.0, 6.0])
    assert estimator.ready
    assert np.allclose(estimator.matrix, [[2000, 0], [500, 2000]], atol=1e-6)
    step = estimator.solve([4.0, 4.0], 0.01)
    assert np.allclose(estimator.matrix @ step, [4.0, 4.0], atol=1e-5)
    assert not estimator.update([0.001, 0], [50, 50], object_moved=True)
    assert not estimator.ready


def test_state_machine_stops_before_contact_and_recovers():
    adapter = FakeAdapter()
    template = ManipulationTemplate(
        reference_object_position_m=adapter.position,
        goal_ee_pose=[1, 0, 0, 0, 0.30, 0.40, 0.80],
        pregrasp_offset_m=[0, 0, 0.01],
        fine_feature_goal=None,
    )
    servo = DemoRelativeServo(adapter, template)
    far = observation([1, 0, 0, 0, 0.0, 0.0, 0.9])
    decision = servo.step(far)
    assert decision.action == ServoAction.MOVE
    assert np.linalg.norm(decision.target_pose[4:7] - far.ee_pose[4:7]) <= 0.030001

    pregrasp = observation([1, 0, 0, 0, 0.30, 0.40, 0.81])
    decision = servo.step(pregrasp)
    assert decision.phase == ServoPhase.CONTACT_CONFIRMATION
    assert decision.action == ServoAction.HOLD

    servo.confirm_contact()
    decision = servo.step(pregrasp)
    assert decision.action == ServoAction.MOVE
    contact = observation([1, 0, 0, 0, 0.30, 0.40, 0.80])
    assert servo.step(contact).action == ServoAction.CLOSE
    failed = observation([1, 0, 0, 0, 0.30, 0.40, 0.80], ratio=0.0)
    assert servo.step(failed).action == ServoAction.OPEN


def test_stale_observation_and_joint_continuity():
    adapter = FakeAdapter()
    template = ManipulationTemplate(adapter.position, [1, 0, 0, 0, 0, 0, 0])
    servo = DemoRelativeServo(adapter, template, ServoConfig(max_observation_age_s=0.1))
    decision = servo.step(observation([1, 0, 0, 0, 0, 0, 0], timestamp=time.time() - 1))
    assert decision.action == ServoAction.HOLD
    assert "stale" in decision.reason

    ok, _ = joint_target_is_continuous(np.zeros(6), np.ones(6) * 0.2)
    bad, delta = joint_target_is_continuous(np.zeros(6), [0, 0, 0, 0, 0, 0.8])
    assert ok and not bad and np.isclose(delta[-1], 0.8)
    assert trust_region_step(0.04) == 0.03
    assert trust_region_step(0.02) == 0.01
    assert trust_region_step(0.005) == 0.005


if __name__ == "__main__":
    test_template_tracks_only_selected_axes()
    test_measured_jacobian_uses_actual_motion()
    test_state_machine_stops_before_contact_and_recovers()
    test_stale_observation_and_joint_continuity()
    print("visual servo checks passed")
