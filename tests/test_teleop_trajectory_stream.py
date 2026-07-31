from __future__ import annotations

from pathlib import Path

import numpy as np

from robot.arm.home import physical_home_q
from rollout.teleop_trajectory_stream import (
    COMMAND_PREVIEW_S,
    CONTROL_HZ,
    JointTrajectorySample,
    ProductionRightFK,
    TeleopTrajectoryStreamer,
    TrajectoryStreamError,
    sample_joint_knots,
)


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_MODEL = (
    ROOT / "robot/cone-e-description/robot-welded-base-and-lift.mjcf"
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, duration):
        self.now += max(0.0, float(duration))


class FakeRPC:
    def __init__(self, fk, *, torque=None):
        self.fk = fk
        self.q = physical_home_q("right")
        self.torque = (
            np.zeros(6) if torque is None else np.asarray(torque, dtype=float)
        )
        self.gripper = 1.0
        self.ee_commands = []
        self.joint_commands = []
        self.gains = []

    def get_right_joint_positions(self):
        return self.q.copy()

    def get_right_ee_pose(self):
        return self.fk.pose(self.q)

    def get_right_joint_torque(self):
        return self.torque.copy()

    def get_right_gripper_exact(self):
        return self.gripper

    def set_right_joint_target(
        self, target, *, gripper_target, preview_time
    ):
        self.joint_commands.append(
            (np.asarray(target).copy(), gripper_target, preview_time)
        )
        return True

    def set_right_gain(self, kp, kd):
        self.gains.append((np.asarray(kp).copy(), np.asarray(kd).copy()))

    def set_right_ee_target(
        self, ee_target, *, gripper_target, preview_time
    ):
        self.ee_commands.append(
            (
                np.asarray(ee_target.parameters()).copy(),
                gripper_target,
                preview_time,
            )
        )
        if gripper_target is not None:
            self.gripper = float(gripper_target)
        return True


class AlternateIKFakeRPC(FakeRPC):
    """Report a different joint branch while following the commanded EE."""

    def __init__(self, fk):
        super().__init__(fk)
        self.measured_ee = fk.pose(self.q)
        self.command_count = 0

    def get_right_joint_positions(self):
        if self.command_count:
            return self.q + np.asarray([0.0, 0.0, 0.0, 0.8, 0.0, -0.8])
        return self.q.copy()

    def get_right_ee_pose(self):
        return self.measured_ee

    def set_right_ee_target(
        self, ee_target, *, gripper_target, preview_time
    ):
        accepted = super().set_right_ee_target(
            ee_target,
            gripper_target=gripper_target,
            preview_time=preview_time,
        )
        self.measured_ee = ee_target
        self.command_count += 1
        return accepted


def _streamer(rpc, fk, clock, *, enforce=False):
    return TeleopTrajectoryStreamer(
        rpc,
        fk,
        torque_limit_nm=np.ones(6),
        consecutive_torque_samples=2,
        enforce_torque_stop=enforce,
        gain_ramp_s=0.0,
        mode_settle_s=0.0,
        hold_settle_s=0.0,
        tracking_check_interval=1000,
        mode_refresher=lambda: None,
        clock=clock,
        sleep=clock.sleep,
    )


def test_production_right_fk_matches_crossed_cone_e_branch():
    fk = ProductionRightFK(PRODUCTION_MODEL)
    q = physical_home_q("right")
    pose = fk.pose(q)
    report = fk.validate_measured(q, pose)
    assert report["accepted"]
    assert report["physical_right_model_branch"] == "left_arm_*"
    assert report["position_error_m"] == 0.0


def test_joint_knots_are_sampled_at_teleop_rate_without_demo_width():
    home = physical_home_q("right")
    knots = [
        {
            "stage": "home",
            "right_q_physical_rad": home.tolist(),
            "right_gripper_open_ratio": 1.0,
            "minimum_duration_s": 0.5,
        },
        {
            "stage": "insert",
            "right_q_physical_rad": (home + 0.01).tolist(),
            "right_gripper_open_ratio": 1.0,
            "minimum_duration_s": 0.1,
        },
        {
            "stage": "close",
            "right_q_physical_rad": (home + 0.01).tolist(),
            "right_gripper_open_ratio": 0.0,
            "minimum_duration_s": 0.1,
        },
    ]
    samples = sample_joint_knots(knots)
    assert len(samples) == 6
    assert samples[-1].stage == "close"
    assert samples[-1].right_gripper_open_ratio == 0.0
    assert np.isclose(samples[-1].t_s, 0.2)


def test_streamer_uses_uninterrupted_teleop_ee_targets():
    fk = ProductionRightFK(PRODUCTION_MODEL)
    rpc = FakeRPC(fk)
    clock = FakeClock()
    home = physical_home_q("right")
    samples = [
        JointTrajectorySample(
            t_s=(index + 1) / CONTROL_HZ,
            stage="close" if index == 2 else "insert",
            right_q_physical_rad=home.copy(),
            right_gripper_open_ratio=0.0 if index == 2 else 1.0,
        )
        for index in range(3)
    ]
    gated = []
    report = _streamer(rpc, fk, clock).execute(
        samples, stage_gate=gated.append
    )
    assert len(rpc.ee_commands) == 3
    assert all(
        np.isclose(command[2], COMMAND_PREVIEW_S)
        for command in rpc.ee_commands
    )
    assert report["command_path"] == "set_right_ee_target"
    assert report["stages"] == ["insert", "close"]
    assert report["gated_stages"] == ["insert", "close"]
    assert gated == ["insert", "close"]
    assert report["final_right_gripper_open_ratio"] == 0.0
    assert rpc.joint_commands
    assert len(rpc.gains) >= 2


def test_cartesian_tracking_accepts_a_different_joint_branch():
    fk = ProductionRightFK(PRODUCTION_MODEL)
    rpc = AlternateIKFakeRPC(fk)
    clock = FakeClock()
    sample = JointTrajectorySample(
        t_s=1.0 / CONTROL_HZ,
        stage="hover_xy",
        right_q_physical_rad=physical_home_q("right"),
        right_gripper_open_ratio=1.0,
    )
    streamer = TeleopTrajectoryStreamer(
        rpc,
        fk,
        torque_limit_nm=np.ones(6),
        consecutive_torque_samples=2,
        enforce_torque_stop=False,
        gain_ramp_s=0.0,
        mode_settle_s=0.0,
        hold_settle_s=0.0,
        tracking_check_interval=1,
        mode_refresher=lambda: None,
        clock=clock,
        sleep=clock.sleep,
    )
    report = streamer.execute([sample])
    assert report["maximum_tracking_joint_error_rad"] > 0.45
    assert report["maximum_tracking_position_error_m"] == 0.0
    assert report["maximum_tracking_rotation_error_rad"] == 0.0


def test_observer_only_torque_does_not_shorten_autonomous_stream():
    fk = ProductionRightFK(PRODUCTION_MODEL)
    rpc = FakeRPC(fk, torque=np.full(6, 2.0))
    clock = FakeClock()
    sample = JointTrajectorySample(
        t_s=1.0 / CONTROL_HZ,
        stage="depart_up",
        right_q_physical_rad=physical_home_q("right"),
        right_gripper_open_ratio=1.0,
    )
    report = _streamer(rpc, fk, clock, enforce=False).execute([sample])
    assert len(rpc.ee_commands) == 1
    assert report["torque_stop_enforced"] is False
    assert report["torque_warning_count"] > 0


def test_enforced_torque_latches_measured_pose():
    fk = ProductionRightFK(PRODUCTION_MODEL)
    rpc = FakeRPC(fk, torque=np.full(6, 2.0))
    clock = FakeClock()
    sample = JointTrajectorySample(
        t_s=1.0 / CONTROL_HZ,
        stage="depart_up",
        right_q_physical_rad=physical_home_q("right"),
        right_gripper_open_ratio=1.0,
    )
    try:
        _streamer(rpc, fk, clock, enforce=True).execute([sample])
    except TrajectoryStreamError as error:
        assert "torque stop" in str(error)
    else:
        raise AssertionError("enforced torque did not stop the stream")
    assert rpc.joint_commands
