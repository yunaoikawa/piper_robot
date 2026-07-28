#!/usr/bin/env python3

import socket
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_realtime_sam_grasp import (
    DEFAULT_HOLDING_KD,
    DEFAULT_HOLDING_KP,
    DEFAULT_MOTION_KD,
    DEFAULT_MOTION_KP,
    LiveSamGrasp,
    PIPER_MIT_MODE_CAN_ID,
    PIPER_MIT_MODE_PAYLOAD,
    TorqueStop,
    refresh_right_mit_mode,
)
from rollout.realtime_sam_servo import (
    bounded_reachable_servo_step,
    bounded_servo_step,
    estimate_feature_jacobian,
    estimate_reachable_feature_model,
    gripper_tip_px,
    scene_feature,
)
from rollout.sam_segmentation import MaskCandidate


def circle_mask(center, radius, shape=(240, 320)):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2


lid_mask = circle_mask((220, 150), 28)
gripper_mask = np.zeros((240, 320), bool)
gripper_mask[130:170, 120:190] = True
left_gripper = np.zeros((240, 320), bool)
left_gripper[40:80, 20:90] = True

lid = MaskCandidate(lid_mask, np.array([192, 122, 248, 178]), 0.95)
right = MaskCandidate(gripper_mask, np.array([120, 130, 190, 170]), 0.85)
left = MaskCandidate(left_gripper, np.array([20, 40, 90, 80]), 0.99)
depth = np.ones((240, 320), float)
depth[lid_mask] = 1.10
depth[gripper_mask] = 0.95

feature = scene_feature(
    lid_candidates=[lid],
    gripper_candidates=[left, right],
    depth_m=depth,
    clearance_m=0.04,
)
assert np.allclose(feature.lid_grasp_feature[2], 1060)
assert np.allclose(feature.gripper_feature[2], 950)
assert gripper_tip_px(right)[0] >= 188
assert feature.lid_grasp_feature[0] < 200

true_jacobian = np.array(
    [[500, 20, -80], [-30, 420, 100], [100, -50, 900]], dtype=float
)
robot_deltas = np.eye(3) * 0.005
feature_deltas = robot_deltas @ true_jacobian.T
estimated = estimate_feature_jacobian(robot_deltas, feature_deltas)
assert np.allclose(estimated, true_jacobian)

error = np.array([40, -30, 50], dtype=float)
step = bounded_servo_step(estimated, error)
assert np.linalg.norm(step) <= 0.012001
assert np.max(np.abs(step)) <= 0.008001

# A near-singular arm can produce only two independent Cartesian directions.
# The controller must stay inside those measured directions, not invent the
# missing third one.
singular_robot = np.array(
    [
        [0.00445, 0.00022, 0.00025],
        [-0.00004, 0.00285, -0.00285],
        [0.00005, -0.00233, 0.00233],
    ]
)
singular_feature = np.array(
    [[2.0, -2.0, 0.0], [-4.0, -4.0, -10.25], [1.0, 1.0, 12.7]]
)
reachable = estimate_reachable_feature_model(
    singular_robot, singular_feature
)
assert reachable.rank == 2
reachable_step = bounded_reachable_servo_step(
    reachable,
    np.array([-8.0, 105.0, 78.0]),
    tolerances=np.array([6.0, 6.0, 8.0]),
)
projected = reachable.basis_xyz @ (
    reachable.basis_xyz.T @ reachable_step
)
assert np.allclose(reachable_step, projected)
assert 0.001 < np.linalg.norm(reachable_step) <= 0.012001
assert np.max(np.abs(reachable_step)) <= 0.008001


class FakeMotionRPC:
    def __init__(
        self,
        *,
        reject_at=None,
        torque=None,
        torque_sequence=None,
        command_delay_s=0.0,
    ):
        self.pose = np.array([1.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.8])
        self.commands = []
        self.holds = []
        self.gains = []
        self.events = []
        self.reject_at = reject_at
        self.command_delay_s = float(command_delay_s)
        self.torque = (
            np.zeros(6) if torque is None else np.asarray(torque, dtype=float)
        )
        self.torque_sequence = (
            []
            if torque_sequence is None
            else [
                np.asarray(sample, dtype=float)
                for sample in torque_sequence
            ]
        )

    def get_right_ee_pose(self):
        return mink.SE3(self.pose.copy())

    def get_right_joint_torque(self):
        if self.torque_sequence:
            return self.torque_sequence.pop(0).copy()
        return self.torque.copy()

    def set_right_ee_target(
        self, ee_target, *, gripper_target, preview_time
    ):
        if self.command_delay_s > 0.0:
            time.sleep(self.command_delay_s)
        parameters = np.asarray(ee_target.parameters(), dtype=float)
        self.commands.append(
            (parameters.copy(), gripper_target, float(preview_time))
        )
        self.events.append("move")
        if self.reject_at == len(self.commands):
            return False
        self.pose = parameters
        return True

    def get_right_joint_positions(self):
        return np.zeros(6)

    def set_right_joint_target(
        self, target, *, gripper_target, preview_time
    ):
        self.holds.append(
            (
                np.asarray(target, dtype=float).copy(),
                gripper_target,
                float(preview_time),
            )
        )
        self.events.append("hold")
        return True

    def set_right_gain(self, kp, kd):
        self.gains.append(
            (
                np.asarray(kp, dtype=float).copy(),
                np.asarray(kd, dtype=float).copy(),
            )
        )
        self.events.append("gain")
        return True


class LegacyNoneMotionRPC(FakeMotionRPC):
    def set_right_ee_target(self, *args, **kwargs):
        super().set_right_ee_target(*args, **kwargs)
        return None


class LateralOnlyMotionRPC(FakeMotionRPC):
    def set_right_ee_target(
        self, ee_target, *, gripper_target, preview_time
    ):
        parameters = np.asarray(ee_target.parameters(), dtype=float)
        self.commands.append(
            (parameters.copy(), gripper_target, float(preview_time))
        )
        self.pose[4] = 0.3 + (parameters[6] - 0.8)
        return True


class InvalidJointMotionRPC(FakeMotionRPC):
    def get_right_joint_positions(self):
        return np.full(6, np.nan)


def fake_motion_runner(rpc, *, torque_samples=5):
    runner = object.__new__(LiveSamGrasp)
    runner.args = SimpleNamespace(preview_time=0.06, minimum_progress=0.9)
    runner.rpc = rpc
    runner.torque_limit = np.ones(6)
    runner.torque_samples = torque_samples
    runner.joint_command = None
    runner.holding_kp = DEFAULT_HOLDING_KP.copy()
    runner.holding_kd = DEFAULT_HOLDING_KD.copy()
    runner.motion_kp = DEFAULT_MOTION_KP.copy()
    runner.motion_kd = DEFAULT_MOTION_KD.copy()
    runner.gain_ramp_s = 0.0
    runner.mode_settle_s = 0.0
    runner.hold_settle_s = 0.0
    runner.motion_mode_refresher = lambda: rpc.events.append("mode")
    return runner


class FakeCanSocket:
    def __init__(self, sent_length=None):
        self.bound = None
        self.frame = None
        self.closed = False
        self.sent_length = sent_length

    def bind(self, address):
        self.bound = address

    def send(self, frame):
        self.frame = bytes(frame)
        if self.sent_length is None:
            return len(frame)
        return self.sent_length

    def close(self):
        self.closed = True


can_socket = FakeCanSocket()
factory_args = []


def fake_socket_factory(*args):
    factory_args.append(args)
    return can_socket


refresh_right_mit_mode("can_right", socket_factory=fake_socket_factory)
assert factory_args == [(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)]
assert can_socket.bound == ("can_right",)
assert can_socket.closed
can_id, dlc, payload = struct.unpack("=IB3x8s", can_socket.frame)
assert can_id == PIPER_MIT_MODE_CAN_ID
assert dlc == 8
assert payload == PIPER_MIT_MODE_PAYLOAD

try:
    refresh_right_mit_mode("can_left", socket_factory=fake_socket_factory)
    raise AssertionError("unsafe CAN interface name was accepted")
except ValueError:
    pass

short_socket = FakeCanSocket(sent_length=8)
try:
    refresh_right_mit_mode(
        "can_right", socket_factory=lambda *_: short_socket
    )
    raise AssertionError("short CAN write was accepted")
except RuntimeError as exc:
    assert "short" in str(exc)
assert short_socket.closed


motion_rpc = FakeMotionRPC()
motion_runner = fake_motion_runner(motion_rpc)
actual = motion_runner.move_cartesian_delta(
    [0.0, 0.0, 0.004], preview_time=0.06
)
assert np.allclose(actual, [0.0, 0.0, 0.004])
assert len(motion_rpc.commands) == 2
assert motion_rpc.commands[0][0][6] < motion_rpc.commands[1][0][6]
assert np.isclose(motion_rpc.commands[-1][0][6], 0.804)
assert all(command[1] is None for command in motion_rpc.commands)
assert all(np.isclose(command[2], 0.05) for command in motion_rpc.commands)
assert len(motion_rpc.holds) == 2
assert all(hold[1] is None for hold in motion_rpc.holds)
first_move = motion_rpc.events.index("move")
assert motion_rpc.events[:first_move].count("mode") == 2
assert motion_rpc.events[-2:] == ["hold", "gain"]
assert np.allclose(motion_rpc.gains[-1][0], DEFAULT_HOLDING_KP)
assert np.allclose(motion_rpc.gains[-1][1], DEFAULT_HOLDING_KD)

rejecting_rpc = FakeMotionRPC(reject_at=2)
try:
    fake_motion_runner(rejecting_rpc).move_cartesian_delta(
        [0.0, 0.0, 0.004], preview_time=0.06
    )
    raise AssertionError("rejected streamed setpoint did not stop the move")
except RuntimeError as exc:
    assert "rejected" in str(exc)
assert rejecting_rpc.holds
assert rejecting_rpc.holds[-1][1] is None

legacy_rpc = LegacyNoneMotionRPC()
try:
    fake_motion_runner(legacy_rpc).move_cartesian_delta(
        [0.0, 0.0, 0.004], preview_time=0.06
    )
    raise AssertionError("legacy None setter result was treated as accepted")
except RuntimeError as exc:
    assert "rejected" in str(exc)
assert len(legacy_rpc.commands) == 1
assert legacy_rpc.holds

torque_rpc = FakeMotionRPC(torque=np.full(6, 2.0))
try:
    fake_motion_runner(
        torque_rpc, torque_samples=1
    ).move_cartesian_delta([0.0, 0.0, 0.004], preview_time=0.06)
    raise AssertionError("high torque did not stop the streamed move")
except TorqueStop:
    pass
assert not torque_rpc.commands
assert torque_rpc.holds

invalid_torque_rpc = FakeMotionRPC(torque=np.full(6, np.nan))
try:
    fake_motion_runner(invalid_torque_rpc).move_cartesian_delta(
        [0.0, 0.0, 0.004], preview_time=0.06
    )
    raise AssertionError("non-finite torque sample did not stop the move")
except TorqueStop:
    pass
assert not invalid_torque_rpc.commands
assert invalid_torque_rpc.holds

zero_torque = np.zeros(6)
high_torque = np.full(6, 2.0)
late_torque_rpc = FakeMotionRPC(
    torque_sequence=[
        zero_torque,
        zero_torque,
        high_torque,
        high_torque,
        high_torque,
        zero_torque,
    ]
)
try:
    fake_motion_runner(
        late_torque_rpc, torque_samples=3
    ).move_cartesian_delta([0.0, 0.0, 0.004], preview_time=0.06)
    raise AssertionError("settle monitoring lost consecutive torque strikes")
except TorqueStop:
    pass
assert len(late_torque_rpc.commands) == 2
assert late_torque_rpc.holds

slow_rpc = FakeMotionRPC(command_delay_s=0.08)
try:
    fake_motion_runner(slow_rpc).move_cartesian_delta(
        [0.0, 0.0, 0.004], preview_time=0.06
    )
    raise AssertionError("late streamed setpoints were allowed to burst")
except RuntimeError as exc:
    assert "deadline" in str(exc)
assert len(slow_rpc.commands) == 1
assert slow_rpc.holds

lateral_rpc = LateralOnlyMotionRPC()
try:
    fake_motion_runner(lateral_rpc).move_cartesian_delta(
        [0.0, 0.0, 0.004], preview_time=0.06
    )
    raise AssertionError("lateral drift was counted as requested progress")
except RuntimeError as exc:
    assert "did not follow" in str(exc)
assert lateral_rpc.holds

mode_failure_rpc = FakeMotionRPC()
mode_failure_runner = fake_motion_runner(mode_failure_rpc)


def fail_mode_refresh():
    mode_failure_rpc.events.append("mode")
    raise RuntimeError("mode refresh failed")


mode_failure_runner.motion_mode_refresher = fail_mode_refresh
try:
    mode_failure_runner.move_cartesian_delta(
        [0.0, 0.0, 0.004], preview_time=0.06
    )
    raise AssertionError("mode refresh failure did not stop the move")
except RuntimeError as exc:
    assert "mode refresh failed" in str(exc)
assert not mode_failure_rpc.commands
assert mode_failure_rpc.events[-2:] == ["hold", "gain"]
assert np.allclose(mode_failure_rpc.gains[-1][0], DEFAULT_HOLDING_KP)
assert np.allclose(mode_failure_rpc.gains[-1][1], DEFAULT_HOLDING_KD)

invalid_joint_rpc = InvalidJointMotionRPC()
try:
    fake_motion_runner(invalid_joint_rpc).move_cartesian_delta(
        [0.0, 0.0, 0.004], preview_time=0.06
    )
    raise AssertionError("invalid measured joints reached a move command")
except RuntimeError as exc:
    assert "invalid measured right-arm state" in str(exc)
assert not invalid_joint_rpc.commands
assert invalid_joint_rpc.gains
assert np.allclose(invalid_joint_rpc.gains[-1][0], DEFAULT_HOLDING_KP)

invalid_rpc = FakeMotionRPC()
try:
    fake_motion_runner(invalid_rpc).move_cartesian_delta(
        [0.0, np.nan, 0.004], preview_time=0.06
    )
    raise AssertionError("non-finite Cartesian delta was accepted")
except ValueError:
    pass
assert not invalid_rpc.commands
assert not invalid_rpc.holds
assert not invalid_rpc.gains
assert not invalid_rpc.events

print("real-time SAM servo checks passed")
