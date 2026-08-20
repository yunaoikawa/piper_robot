import robot.arm.arm as arm_module
from robot.arm.arm import (
    ArmNode,
    COMM_SUCCESS,
    DynamixelGripper,
    PIPER_CONTROLLER_DT_S,
)
from teleop_collect_example import (
    CONTROL_FREQ,
    TELEOP_COMMAND_PREVIEW_S,
    TELEOP_MAX_JOINT_STEP_RAD,
)
import numpy as np
from types import SimpleNamespace


def _gripper():
    gripper = object.__new__(DynamixelGripper)
    gripper.dxl_id = 1
    gripper.port = object()
    gripper.packet = object()
    gripper.pos_open = 2800
    gripper.pos_close = 6300
    gripper._last_commanded_pos = None
    return gripper


def test_repeated_gripper_hold_is_latched_without_usb_rewrites(monkeypatch):
    writes = []

    def fake_write(_packet, _port, dxl_id, pos):
        writes.append((dxl_id, pos))
        return COMM_SUCCESS, 0

    monkeypatch.setattr(arm_module, "_write_pos", fake_write)
    gripper = _gripper()

    assert gripper.set_open_ratio(0.0) is True
    assert gripper.set_open_ratio(0.0) is False
    assert gripper.set_open_ratio(0.0) is False
    assert writes == [(1, 6300)]

    assert gripper.set_open_ratio(1.0) is True
    assert writes == [(1, 6300), (1, 2800)]


def test_failed_gripper_write_is_retried(monkeypatch):
    writes = []

    def fake_write(_packet, _port, _dxl_id, pos):
        writes.append(pos)
        if len(writes) == 1:
            return COMM_SUCCESS + 1, 0
        return COMM_SUCCESS, 0

    monkeypatch.setattr(arm_module, "_write_pos", fake_write)
    gripper = _gripper()

    assert gripper.set_open_ratio(0.0) is True
    assert gripper._last_commanded_pos is None
    assert gripper.set_open_ratio(0.0) is True
    assert gripper._last_commanded_pos == 6300
    assert writes == [6300, 6300]


def test_stream_preview_bridges_measured_tail_and_controller_is_schedulable():
    assert TELEOP_COMMAND_PREVIEW_S >= 3.0 / CONTROL_FREQ
    assert PIPER_CONTROLLER_DT_S == 0.01


class _TeleopPiper:
    def __init__(self, q):
        self.q = np.asarray(q, dtype=float)
        self.commands = []

    def get_joint_state(self):
        return SimpleNamespace(pos=self.q.copy())

    def get_timestamp(self):
        return 10.0

    def set_joint_cmd(self, cmd):
        self.commands.append(cmd)


class _TeleopIK:
    def __init__(self, qd):
        self.qd = np.asarray(qd, dtype=float)
        self.updated = []
        self.max_iters = []

    def update_configuration(self, q):
        self.updated.append(np.asarray(q, dtype=float).copy())

    def solve_ik(self, _target, *, max_iter):
        self.max_iters.append(max_iter)
        return self.qd.copy(), False


class _TeleopGripper:
    def __init__(self):
        self.targets = []

    def set_open_ratio(self, target):
        self.targets.append(target)


def _teleop_arm(q, qd):
    arm = object.__new__(ArmNode)
    arm.piper = _TeleopPiper(q)
    arm.ik_solver = _TeleopIK(qd)
    arm.gripper = _TeleopGripper()
    arm.robot_config = SimpleNamespace(joint_dof=6)
    arm._last_ik_warning_time = 0.0
    return arm


def test_teleop_ik_uses_one_iteration_and_steps_toward_far_target():
    arm = _teleop_arm(np.zeros(6), np.array([1.5, -0.6, 0.3, 0, 0, 0]))

    assert arm.set_teleop_ee_target(
        object(),
        gripper_target=0.0,
        preview_time=TELEOP_COMMAND_PREVIEW_S,
        max_joint_step_rad=TELEOP_MAX_JOINT_STEP_RAD,
    )

    assert arm.ik_solver.max_iters == [1]
    np.testing.assert_array_equal(arm.ik_solver.updated[0], np.zeros(6))
    assert len(arm.piper.commands) == 1
    command = arm.piper.commands[0]
    assert np.max(np.abs(command.pos)) <= TELEOP_MAX_JOINT_STEP_RAD + 1e-12
    # Uniform scaling preserves the IK direction instead of clipping joints
    # independently or rejecting every subsequent frame.
    np.testing.assert_allclose(command.pos / command.pos[0],
                               np.array([1.0, -0.4, 0.2, 0, 0, 0]))
    assert command.timestamp == 10.0 + TELEOP_COMMAND_PREVIEW_S
    assert arm.gripper.targets == [0.0]


def test_teleop_ik_preserves_nearby_solution_without_scaling():
    q = np.arange(6, dtype=float) * 0.1
    qd = q + np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.0])
    arm = _teleop_arm(q, qd)

    assert arm.set_teleop_ee_target(object())

    np.testing.assert_allclose(arm.piper.commands[0].pos, qd)
