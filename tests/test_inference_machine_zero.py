from types import SimpleNamespace

import numpy as np

from robot.arm.arm import ArmNode
from robot.cone_e import ConeE
from rollout.controller import prepare_arms_for_inference


class _RecordingRobotRPC:
    def __init__(self):
        self.calls = []

    def machine_zero_arms(self):
        self.calls.append("machine_zero_arms")

    def home_left_arm(self):
        self.calls.append("home_left_arm")

    def home_right_arm(self):
        self.calls.append("home_right_arm")


def test_inference_start_visits_machine_zero_before_manipulation_home():
    robot = _RecordingRobotRPC()

    prepare_arms_for_inference(robot)

    assert robot.calls == [
        "machine_zero_arms",
        "home_left_arm",
        "home_right_arm",
    ]


class _RecordingArm:
    def __init__(self, side, calls):
        self.side = side
        self.calls = calls

    def machine_zero(self):
        self.calls.append(self.side)


class _RecordingInitArm:
    def __init__(self, side, calls):
        self.side = side
        self.calls = calls

    def init(self, reset=True):
        self.calls.append((self.side, reset))


def test_cone_e_machine_zero_uses_server_startup_arm_order():
    calls = []
    cone = object.__new__(ConeE)
    cone._initialized = True
    cone.left_arm = _RecordingArm("left", calls)
    cone.right_arm = _RecordingArm("right", calls)

    cone.machine_zero_arms()

    assert calls == ["left", "right"]


def test_cone_e_can_initialize_without_implicit_motion_before_explicit_zero():
    calls = []
    cone = object.__new__(ConeE)
    cone._initialized = False
    cone.no_arms = False
    cone.reset_arms_on_init = True
    cone.left_arm = _RecordingInitArm("left", calls)
    cone.right_arm = _RecordingInitArm("right", calls)

    cone.init(reset_arms=False)

    assert calls == [("left", False), ("right", False)]
    assert cone._initialized is True


def test_cone_e_default_init_keeps_normal_teleop_machine_zero_behavior():
    calls = []
    cone = object.__new__(ConeE)
    cone._initialized = False
    cone.no_arms = False
    cone.reset_arms_on_init = True
    cone.left_arm = _RecordingInitArm("left", calls)
    cone.right_arm = _RecordingInitArm("right", calls)

    cone.init()

    assert calls == [("left", True), ("right", True)]


class _FakePiper:
    def __init__(self):
        self.reset_count = 0

    def reset_to_home(self):
        self.reset_count += 1

    def get_joint_state(self):
        return SimpleNamespace(pos=np.zeros(6))


class _FakeGripper:
    def __init__(self):
        self.open_count = 0

    def open(self):
        self.open_count += 1


class _FakeIKSolver:
    def __init__(self):
        self.configuration = None

    def update_configuration(self, q):
        self.configuration = np.asarray(q).copy()


def test_arm_machine_zero_calls_driver_q_zero_and_resyncs_ik(monkeypatch):
    arm = object.__new__(ArmNode)
    arm.piper = _FakePiper()
    arm.gripper = _FakeGripper()
    arm.ik_solver = _FakeIKSolver()
    monkeypatch.setattr("robot.arm.arm.time.sleep", lambda _seconds: None)

    q = arm.machine_zero()

    assert arm.piper.reset_count == 1
    assert arm.gripper.open_count == 1
    np.testing.assert_array_equal(q, np.zeros(6))
    np.testing.assert_array_equal(arm.ik_solver.configuration, np.zeros(6))
