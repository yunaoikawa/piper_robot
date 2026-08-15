import numpy as np

from robot.cone_e import ConeE


class _Rotation:
    wxyz = np.array([1.0, 0.0, 0.0, 0.0])


class _Pose:
    def __init__(self, marker):
        self.marker = marker

    def translation(self):
        return np.array([self.marker, 0.0, 0.0])

    def rotation(self):
        return _Rotation()


class _Gripper:
    def __init__(self, value):
        self.value = value
        self.reads = 0

    def get_open_ratio(self):
        self.reads += 1
        return self.value


class _Arm:
    def __init__(self, marker, gripper):
        self.marker = marker
        self.gripper = gripper
        self.joint_reads = 0
        self.torque_reads = 0
        self.pose_inputs = []

    def get_joint_positions(self):
        self.joint_reads += 1
        return np.full(6, self.marker)

    def get_ee_pose(self, joints):
        self.pose_inputs.append(np.asarray(joints).copy())
        return _Pose(self.marker)

    def get_joint_torque(self):
        self.torque_reads += 1
        return np.full(6, self.marker / 10.0)


def _fake_cone():
    cone = ConeE.__new__(ConeE)
    cone._initialized = True
    cone.left_arm = _Arm(1.0, _Gripper(.8))
    cone.right_arm = _Arm(2.0, _Gripper(.6))
    return cone


def test_single_arm_observation_snapshot_reads_joints_once_and_only_active_gripper():
    cone = _fake_cone()
    state = cone.get_observation_state(active_arm="right")

    assert cone.left_arm.joint_reads == 1
    assert cone.right_arm.joint_reads == 1
    assert cone.left_arm.gripper.reads == 0
    assert cone.right_arm.gripper.reads == 1
    assert cone.left_arm.torque_reads == 0
    assert cone.right_arm.torque_reads == 1
    assert state["right_joint_torque"].tolist() == [.2] * 6
    assert state["left_gripper_exact"] is None
    assert state["right_gripper_exact"] == .6
    assert cone.left_arm.pose_inputs[0].tolist() == [1.0] * 6
    assert cone.right_arm.pose_inputs[0].tolist() == [2.0] * 6


def test_bimanual_observation_snapshot_reads_both_grippers():
    cone = _fake_cone()
    state = cone.get_observation_state()

    assert state["left_gripper_exact"] == .8
    assert state["right_gripper_exact"] == .6
    assert cone.left_arm.gripper.reads == 1
    assert cone.right_arm.gripper.reads == 1
    assert cone.left_arm.torque_reads == 1
    assert cone.right_arm.torque_reads == 1
