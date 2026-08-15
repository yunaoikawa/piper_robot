import numpy as np

from robot.arm.arm import ArmNode, DynamixelGripper, _write_pos


class _Packet:
    def __init__(self):
        self.calls = []

    def write4ByteTxRx(self, *args):
        self.calls.append(("rx", args))

    def write4ByteTxOnly(self, *args):
        self.calls.append(("only", args))


def test_dynamixel_stream_write_does_not_wait_for_status():
    packet = _Packet()
    _write_pos(packet, object(), 1, 4200, wait_for_status=False)
    assert [call[0] for call in packet.calls] == ["only"]


def test_dynamixel_normal_write_keeps_status_confirmation():
    packet = _Packet()
    _write_pos(packet, object(), 1, 4200)
    assert [call[0] for call in packet.calls] == ["rx"]


def test_set_open_ratio_selects_streaming_write_without_hardware():
    packet = _Packet()
    gripper = object.__new__(DynamixelGripper)
    gripper.dxl_id = 1
    gripper.inverted = False
    gripper.port = object()
    gripper.packet = packet
    gripper.pos_open = 2800
    gripper.pos_close = 6300
    gripper.set_open_ratio(0.5, wait_for_status=False)
    assert packet.calls[0][0] == "only"


def test_arm_stream_skips_unchanged_gripper_targets():
    class Gripper:
        def __init__(self):
            self.calls = []

        def set_open_ratio(self, value, *, wait_for_status=True):
            self.calls.append((value, wait_for_status))

    arm = object.__new__(ArmNode)
    arm.gripper = Gripper()
    arm._last_streamed_gripper_target = None
    arm._stream_gripper_target(1.0)
    arm._stream_gripper_target(1.0)
    arm._stream_gripper_target(0.5)
    assert arm.gripper.calls == [(1.0, False), (0.5, False)]
