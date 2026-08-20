import robot.arm.arm as arm_module
from robot.arm.arm import (
    COMM_SUCCESS,
    DynamixelGripper,
    PIPER_CONTROLLER_DT_S,
)
from teleop_collect_example import (
    CONTROL_FREQ,
    TELEOP_COMMAND_PREVIEW_S,
)


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
