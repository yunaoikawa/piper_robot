#!/usr/bin/env python3
"""Hardware-free checks for strict right-only Quest teleoperation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.right_only_teleop import RightOnlyTeleop, RightTeleopEvent


class FakeSafety:
    def __init__(self):
        self.reset_arms = []

    def reset(self, arm=None):
        self.reset_arms.append(arm)

    def check(self, arm, position):
        assert arm == "right"
        return np.asarray(position, dtype=float)


class RightOnlyRPC:
    def __init__(self):
        self.pose = mink.SE3(np.array([1, 0, 0, 0, 0.3, 0.1, 0.8], dtype=float))
        self.calls = []

    def get_right_ee_pose(self):
        self.calls.append(("get_right_ee_pose",))
        return self.pose

    def set_right_ee_target(self, **kwargs):
        self.calls.append(("set_right_ee_target", kwargs))

    def __getattr__(self, name):
        if "left" in name:
            raise AssertionError(f"left-arm access is forbidden: {name}")
        raise AttributeError(name)


def controller_state(now, *, a=False, b=False, trigger=0.0, xyz=(0, 0, 0)):
    pose = mink.SE3(np.array([1, 0, 0, 0, *xyz], dtype=float))
    return SimpleNamespace(
        created_timestamp=now,
        right_a=a,
        right_b=b,
        right_index_trigger=trigger,
        right_SE3=pose,
    )


def test_idle_makes_no_rpc_calls():
    rpc = RightOnlyRPC()
    teleop = RightOnlyTeleop(rpc, safety=FakeSafety())
    assert teleop.step(controller_state(10.0), now=10.0) == RightTeleopEvent.NONE
    assert rpc.calls == []


def test_startup_held_a_cannot_engage_until_release_and_repress():
    rpc = RightOnlyRPC()
    teleop = RightOnlyTeleop(rpc, safety=FakeSafety())
    assert teleop.step(controller_state(10.0, a=True), now=10.0) == RightTeleopEvent.NONE
    assert rpc.calls == []
    assert teleop.step(controller_state(10.1), now=10.1) == RightTeleopEvent.NONE
    assert teleop.step(controller_state(10.2, a=True), now=10.2) == RightTeleopEvent.ENGAGED


def test_right_engage_move_and_stop_never_access_left():
    rpc = RightOnlyRPC()
    safety = FakeSafety()
    teleop = RightOnlyTeleop(rpc, safety=safety)

    teleop.step(controller_state(9.9), now=9.9)
    event = teleop.step(controller_state(10.0, a=True), now=10.0)
    assert event == RightTeleopEvent.ENGAGED
    assert [call[0] for call in rpc.calls] == [
        "get_right_ee_pose",
        "set_right_ee_target",
    ]

    event = teleop.step(controller_state(10.1, xyz=(0.01, 0.02, 0.03)), now=10.1)
    assert event == RightTeleopEvent.COMMANDED
    assert rpc.calls[-1][0] == "set_right_ee_target"

    event = teleop.step(controller_state(10.2, b=True), now=10.2)
    assert event == RightTeleopEvent.DISENGAGED
    call_count = len(rpc.calls)
    assert teleop.step(controller_state(10.3), now=10.3) == RightTeleopEvent.NONE
    assert len(rpc.calls) == call_count
    assert set(safety.reset_arms) == {"right"}


def test_stale_stream_disengages_without_new_command():
    rpc = RightOnlyRPC()
    teleop = RightOnlyTeleop(rpc, timeout_s=0.5, safety=FakeSafety())
    teleop.step(controller_state(9.9), now=9.9)
    teleop.step(controller_state(10.0, a=True), now=10.0)
    call_count = len(rpc.calls)
    assert teleop.step(controller_state(10.0), now=10.6) == RightTeleopEvent.STALE
    assert not teleop.engaged
    assert len(rpc.calls) == call_count


if __name__ == "__main__":
    test_idle_makes_no_rpc_calls()
    test_startup_held_a_cannot_engage_until_release_and_repress()
    test_right_engage_move_and_stop_never_access_left()
    test_stale_stream_disengages_without_new_command()
    print("right-only teleop checks passed")
