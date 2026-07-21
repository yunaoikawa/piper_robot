#!/usr/bin/env python3
"""Hardware-free checks for replay_demo preflight and active-arm filtering."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replay_demo import (
    camera_rgb_to_bgr,
    detect_active_arms,
    frame_to_action,
    longest_grip_start,
    validate_demo,
    validate_start_pose,
)

rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
bgr = camera_rgb_to_bgr(rgb)
assert bgr.shape == (2, 1, 3)
assert bgr[0, 0].tolist() == [0, 0, 255]


def demo(n=4):
    d = {"timestamps": np.arange(n, dtype=float) / 30.0}
    for side, y in (("left", -0.46), ("right", 0.04)):
        d[f"{side}_ee_pos"] = np.tile([0.3, y, 0.83], (n, 1)).astype(float)
        d[f"{side}_ee_quat"] = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        d[f"{side}_gripper"] = np.ones(n)
    return d


def expect_exit(fn, text):
    try:
        fn()
    except SystemExit as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"expected SystemExit containing {text!r}")


d = demo()
d["right_ee_pos"][2, 0] += 0.01
d["right_gripper"][3] = 0.0
active, _ = detect_active_arms(d)
assert active == ["right"], active
action = frame_to_action(d, 2, active)
assert "right_ee_pose" in action and "right_gripper" in action
assert "left_ee_pose" not in action and "left_gripper" not in action
goal = longest_grip_start(d, active)
assert goal == {"arm": "right", "frame": 3, "length": 1}, goal
validate_demo(d, 4, active, max_step_m=0.04)

bad = demo()
bad["right_ee_pos"][1, 0] += 0.05
expect_exit(lambda: validate_demo(bad, 4, ["right"], 0.04), "max step")

bad = demo()
bad["right_ee_quat"][1] = [2.0, 0.0, 0.0, 0.0]
expect_exit(lambda: validate_demo(bad, 4, ["right"], 0.04), "quaternion norm")

bad = demo()
bad["right_ee_pos"][1, 2] = 2.0
expect_exit(lambda: validate_demo(bad, 4, ["right"], 2.0), "outside workspace")


class FakePose:
    def __init__(self, pos, quat):
        self._pos = np.asarray(pos)
        self._rotation = type("Rotation", (), {"wxyz": np.asarray(quat)})()

    def translation(self):
        return self._pos

    def rotation(self):
        return self._rotation


class FakeRobot:
    def __init__(self, pose):
        self.pose = pose

    def get_right_ee_pose(self):
        return self.pose


class FakeController:
    def __init__(self, pose):
        self.cone_e = FakeRobot(pose)


d = demo()
near = FakeController(FakePose([0.301, 0.04, 0.83], [1, 0, 0, 0]))
validate_start_pose(near, d, ["right"])
far = FakeController(FakePose([0.4, 0.04, 0.83], [1, 0, 0, 0]))
expect_exit(lambda: validate_start_pose(far, d, ["right"]), "start distance")

print("all replay_demo checks passed")
