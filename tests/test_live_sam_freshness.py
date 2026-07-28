#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.run_realtime_sam_grasp as realtime
from src.run_realtime_sam_grasp import LiveSamGrasp, _rgb_frame_sha256


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.calls = 0

    def get_latest_frame(self):
        index = min(self.calls, len(self.frames) - 1)
        self.calls += 1
        return (
            self.frames[index].copy(),
            time.time() + 0.001,
            np.ones((3, 4), dtype=np.float32),
        )


def runner_with(camera, previous=None):
    runner = object.__new__(LiveSamGrasp)
    runner.camera = camera
    runner.last_head_timestamp = (
        None if previous is None else time.time() - 1.0
    )
    runner.last_head_rgb_sha256 = (
        None if previous is None else _rgb_frame_sha256(previous)
    )
    return runner


first = np.zeros((4, 5, 3), dtype=np.uint8)
second = first.copy()
second[1, 2, 0] = 1

fresh_camera = FakeCamera([first, first, second])
fresh_runner = runner_with(fresh_camera, previous=first)
rgb, timestamp, depth = fresh_runner._await_fresh_head_frame(timeout_s=0.2)
assert fresh_camera.calls == 3
assert np.array_equal(rgb, second)
assert timestamp > 0.0
assert depth.shape == (3, 4)

frozen_runner = runner_with(FakeCamera([first]), previous=first)
try:
    frozen_runner._await_fresh_head_frame(timeout_s=0.03)
    raise AssertionError("repeated stale RGB bytes were accepted")
except RuntimeError as exc:
    assert "RGB bytes repeated" in str(exc)

initial_runner = runner_with(FakeCamera([first]))
rgb, _, _ = initial_runner._await_fresh_head_frame(timeout_s=0.1)
assert np.array_equal(rgb, first)


class ForbiddenPhysicalRunner:
    def __init__(self, _args):
        raise AssertionError("disabled standalone execution constructed a runner")


original_runner_class = realtime.LiveSamGrasp
realtime.LiveSamGrasp = ForbiddenPhysicalRunner
try:
    try:
        realtime.main(["--execute-pregrasp"])
        raise AssertionError("standalone 3D execution was accepted")
    except SystemExit as exc:
        assert exc.code == 2
finally:
    realtime.LiveSamGrasp = original_runner_class


class FailingDryRunner:
    holds = 0
    stops = 0

    def __init__(self, _args):
        pass

    def start(self):
        pass

    def observe(self, _clearance_m):
        raise RuntimeError("camera-only observation failed")

    def hold_measured(self):
        type(self).holds += 1

    def stop(self):
        type(self).stops += 1


realtime.LiveSamGrasp = FailingDryRunner
try:
    try:
        realtime.main([])
        raise AssertionError("failed dry observation did not propagate")
    except RuntimeError as exc:
        assert "camera-only observation failed" in str(exc)
finally:
    realtime.LiveSamGrasp = original_runner_class
assert FailingDryRunner.holds == 0
assert FailingDryRunner.stops == 1

print("live SAM freshness checks passed")
