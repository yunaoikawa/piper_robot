#!/usr/bin/env python3

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_staged_sam_pregrasp import (
    bound_horizontal_step,
    estimate_horizontal_displacement,
    execute_single_horizontal_probe,
    fit_horizontal_jacobian,
)


true_jacobian = np.array(
    [[420.0, -90.0, 110.0], [60.0, 310.0, -520.0], [80.0, 640.0, -220.0]]
)
robot = np.array(
    [[0.006, 0.0, 0.0], [0.0, 0.006, 0.0], [0.0, 0.0, 0.006]]
)
feature = robot @ true_jacobian.T
estimated = fit_horizontal_jacobian(robot[:2], feature[:2])
assert np.allclose(estimated, true_jacobian[:2, :2])

expected_displacement = np.array([0.045, -0.070, -0.090])
error = np.r_[estimated @ expected_displacement[:2], 999.0]
displacement = estimate_horizontal_displacement(estimated, error)
assert np.allclose(displacement, [*expected_displacement[:2], 0.0])

step = bound_horizontal_step(
    displacement, max_norm_m=0.008, max_axis_m=0.006
)
assert step[2] == 0.0
assert np.linalg.norm(step[:2]) <= 0.008001
assert np.max(np.abs(step[:2])) <= 0.006001
assert np.dot(step[:2], expected_displacement[:2]) > 0.0


class FakeSingleProbeRunner:
    def __init__(self):
        self.moves = []

    def move_cartesian_delta(self, request, minimum_progress):
        self.moves.append(
            (np.asarray(request, dtype=float).copy(), minimum_progress)
        )
        return np.array([0.0058, -0.0001, 0.0])

    def observe(self, clearance_m):
        feature = object()
        error = np.array([-80.0, 180.0, -60.0])
        return feature, error, "/tmp/head.png", 123.0


class FakeSingleProbeRight:
    def __init__(self):
        self.observations = 0

    def observe(self, require_lid):
        assert require_lid is False
        self.observations += 1
        geometry = SimpleNamespace(center_px=np.array([120.0, 90.0]))
        candidate = SimpleNamespace(score=0.8)
        return geometry, candidate, "/tmp/right.png", 124.0


single_runner = FakeSingleProbeRunner()
single_right = FakeSingleProbeRight()
single_report = execute_single_horizontal_probe(
    single_runner, single_right, "x", 0.006
)
assert len(single_runner.moves) == 1
assert np.array_equal(single_runner.moves[0][0], [0.006, 0.0, 0.0])
assert single_runner.moves[0][1] == 0.0
assert single_right.observations == 1
assert single_report["status"] == "SINGLE_X_PROBE_COMPLETE_AND_HELD"
assert single_report["head_image"] == "/tmp/head.png"
assert single_report["right_image"] == "/tmp/right.png"

try:
    execute_single_horizontal_probe(
        FakeSingleProbeRunner(), FakeSingleProbeRight(), "z", 0.006
    )
    raise AssertionError("non-horizontal single probe was accepted")
except ValueError:
    pass

print("staged SAM pregrasp checks passed")
