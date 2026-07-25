#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_staged_sam_pregrasp import (
    bound_horizontal_step,
    estimate_horizontal_displacement,
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

print("staged SAM pregrasp checks passed")
