#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rollout.endpoint_interpolation import EndpointCalibration, EndpointSample


left_pose = np.array([1, 0, 0, 0, 0.10, -0.04, 0.88], dtype=float)
right_pose = np.array([0, 1, 0, 0, 0.30, 0.06, 0.91], dtype=float)
cal = EndpointCalibration(
    left=EndpointSample(np.array([100, 200]), left_pose),
    right=EndpointSample(np.array([500, 300]), right_pose),
    observer_pose=np.array([1, 0, 0, 0, 0, 0, 1], dtype=float),
)

for feature, expected_t in [
    ([100, 200], 0.0),
    ([300, 250], 0.5),
    ([500, 300], 1.0),
]:
    result = cal.interpolate(feature)
    assert abs(result.fraction - expected_t) < 1e-9

middle = cal.interpolate([300, 250])
assert np.allclose(middle.target_pose[4:6], [0.20, 0.01])
assert np.allclose(middle.target_pose[:4], left_pose[:4])
assert middle.target_pose[6] == left_pose[6]

try:
    cal.interpolate([300, 300])
    raise AssertionError("cross-track feature should be rejected")
except ValueError:
    pass

outside = cal.interpolate([700, 350])
assert outside.fraction == 1.0
assert outside.unclamped_fraction > 1.0

print("endpoint interpolation checks passed")
