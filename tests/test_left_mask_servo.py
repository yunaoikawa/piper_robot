#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from rollout.left_mask_servo import LeftMaskServo


servo = LeftMaskServo([100.0, 80.0])
servo.observe([80.0, 80.0], [0.0, 0.0])
first = servo.decide([80.0, 80.0], [0.008, 0.0])
assert not first.jacobian_ready
servo.observe([80.0, 70.0], [0.008, 0.0])
second = servo.decide([90.0, 70.0], [0.008, 0.008])
assert second.jacobian_ready
assert np.linalg.norm(second.delta_xy_m) <= 0.020001
aligned = servo.decide([100.0, 80.0], [0.008, 0.008])
assert np.allclose(aligned.delta_xy_m, 0.0)
print("left mask servo checks passed")
