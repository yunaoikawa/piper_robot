#!/usr/bin/env python3
"""Hardware-free checks for torque calibration and sustained trips."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.torque_safety import (
    TorqueCalibrator,
    TorqueWatchdog,
    torque_stop_enabled_from_config,
)


cal = TorqueCalibrator()
for value in (1.0, 1.1, 0.9, 1.05):
    cal.add("right", [value, 2 * value])
cfg = cal.build_config()
assert cfg["consecutive_samples"] == 5
assert np.all(np.asarray(cfg["thresholds"]["right"]) > [1.1, 2.2])

watchdog = TorqueWatchdog({"right": [2.0, 3.0]}, consecutive_limit=3)
assert watchdog.check("right", [2.1, 0.0])
assert watchdog.check("right", [0.0, 0.0])  # transient resets
assert watchdog.check("right", [2.1, 0.0])
assert watchdog.check("right", [2.1, 0.0])
assert not watchdog.check("right", [2.1, 0.0])
assert watchdog.tripped["joint"] == 0

bad = TorqueWatchdog({"right": [2.0]})
assert not bad.check("right", [float("nan")])

assert torque_stop_enabled_from_config({}) is True
assert torque_stop_enabled_from_config(
    {"motion_torque_policy": "observe_only"}
) is False
try:
    torque_stop_enabled_from_config({"motion_torque_policy": "unknown"})
except ValueError:
    pass
else:
    raise AssertionError("unknown torque policy was accepted")

print("torque safety checks passed")
