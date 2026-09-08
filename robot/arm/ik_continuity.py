"""Helpers for keeping Cartesian commands on the current IK branch."""

from __future__ import annotations

import numpy as np


def joint_target_is_continuous(
    current_joints, target_joints, max_delta_rad: float = 0.50
) -> tuple[bool, np.ndarray]:
    current = np.asarray(current_joints, dtype=float)
    target = np.asarray(target_joints, dtype=float)
    if (
        current.shape != (6,)
        or target.shape != (6,)
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("current and target joints must be finite shape-(6,) arrays")
    delta = target - current
    return bool(np.max(np.abs(delta)) <= max_delta_rad), delta
