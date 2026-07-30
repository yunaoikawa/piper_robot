"""Canonical ConeE Piper home joint positions.

The physical arms are mounted on the opposite-named branches in the production
MuJoCo model:

* physical right -> ``left_arm_*``
* physical left  -> ``right_arm_*``

Keep the physical-arm constants authoritative.  Renderers and planners must
use :func:`mujoco_home_qpos` instead of copying values or trusting an old MJCF
keyframe.
"""

from __future__ import annotations

import numpy as np


PHYSICAL_LEFT_HOME_Q = (0.0, 1.58, -0.58, 0.0, -0.91, 1.40)
PHYSICAL_RIGHT_HOME_Q = (0.0, 1.58, -0.58, 0.0, -0.91, 2.35)


def physical_home_q(side: str) -> np.ndarray:
    """Return a writable copy of one physical arm's canonical home qpos."""

    if side == "left":
        values = PHYSICAL_LEFT_HOME_Q
    elif side == "right":
        values = PHYSICAL_RIGHT_HOME_Q
    else:
        raise ValueError("physical arm side must be 'left' or 'right'")
    return np.asarray(values, dtype=float).copy()


def mujoco_home_qpos() -> np.ndarray:
    """Return production model qpos order: model-left then model-right."""

    return np.asarray(
        (*PHYSICAL_RIGHT_HOME_Q, *PHYSICAL_LEFT_HOME_Q),
        dtype=float,
    )
