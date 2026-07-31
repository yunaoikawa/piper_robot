"""Canonical ConeE Piper home joint positions.

The physical arms are mounted on the opposite-named branches in the production
MuJoCo model:

* physical right -> ``left_arm_*``
* physical left  -> ``right_arm_*``

Keep the physical-arm constants authoritative. Production and semantic MJCF
orders are intentionally exposed through different functions.
"""

from __future__ import annotations

import numpy as np


PHYSICAL_LEFT_HOME_Q = (0.0, 1.58, -0.58, 0.0, -0.91, 1.40)
PHYSICAL_RIGHT_HOME_Q = (0.0, 1.58, -0.58, 0.0, -0.91, 2.35)

# The semantic Piper CAD uses a common joint-6 zero for both arms, while the
# physical right arm reports a different wrist-roll zero.  With the pinned NYU
# gripper, 1.355 rad makes the gripper's fitted plane parallel to the support
# plane at physical home.  Keep this model-frame calibration separate from the
# physical home command above.
SEMANTIC_NYU_HORIZONTAL_HOME_Q6_RAD = 1.355


def physical_home_q(side: str) -> np.ndarray:
    """Return a writable copy of one physical arm's canonical home qpos."""

    if side == "left":
        values = PHYSICAL_LEFT_HOME_Q
    elif side == "right":
        values = PHYSICAL_RIGHT_HOME_Q
    else:
        raise ValueError("physical arm side must be 'left' or 'right'")
    return np.asarray(values, dtype=float).copy()


def semantic_model_home_q(side: str) -> np.ndarray:
    """Return physical home expressed in the semantic MuJoCo joint frame."""

    values = physical_home_q(side)
    values[5] = SEMANTIC_NYU_HORIZONTAL_HOME_Q6_RAD
    return values


def physical_to_semantic_model_q_offset(side: str) -> np.ndarray:
    """Return the fixed joint-zero bridge from physical to semantic q."""

    return semantic_model_home_q(side) - physical_home_q(side)


def production_mujoco_home_qpos() -> np.ndarray:
    """Return ConeE production qpos order: left_arm then right_arm."""

    return np.asarray(
        (*PHYSICAL_RIGHT_HOME_Q, *PHYSICAL_LEFT_HOME_Q),
        dtype=float,
    )


def semantic_mujoco_home_qpos() -> np.ndarray:
    """Return semantic qpos order: physical left then physical right."""

    return np.asarray(
        (*PHYSICAL_LEFT_HOME_Q, *PHYSICAL_RIGHT_HOME_Q),
        dtype=float,
    )
