"""Pure numpy adapters between bimanual robot wire data and ACT checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np


def r6_absolute_to_quat_wxyz(r6: np.ndarray) -> np.ndarray:
    """Convert one or more absolute 6D rotations to wxyz quaternions."""

    value = np.asarray(r6, dtype=np.float64)
    if value.ndim == 1:
        value = value[None]
    r1, r2 = value[..., :3], value[..., 3:6]
    norm1 = np.linalg.norm(r1, axis=-1, keepdims=True)
    if np.any(norm1 < 1e-8):
        raise ValueError("invalid first r6 axis")
    b1 = r1 / norm1
    b2 = r2 - np.sum(b1 * r2, axis=-1, keepdims=True) * b1
    norm2 = np.linalg.norm(b2, axis=-1, keepdims=True)
    if np.any(norm2 < 1e-8):
        raise ValueError("invalid second r6 axis")
    b2 /= norm2
    b3 = np.cross(b1, b2)
    matrix = np.stack([b1, b2, b3], axis=-1)

    result = np.empty((len(matrix), 4), dtype=np.float64)
    for index, rotation in enumerate(matrix):
        trace = np.trace(rotation)
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2
            result[index] = [
                scale / 4,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        else:
            axis = int(np.argmax(np.diag(rotation)))
            if axis == 0:
                scale = np.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
                result[index] = [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    scale / 4,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            elif axis == 1:
                scale = np.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
                result[index] = [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    scale / 4,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            else:
                scale = np.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
                result[index] = [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    scale / 4,
                ]
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result.astype(np.float32)


def adapt_observation_for_active_arm(
    observation: dict[str, Any], expected_state_dim: int, active_arm: str
) -> dict[str, Any]:
    """Slice a 20D left+right observation for a single-arm 10D checkpoint."""

    if active_arm not in {"left", "right", "both"}:
        raise ValueError("active_arm must be left, right, or both")
    output = dict(observation)
    qpos = np.asarray(observation["qpos"])
    if qpos.shape[-1] == expected_state_dim:
        return output
    if qpos.shape[-1] == 20 and expected_state_dim == 10:
        if active_arm == "both":
            raise ValueError("a 10D ACT checkpoint requires one active arm")
        output["qpos"] = qpos[..., :10] if active_arm == "left" else qpos[..., 10:20]
        return output
    raise ValueError(
        f"cannot adapt {qpos.shape[-1]}D robot state to {expected_state_dim}D ACT state"
    )


def action_chunk_to_quat16(action: np.ndarray, active_arm: str) -> np.ndarray:
    """Convert 10D single-arm or 20D bimanual r6 chunks to wire quat16."""

    value = np.asarray(action)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2:
        raise ValueError(f"ACT action chunk must be 2D, got {value.shape}")

    out = np.full((len(value), 16), np.nan, dtype=np.float32)
    if value.shape[1] == 20:
        out[:, 0:4] = r6_absolute_to_quat_wxyz(value[:, 3:9])
        out[:, 4:7] = value[:, 0:3]
        out[:, 7:11] = r6_absolute_to_quat_wxyz(value[:, 13:19])
        out[:, 11:14] = value[:, 10:13]
        out[:, 14] = value[:, 9]
        out[:, 15] = value[:, 19]
        return out
    if value.shape[1] != 10:
        raise ValueError(f"ACT action width must be 10 or 20, got {value.shape[1]}")
    if active_arm == "both":
        raise ValueError("a 10D ACT action requires one active arm")
    quat = r6_absolute_to_quat_wxyz(value[:, 3:9])
    if active_arm == "left":
        out[:, 0:4] = quat
        out[:, 4:7] = value[:, 0:3]
        out[:, 14] = value[:, 9]
    elif active_arm == "right":
        out[:, 7:11] = quat
        out[:, 11:14] = value[:, 0:3]
        out[:, 15] = value[:, 9]
    else:
        raise ValueError("active_arm must be left, right, or both")
    return out
