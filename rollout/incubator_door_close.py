"""Retarget an already executed opening path for open-jaw door closing."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import h5py
import mink
import numpy as np

from rollout.incubator_door_demo import quaternion_distance_rad


def load_open_jaw_close_trajectory(path: str | Path) -> list[dict]:
    """Load a demonstrated right-arm push while rejecting grasp trajectories."""

    with h5py.File(Path(path), "r") as recording:
        position = np.asarray(recording["right_ee_pos"][:], dtype=float)
        quaternion = np.asarray(recording["right_ee_quat"][:], dtype=float)
        gripper = np.asarray(recording["right_gripper"][:], dtype=float)
        left_position = np.asarray(recording["left_ee_pos"][:], dtype=float)
        timestamps = np.asarray(recording["timestamps"][:], dtype=float)
    count = len(position)
    if not (
        position.shape == (count, 3)
        and quaternion.shape == (count, 4)
        and gripper.shape == (count,)
        and timestamps.shape == (count,)
        and count >= 2
        and np.all(np.isfinite(position))
        and np.all(np.isfinite(quaternion))
    ):
        raise ValueError("unsupported door-close recording schema")
    if float(np.ptp(left_position, axis=0).max()) > 0.001:
        raise ValueError("door-close reference moves the inactive arm")
    if float(np.min(gripper)) < 0.95:
        raise ValueError("door-close reference does not keep its jaws open")
    started = float(timestamps[0])
    return [
        {
            "frame": index,
            "t_s": max((index + 1) / 30.0, float(timestamps[index]) - started + 1 / 30),
            "pose_wxyz_xyz": np.r_[quaternion[index], position[index]].tolist(),
            "gripper": 1.0,
        }
        for index in range(count)
    ]


def register_close_trajectory(
    trajectory: Sequence[dict], registration_wxyz_xyz
) -> list[dict]:
    """Apply one scene-registration transform to an absolute close demo."""

    registration = mink.SE3(np.asarray(registration_wxyz_xyz, dtype=float))
    result = []
    for sample in trajectory:
        pose = mink.SE3(np.asarray(sample["pose_wxyz_xyz"], dtype=float))
        result.append(
            {
                **sample,
                "pose_wxyz_xyz": np.asarray(
                    (registration @ pose).parameters(), dtype=float
                ).tolist(),
                "gripper": 1.0,
            }
        )
    return result


def nearest_pose_index(
    pose_wxyz_xyz,
    trajectory: Sequence[dict],
    *,
    rotation_weight_m_per_rad: float = 0.03,
) -> dict:
    """Find the opening sample which best matches the measured live wrist."""

    live = np.asarray(pose_wxyz_xyz, dtype=float).reshape(7)
    candidates = []
    for index, sample in enumerate(trajectory):
        pose = np.asarray(sample["pose_wxyz_xyz"], dtype=float).reshape(7)
        position_error = float(np.linalg.norm(live[4:] - pose[4:]))
        rotation_error = quaternion_distance_rad(live[:4], pose[:4])
        candidates.append(
            {
                "index": index,
                "frame": int(sample["frame"]),
                "position_error_m": position_error,
                "rotation_error_deg": math.degrees(rotation_error),
                "score": position_error
                + float(rotation_weight_m_per_rad) * rotation_error,
            }
        )
    if not candidates:
        raise ValueError("opening trajectory is empty")
    return min(candidates, key=lambda item: item["score"])


def reverse_opening_from_live_pose(
    live_pose_wxyz_xyz,
    opening_trajectory: Sequence[dict],
    *,
    nearest_index: int,
    control_hz: float = 30.0,
    aperture: float = 1.0,
) -> list[dict]:
    """Reverse an opening prefix after rigidly anchoring it at the live pose.

    A constant SE(3) correction maps the chosen, already executed opening
    sample to the measured live wrist.  Applying the same correction to the
    earlier samples preserves the successful physical path while avoiding an
    absolute historical robot pose.
    """

    if not math.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if not 0.0 <= aperture <= 1.0:
        raise ValueError("aperture must be within [0, 1]")
    if not 0 <= int(nearest_index) < len(opening_trajectory):
        raise IndexError("nearest opening index is out of range")
    live = mink.SE3(np.asarray(live_pose_wxyz_xyz, dtype=float).reshape(7))
    anchor = mink.SE3(
        np.asarray(
            opening_trajectory[int(nearest_index)]["pose_wxyz_xyz"],
            dtype=float,
        ).reshape(7)
    )
    correction = live @ anchor.inverse()
    selected = reversed(opening_trajectory[: int(nearest_index) + 1])
    result = []
    for output_index, sample in enumerate(selected, start=1):
        source = mink.SE3(
            np.asarray(sample["pose_wxyz_xyz"], dtype=float).reshape(7)
        )
        result.append(
            {
                "frame": int(sample["frame"]),
                "t_s": output_index / float(control_hz),
                "pose_wxyz_xyz": np.asarray(
                    (correction @ source).parameters(), dtype=float
                ).tolist(),
                "gripper": float(aperture),
            }
        )
    return result
