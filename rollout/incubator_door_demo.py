"""Compile and retarget successful recessed-door demonstrations.

The recorded end-effector poses are expressed in the physical robot frame, but
the incubator may be translated between runs.  The reusable part of a door
opening demonstration is therefore the motion *relative to the pose at which
the gripper closes*.  This module keeps that distinction explicit: absolute
poses are statistics/diagnostics, while executable pull samples are SE(3)
offsets from the demonstrated contact frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import mink
import numpy as np


SCHEMA = "piper_robot.incubator_door_demo/v1"


def _pose(quaternion_wxyz, position_xyz) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
    position = np.asarray(position_xyz, dtype=float).reshape(3)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    result = np.r_[quaternion, position]
    if not np.all(np.isfinite(result)):
        raise ValueError("pose contains non-finite values")
    return result


def _quaternion_mean_wxyz(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=float)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    values[values[:, 0] < 0.0] *= -1.0
    _, vectors = np.linalg.eigh(values.T @ values)
    result = vectors[:, -1]
    if result[0] < 0.0:
        result *= -1.0
    return result


def quaternion_distance_rad(first, second) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(2.0 * math.acos(np.clip(abs(first @ second), -1.0, 1.0)))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DoorEpisode:
    stem: str
    hdf5_path: Path
    close_frame: int
    release_frame: int
    proof_frame: int
    preclose_frame: int
    timestamps_s: np.ndarray
    positions_m: np.ndarray
    quaternions_wxyz: np.ndarray
    gripper: np.ndarray

    @property
    def contact_pose(self) -> np.ndarray:
        return _pose(
            self.quaternions_wxyz[self.close_frame],
            self.positions_m[self.close_frame],
        )

    @property
    def preclose_pose(self) -> np.ndarray:
        return _pose(
            self.quaternions_wxyz[self.preclose_frame],
            self.positions_m[self.preclose_frame],
        )


def load_episode(
    path: str | Path,
    *,
    preclose_frames: int = 10,
    proof_pull_m: float = 0.005,
    closed_threshold: float = 0.8,
) -> DoorEpisode:
    path = Path(path)
    with h5py.File(path, "r") as recording:
        positions = np.asarray(recording["right_ee_pos"][:], dtype=float)
        quaternions = np.asarray(recording["right_ee_quat"][:], dtype=float)
        gripper = np.asarray(recording["right_gripper"][:], dtype=float)
        timestamps = np.asarray(recording["timestamps"][:], dtype=float)
    if not (
        positions.ndim == 2
        and positions.shape[1] == 3
        and quaternions.shape == (len(positions), 4)
        and gripper.shape == (len(positions),)
        and timestamps.shape == (len(positions),)
    ):
        raise ValueError(f"{path}: unsupported door recording schema")
    closed = np.flatnonzero(gripper < float(closed_threshold))
    if not len(closed):
        raise ValueError(f"{path}: gripper never closes")
    close = int(closed[0])
    release = int(closed[-1])
    origin = positions[close]
    distances = np.linalg.norm(positions[close : release + 1] - origin, axis=1)
    proof_candidates = np.flatnonzero(distances >= float(proof_pull_m))
    proof = (
        close + int(proof_candidates[0])
        if len(proof_candidates)
        else release
    )
    return DoorEpisode(
        stem=path.stem,
        hdf5_path=path,
        close_frame=close,
        release_frame=release,
        proof_frame=proof,
        preclose_frame=max(0, close - int(preclose_frames)),
        timestamps_s=timestamps - timestamps[0],
        positions_m=positions,
        quaternions_wxyz=quaternions,
        gripper=gripper,
    )


def _contact_medoid(episodes: Sequence[DoorEpisode]) -> int:
    positions = np.asarray([episode.contact_pose[4:] for episode in episodes])
    quaternions = np.asarray([episode.contact_pose[:4] for episode in episodes])
    scores = []
    for index in range(len(episodes)):
        score = 0.0
        for other in range(len(episodes)):
            score += float(np.linalg.norm(positions[index] - positions[other]) / 0.01)
            score += quaternion_distance_rad(
                quaternions[index], quaternions[other]
            ) / math.radians(5.0)
        scores.append(score)
    return int(np.argmin(scores))


def compile_demonstrations(paths: Iterable[str | Path]) -> dict:
    episodes = tuple(load_episode(path) for path in paths)
    if len(episodes) < 3:
        raise ValueError("at least three verified successful episodes are required")
    contact = np.asarray([episode.contact_pose for episode in episodes])
    mean_position = np.mean(contact[:, 4:], axis=0)
    mean_quaternion = _quaternion_mean_wxyz(contact[:, :4])
    orientation_errors = np.asarray(
        [quaternion_distance_rad(value, mean_quaternion) for value in contact[:, :4]]
    )
    medoid_index = _contact_medoid(episodes)
    medoid = episodes[medoid_index]

    reference_contact = mink.SE3(medoid.contact_pose)
    relative = []
    started = medoid.timestamps_s[medoid.close_frame]
    for frame in range(medoid.close_frame, medoid.release_frame + 1):
        sample = mink.SE3(
            _pose(
                medoid.quaternions_wxyz[frame],
                medoid.positions_m[frame],
            )
        )
        offset = reference_contact.inverse() @ sample
        relative.append(
            {
                "frame": frame,
                "t_s": float(medoid.timestamps_s[frame] - started),
                "pose_wxyz_xyz": np.asarray(offset.parameters(), dtype=float).tolist(),
                "gripper": float(medoid.gripper[frame]),
            }
        )

    return {
        "schema": SCHEMA,
        "verified_success_count": len(episodes),
        "successes": [
            {
                "stem": episode.stem,
                "hdf5": str(episode.hdf5_path.resolve()),
                "hdf5_sha256": sha256_file(episode.hdf5_path),
                "close_frame": episode.close_frame,
                "preclose_frame": episode.preclose_frame,
                "proof_frame": episode.proof_frame,
                "release_frame": episode.release_frame,
            }
            for episode in episodes
        ],
        "contact_statistics": {
            "mean_pose_wxyz_xyz": np.r_[mean_quaternion, mean_position].tolist(),
            "position_std_m": np.std(contact[:, 4:], axis=0).tolist(),
            "position_range_m": np.ptp(contact[:, 4:], axis=0).tolist(),
            "orientation_error_median_deg": float(
                np.degrees(np.median(orientation_errors))
            ),
            "orientation_error_max_deg": float(
                np.degrees(np.max(orientation_errors))
            ),
        },
        "medoid": {
            "stem": medoid.stem,
            "contact_pose_wxyz_xyz": medoid.contact_pose.tolist(),
            "preclose_pose_wxyz_xyz": medoid.preclose_pose.tolist(),
            "close_frame": medoid.close_frame,
            "preclose_frame": medoid.preclose_frame,
            "proof_frame": medoid.proof_frame,
            "release_frame": medoid.release_frame,
        },
        "relative_pull_trajectory": relative,
    }


def retarget_relative_pose(
    contact_pose_wxyz_xyz,
    relative_pose_wxyz_xyz,
) -> np.ndarray:
    """Apply a demonstrated contact-frame offset to a live contact pose."""

    contact = mink.SE3(np.asarray(contact_pose_wxyz_xyz, dtype=float).reshape(7))
    relative = mink.SE3(np.asarray(relative_pose_wxyz_xyz, dtype=float).reshape(7))
    return np.asarray((contact @ relative).parameters(), dtype=float)


def retarget_relative_trajectory(
    contact_pose_wxyz_xyz,
    relative_trajectory: Sequence[dict],
    *,
    first_frame: int | None = None,
    last_frame: int | None = None,
) -> list[dict]:
    """Retarget a frame-bounded demonstration segment to live contact.

    Returned timestamps start strictly after zero.  That makes the result
    directly streamable while preserving the demonstration's timing between
    samples.  Frame bounds are inclusive; the first selected pose is retained
    as a spatial target but receives one control-period of lead time.
    """

    selected = [
        sample
        for sample in relative_trajectory
        if (first_frame is None or int(sample["frame"]) >= int(first_frame))
        and (last_frame is None or int(sample["frame"]) <= int(last_frame))
    ]
    if not selected:
        raise ValueError("relative trajectory segment is empty")
    source_start_s = float(selected[0]["t_s"])
    result = []
    for index, sample in enumerate(selected, start=1):
        elapsed = float(sample["t_s"]) - source_start_s
        result.append(
            {
                "frame": int(sample["frame"]),
                "t_s": max(index / 30.0, elapsed + 1.0 / 30.0),
                "pose_wxyz_xyz": retarget_relative_pose(
                    contact_pose_wxyz_xyz,
                    sample["pose_wxyz_xyz"],
                ).tolist(),
                "gripper": float(sample["gripper"]),
            }
        )
    return result
