"""Deterministic, operator-guided lid demonstration collection.

The module deliberately contains no policy inference.  A reviewed successful
episode supplies a grasp pose and a relative post-grasp transport path.  The
operator changes the next attempt with physical Cartesian offsets; deterministic
code then generates hover, one uninterrupted descent, close, verification lift,
transport, and release commands at 30 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


CONTROL_HZ = 30.0
RIGHT_ACTION_POSE = slice(7, 14)
RIGHT_ACTION_GRIPPER = 15


def minimum_jerk(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _finite_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


@dataclass(frozen=True)
class PoseCommand:
    t_s: float
    stage: str
    pose_wxyz_xyz: tuple[float, float, float, float, float, float, float]
    gripper_open_ratio: float

    def pose(self) -> np.ndarray:
        return _finite_vector(self.pose_wxyz_xyz, 7, "pose")


@dataclass(frozen=True)
class BaselineTrajectory:
    task: str
    source: str
    source_sha256: str
    grasp_pose_wxyz_xyz: tuple[float, ...]
    review_pose_wxyz_xyz: tuple[float, ...]
    release_pose_wxyz_xyz: tuple[float, ...]
    post_review: tuple[PoseCommand, ...]
    source_grasp_index: int
    source_review_index: int

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        indices = np.flatnonzero(mask)
        if not len(indices):
            return []
        starts = indices[np.r_[True, np.diff(indices) > 1]]
        ends = indices[np.r_[np.diff(indices) > 1, True]]
        return [(int(first), int(last)) for first, last in zip(starts, ends)]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        task: str,
        close_threshold: float = 0.35,
        open_threshold: float = 0.75,
        review_lift_m: float = 0.015,
    ) -> "BaselineTrajectory":
        if task not in {"lid_open", "lid_close"}:
            raise ValueError("task must be lid_open or lid_close")
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        import hashlib

        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with h5py.File(path, "r") as source:
            timestamps = np.asarray(source["active_timestamps"], dtype=float)
            measured_xyz = np.asarray(source["right_ee_pos"], dtype=float)
            measured_quat = np.asarray(source["right_ee_quat"], dtype=float)
            commanded = np.asarray(source["commanded_target_quat16"], dtype=float)

        count = len(timestamps)
        if (
            count < 3
            or measured_xyz.shape != (count, 3)
            or measured_quat.shape != (count, 4)
            or commanded.shape != (count, 16)
        ):
            raise ValueError("baseline HDF5 arrays have incompatible shapes")
        gripper = commanded[:, RIGHT_ACTION_GRIPPER]
        pose = np.concatenate([measured_quat, measured_xyz], axis=1)
        valid = (
            np.isfinite(timestamps)
            & np.isfinite(pose).all(axis=1)
            & np.isfinite(gripper)
        )
        if np.count_nonzero(valid) < 3:
            raise ValueError("baseline has too few finite command samples")

        close_runs = cls._runs(valid & (gripper <= close_threshold))
        if not close_runs:
            raise ValueError("baseline contains no close run")
        # A time-agnostic policy can make transient close/open attempts.  The
        # final close run is the one that leads to the demonstrated transport
        # and final release; a no-release lid_open trace has one sustained run.
        grasp_index = close_runs[-1][0]
        grasp_xyz = measured_xyz[grasp_index]
        after = np.arange(count) >= grasp_index
        displacement = np.linalg.norm(measured_xyz - grasp_xyz, axis=1)
        lifted = (
            valid
            & after
            & (displacement >= review_lift_m)
            & (measured_xyz[:, 2] >= grasp_xyz[2] + min(0.005, review_lift_m))
        )
        candidates = np.flatnonzero(lifted)
        if not len(candidates):
            candidates = np.flatnonzero(valid & after & (displacement >= review_lift_m))
        if not len(candidates):
            raise ValueError("baseline never transports far enough for grasp review")
        review_index = int(candidates[0])

        final_open_runs = [
            run for run in cls._runs(valid & (gripper >= open_threshold))
            if run[0] > grasp_index
        ]
        release_index = (
            final_open_runs[-1][0]
            if final_open_runs
            else int(np.flatnonzero(valid)[-1])
        )

        post: list[PoseCommand] = []
        t0 = float(timestamps[review_index])
        previous_t = -math.inf
        for index in range(review_index, count):
            if not valid[index]:
                continue
            relative_t = float(timestamps[index] - t0)
            if relative_t <= previous_t:
                continue
            previous_t = relative_t
            post.append(
                PoseCommand(
                    t_s=relative_t,
                    stage="transport",
                    pose_wxyz_xyz=tuple(float(v) for v in pose[index]),
                    gripper_open_ratio=float(np.clip(gripper[index], 0.0, 1.0)),
                )
            )
        if len(post) < 2:
            raise ValueError("baseline has no usable post-grasp transport")
        # lid_open success was historically stopped while still holding.  A
        # complete demonstration ends with a smooth explicit release.
        if task == "lid_open" and post[-1].gripper_open_ratio < open_threshold:
            last = post[-1]
            duration = 0.8
            start_gripper = last.gripper_open_ratio
            for index in range(1, int(math.ceil(duration * CONTROL_HZ)) + 1):
                fraction = index / math.ceil(duration * CONTROL_HZ)
                post.append(
                    PoseCommand(
                        t_s=last.t_s + fraction * duration,
                        stage="release",
                        pose_wxyz_xyz=last.pose_wxyz_xyz,
                        gripper_open_ratio=float(
                            start_gripper
                            + minimum_jerk(fraction) * (1.0 - start_gripper)
                        ),
                    )
                )
        return cls(
            task=task,
            source=str(path),
            source_sha256=source_hash,
            grasp_pose_wxyz_xyz=tuple(float(v) for v in pose[grasp_index]),
            review_pose_wxyz_xyz=tuple(float(v) for v in pose[review_index]),
            release_pose_wxyz_xyz=tuple(float(v) for v in pose[release_index]),
            post_review=tuple(post),
            source_grasp_index=grasp_index,
            source_review_index=review_index,
        )


def sample_pose_segment(
    start_pose: Sequence[float],
    end_pose: Sequence[float],
    *,
    duration_s: float,
    stage: str,
    gripper_start: float,
    gripper_end: float | None = None,
    start_t_s: float = 0.0,
    control_hz: float = CONTROL_HZ,
) -> list[PoseCommand]:
    start = _finite_vector(start_pose, 7, "start pose")
    end = _finite_vector(end_pose, 7, "end pose")
    if float(np.dot(start[:4], end[:4])) < 0.0:
        end = end.copy()
        end[:4] *= -1.0
    if duration_s <= 0 or control_hz <= 0:
        raise ValueError("duration and control rate must be positive")
    count = max(1, int(math.ceil(duration_s * control_hz)))
    g0 = float(np.clip(gripper_start, 0.0, 1.0))
    g1 = g0 if gripper_end is None else float(np.clip(gripper_end, 0.0, 1.0))
    result = []
    # Orientation is intentionally fixed within generated grasp segments.  The
    # reviewed baseline quaternion is used at both endpoints.
    for index in range(1, count + 1):
        fraction = index / count
        blend = minimum_jerk(fraction)
        value = start + blend * (end - start)
        quat_norm = float(np.linalg.norm(value[:4]))
        if quat_norm < 1e-9:
            raise ValueError("interpolated quaternion is degenerate")
        value[:4] /= quat_norm
        result.append(
            PoseCommand(
                t_s=start_t_s + fraction * duration_s,
                stage=stage,
                pose_wxyz_xyz=tuple(float(v) for v in value),
                gripper_open_ratio=float(g0 + blend * (g1 - g0)),
            )
        )
    return result


def build_grasp_prefix(
    measured_start_pose: Sequence[float],
    baseline: BaselineTrajectory,
    correction_xyz_m: Sequence[float],
    *,
    hover_clearance_m: float,
    transit_speed_m_s: float,
    descent_speed_m_s: float,
    close_duration_s: float,
    verification_lift_m: float,
) -> list[PoseCommand]:
    start = _finite_vector(measured_start_pose, 7, "measured start pose")
    correction = _finite_vector(correction_xyz_m, 3, "correction")
    grasp = np.asarray(baseline.grasp_pose_wxyz_xyz, dtype=float).copy()
    grasp[4:7] += correction
    # Preserve the reviewed horizontal gripper orientation for the entire
    # approach, regardless of the measured start wrist orientation.
    hover = grasp.copy()
    hover[6] += float(hover_clearance_m)
    clearance = start.copy()
    clearance[6] = max(start[6], hover[6])
    level_at_clearance = clearance.copy()
    level_at_clearance[:4] = grasp[:4]

    result: list[PoseCommand] = []
    cursor = 0.0

    def append_segment(a, b, duration, stage, g0, g1=None):
        nonlocal cursor
        segment = sample_pose_segment(
            a, b, duration_s=max(duration, 1.0 / CONTROL_HZ), stage=stage,
            gripper_start=g0, gripper_end=g1, start_t_s=cursor,
        )
        result.extend(segment)
        cursor = segment[-1].t_s

    append_segment(
        start, clearance,
        float(np.linalg.norm(clearance[4:7] - start[4:7])) / transit_speed_m_s,
        "clearance", 1.0,
    )
    quaternion_dot = float(abs(np.dot(clearance[:4], grasp[:4])))
    angle = 2.0 * math.acos(float(np.clip(quaternion_dot, -1.0, 1.0)))
    append_segment(
        clearance, level_at_clearance,
        max(angle / 0.6, 0.3),
        "level_at_clearance", 1.0,
    )
    append_segment(
        level_at_clearance, hover,
        float(np.linalg.norm(hover[4:7] - level_at_clearance[4:7])) / transit_speed_m_s,
        "hover_transit", 1.0,
    )
    # Exactly one uninterrupted descent command sequence.  No low XY move is
    # generated after this stage begins.
    append_segment(
        hover, grasp, hover_clearance_m / descent_speed_m_s,
        "continuous_descent", 1.0,
    )
    append_segment(grasp, grasp, close_duration_s, "close", 1.0, 0.0)
    review = grasp.copy()
    review[6] += verification_lift_m
    append_segment(
        grasp, review, verification_lift_m / transit_speed_m_s,
        "vertical_verification_lift", 0.0,
    )
    return result


def rebase_post_grasp(
    baseline: BaselineTrajectory,
    measured_review_pose: Sequence[float],
    *,
    start_t_s: float = 0.0,
) -> list[PoseCommand]:
    anchor = _finite_vector(measured_review_pose, 7, "measured review pose")
    source_anchor = np.asarray(baseline.review_pose_wxyz_xyz, dtype=float)
    result = []
    for command in baseline.post_review:
        pose = command.pose()
        pose[4:7] = anchor[4:7] + (pose[4:7] - source_anchor[4:7])
        result.append(
            PoseCommand(
                t_s=start_t_s + command.t_s,
                stage=command.stage,
                pose_wxyz_xyz=tuple(float(v) for v in pose),
                gripper_open_ratio=command.gripper_open_ratio,
            )
        )
    return result


def build_reposition_commands(
    current_pose: Sequence[float],
    placement_pose: Sequence[float],
    displacement_xyz_m: Sequence[float],
    *,
    lift_m: float = 0.020,
    speed_m_s: float = 0.05,
    close_duration_s: float = 1.2,
    open_duration_s: float = 0.8,
) -> list[PoseCommand]:
    """Regrasp a just-placed lid and move it by a known physical delta."""

    current = _finite_vector(current_pose, 7, "current pose")
    placement = _finite_vector(placement_pose, 7, "placement pose")
    delta = _finite_vector(displacement_xyz_m, 3, "reposition displacement")
    if abs(delta[2]) > 1e-12:
        raise ValueError("reposition sweep must stay on the support plane")
    result: list[PoseCommand] = []
    cursor = 0.0

    def add(a, b, duration, stage, g0, g1=None):
        nonlocal cursor
        segment = sample_pose_segment(
            a, b, duration_s=max(duration, 1.0 / CONTROL_HZ), stage=stage,
            gripper_start=g0, gripper_end=g1, start_t_s=cursor,
        )
        result.extend(segment)
        cursor = result[-1].t_s

    hover = placement.copy(); hover[6] += lift_m
    clearance = current.copy(); clearance[6] = max(current[6], hover[6])
    add(current, clearance, max(np.linalg.norm(clearance[4:7] - current[4:7]) / speed_m_s, 1 / CONTROL_HZ), "reposition_clearance", 1.0)
    add(clearance, hover, max(np.linalg.norm(hover[4:7] - clearance[4:7]) / speed_m_s, 0.3), "reposition_hover", 1.0)
    add(hover, placement, lift_m / speed_m_s, "reposition_descent", 1.0)
    add(placement, placement, close_duration_s, "reposition_close", 1.0, 0.0)
    lifted = placement.copy(); lifted[6] += lift_m
    add(placement, lifted, lift_m / speed_m_s, "reposition_vertical_lift", 0.0)
    shifted = lifted.copy(); shifted[4:7] += delta
    add(lifted, shifted, max(np.linalg.norm(delta) / speed_m_s, 1 / CONTROL_HZ), "reposition_planar", 0.0)
    placed = shifted.copy(); placed[6] -= lift_m
    add(shifted, placed, lift_m / speed_m_s, "reposition_vertical_place", 0.0)
    add(placed, placed, open_duration_s, "reposition_release", 0.0, 1.0)
    retreat = placed.copy(); retreat[6] += lift_m
    add(placed, retreat, lift_m / speed_m_s, "reposition_vertical_retreat", 1.0)
    return result


class CollectionPhase(str, Enum):
    READY = "ready"
    EXECUTING = "executing"
    REVIEW = "review"
    COMPLETING = "completing"
    REPOSITIONING = "repositioning"
    STOPPED = "stopped"


@dataclass
class GuidedLidCycle:
    sweep_right_mm: tuple[int, ...] = (
        0, 10, 20, 30, 20, 10, 0, -10, -20, -30, -20, -10,
    )
    task: str = "lid_open"
    phase: CollectionPhase = CollectionPhase.READY
    sweep_index: int = 0
    auto_enabled: bool = False
    attempt_correction_m: np.ndarray | None = None
    active_attempt_correction_m: np.ndarray | None = None
    successful_correction_by_task_m: dict[str, np.ndarray] = field(
        default_factory=dict
    )
    attempt_index: int = 0

    def __post_init__(self) -> None:
        if not self.sweep_right_mm or self.sweep_right_mm[0] != 0:
            raise ValueError("sweep must begin at the baseline position")
        if any(abs(v) > 30 or v % 10 for v in self.sweep_right_mm):
            raise ValueError("sweep positions must be 10 mm steps within ±30 mm")
        if self.task not in {"lid_open", "lid_close"}:
            raise ValueError("invalid initial task")
        if self.attempt_correction_m is None:
            self.attempt_correction_m = np.zeros(3, dtype=float)
        else:
            self.attempt_correction_m = _finite_vector(
                self.attempt_correction_m, 3, "attempt correction"
            ).copy()
        if self.active_attempt_correction_m is not None:
            self.active_attempt_correction_m = _finite_vector(
                self.active_attempt_correction_m, 3, "active attempt correction"
            ).copy()
        self.successful_correction_by_task_m = {
            task: _finite_vector(value, 3, f"{task} successful correction").copy()
            for task, value in self.successful_correction_by_task_m.items()
        }
        if set(self.successful_correction_by_task_m) - {"lid_open", "lid_close"}:
            raise ValueError("successful correction has an unknown task")

    @property
    def placement_right_mm(self) -> int:
        return int(self.sweep_right_mm[self.sweep_index])

    @property
    def placement_robot_xyz_m(self) -> np.ndarray:
        # Robot frame: +Y is physical left, therefore physical right is -Y.
        return np.array([0.0, -self.placement_right_mm / 1000.0, 0.0])

    @property
    def next_placement_robot_xyz_m(self) -> np.ndarray:
        index = (self.sweep_index + 1) % len(self.sweep_right_mm)
        return np.array([0.0, -self.sweep_right_mm[index] / 1000.0, 0.0])

    def adjust(self, axis: str, delta_mm: float) -> np.ndarray:
        if self.phase not in {CollectionPhase.READY, CollectionPhase.REVIEW}:
            raise RuntimeError("corrections are accepted only while ready or reviewing")
        index = {"x": 0, "y": 1, "z": 2}.get(axis)
        if index is None or not math.isfinite(delta_mm):
            raise ValueError("invalid correction")
        self.attempt_correction_m[index] += float(delta_mm) / 1000.0
        if np.any(np.abs(self.attempt_correction_m) > 0.08 + 1e-12):
            self.attempt_correction_m[index] -= float(delta_mm) / 1000.0
            raise ValueError("correction exceeds ±80 mm")
        return self.attempt_correction_m.copy()

    def start_attempt(self) -> None:
        if self.phase != CollectionPhase.READY:
            raise RuntimeError("attempt can start only from ready")
        self.phase = CollectionPhase.EXECUTING
        self.active_attempt_correction_m = self.attempt_correction_m.copy()
        self.attempt_index += 1

    def enter_review(self) -> None:
        if self.phase != CollectionPhase.EXECUTING:
            raise RuntimeError("review requires an executing attempt")
        self.phase = CollectionPhase.REVIEW

    def fail(self) -> None:
        if self.phase != CollectionPhase.REVIEW:
            raise RuntimeError("failure requires review")
        self.active_attempt_correction_m = None
        self.phase = CollectionPhase.READY

    def succeed(self) -> None:
        if self.phase != CollectionPhase.REVIEW:
            raise RuntimeError("success requires review")
        if self.active_attempt_correction_m is None:
            raise RuntimeError("successful attempt has no frozen correction")
        self.successful_correction_by_task_m[self.task] = (
            self.active_attempt_correction_m.copy()
        )
        self.attempt_correction_m = self.active_attempt_correction_m.copy()
        self.active_attempt_correction_m = None
        self.phase = CollectionPhase.COMPLETING

    def task_complete(self) -> str:
        if self.phase != CollectionPhase.COMPLETING:
            raise RuntimeError("task is not completing")
        if self.task == "lid_open":
            self.task = "lid_close"
            self.attempt_correction_m = self.successful_correction_by_task_m.get(
                self.task, np.zeros(3, dtype=float)
            ).copy()
            self.phase = CollectionPhase.READY
            return "start_lid_close"
        self.phase = CollectionPhase.REPOSITIONING
        return "reposition"

    def reposition_complete(self) -> None:
        if self.phase != CollectionPhase.REPOSITIONING:
            raise RuntimeError("reposition is not active")
        self.sweep_index = (self.sweep_index + 1) % len(self.sweep_right_mm)
        self.task = "lid_open"
        # Depth and any operator-calibrated planar correction remain fixed.
        # The known physical placement displacement is added separately.
        self.attempt_correction_m = self.successful_correction_by_task_m.get(
            "lid_open", np.zeros(3, dtype=float)
        ).copy()
        self.phase = CollectionPhase.READY

    def enable_auto(self) -> None:
        self.auto_enabled = True

    def stop(self) -> None:
        self.phase = CollectionPhase.STOPPED

    def snapshot(self) -> dict:
        return {
            "task": self.task,
            "phase": self.phase.value,
            "auto_enabled": self.auto_enabled,
            "attempt_index": self.attempt_index,
            "attempt_correction_m": self.attempt_correction_m.tolist(),
            "active_attempt_correction_m": (
                None
                if self.active_attempt_correction_m is None
                else self.active_attempt_correction_m.tolist()
            ),
            "successful_correction_by_task_m": {
                task: value.tolist()
                for task, value in self.successful_correction_by_task_m.items()
            },
            "sweep_index": self.sweep_index,
            "placement_right_mm": self.placement_right_mm,
            "placement_robot_xyz_m": self.placement_robot_xyz_m.tolist(),
        }
