"""Hardware-independent demo-relative visual servo primitives.

The controller in this module deliberately does not turn pixels directly into
robot motion by prompting a vision model.  A task adapter estimates an object
frame, a demonstration supplies the desired object-to-end-effector relation,
and the fine image Jacobian is estimated from *measured* end-effector motion.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {result}")
    return result


@dataclass(frozen=True)
class StampedObservation:
    """Images and robot state which refer to the same capture instant."""

    timestamp: float
    ee_pose: np.ndarray
    joint_positions: np.ndarray
    gripper_ratio: float
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    depths: Mapping[str, np.ndarray] = field(default_factory=dict)
    gripper_pressure: float | None = None
    torque: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "ee_pose", _array(self.ee_pose, (7,), "ee_pose"))
        object.__setattr__(
            self, "joint_positions", _array(self.joint_positions, (6,), "joint_positions")
        )
        if not np.isfinite(self.timestamp) or not np.isfinite(self.gripper_ratio):
            raise ValueError("timestamp and gripper_ratio must be finite")

    def is_fresh(self, now: float | None = None, max_age_s: float = 0.5) -> bool:
        now = time.time() if now is None else now
        age = now - self.timestamp
        return -0.05 <= age <= max_age_s


@dataclass(frozen=True)
class ObjectEstimate:
    position_m: np.ndarray
    timestamp: float
    confidence: float
    yaw_rad: float | None = None
    source: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(
            self, "position_m", _array(self.position_m, (3,), "object position")
        )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class FineObservation:
    feature: np.ndarray
    confidence: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        feature = np.asarray(self.feature, dtype=float).reshape(-1)
        if feature.size != 2 or not np.all(np.isfinite(feature)):
            raise ValueError("fine feature must be a finite image point")
        object.__setattr__(self, "feature", feature)


class TaskAdapter(Protocol):
    """Task-specific perception and grasp verification."""

    def detect_object(self, observation: StampedObservation) -> ObjectEstimate | None:
        ...

    def fine_observation(self, observation: StampedObservation) -> FineObservation | None:
        ...

    def check_success(self, observation: StampedObservation) -> tuple[bool, Mapping[str, Any]]:
        ...


@dataclass(frozen=True)
class ManipulationTemplate:
    """A successful gripper pose expressed relative to an observed task frame.

    ``tracked_translation_axes`` selects which object translations are copied to
    the goal.  A circular lid on a table uses XY only; its arbitrary yaw is not
    allowed to rotate the wrist.
    """

    reference_object_position_m: np.ndarray
    goal_ee_pose: np.ndarray
    pregrasp_offset_m: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.010])
    )
    tracked_translation_axes: np.ndarray = field(
        default_factory=lambda: np.array([True, True, False])
    )
    fine_feature_goal: np.ndarray | None = None
    empty_close_ratio: float = 0.01
    active_view_ee_pose: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "reference_object_position_m",
            _array(self.reference_object_position_m, (3,), "reference object position"),
        )
        object.__setattr__(
            self, "goal_ee_pose", _array(self.goal_ee_pose, (7,), "goal ee pose")
        )
        object.__setattr__(
            self,
            "pregrasp_offset_m",
            _array(self.pregrasp_offset_m, (3,), "pregrasp offset"),
        )
        axes = np.asarray(self.tracked_translation_axes, dtype=bool)
        if axes.shape != (3,):
            raise ValueError("tracked_translation_axes must have shape (3,)")
        object.__setattr__(self, "tracked_translation_axes", axes)
        if self.fine_feature_goal is not None:
            object.__setattr__(
                self,
                "fine_feature_goal",
                _array(self.fine_feature_goal, (2,), "fine feature goal"),
            )
        if self.active_view_ee_pose is not None:
            object.__setattr__(
                self,
                "active_view_ee_pose",
                _array(self.active_view_ee_pose, (7,), "active view ee pose"),
            )

    def target_pose(self, estimate: ObjectEstimate, pregrasp: bool = False) -> np.ndarray:
        target = self.goal_ee_pose.copy()
        delta = estimate.position_m - self.reference_object_position_m
        target[4:7] += np.where(self.tracked_translation_axes, delta, 0.0)
        if pregrasp:
            target[4:7] += self.pregrasp_offset_m
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ManipulationTemplate":
        cfg = json.loads(Path(path).read_text())
        return cls(
            reference_object_position_m=cfg["reference_object_position_m"],
            goal_ee_pose=cfg["goal_ee_pose_wxyz_xyz"],
            pregrasp_offset_m=cfg.get("pregrasp_offset_m", [0.0, 0.0, 0.010]),
            tracked_translation_axes=cfg.get(
                "tracked_translation_axes", [True, True, False]
            ),
            fine_feature_goal=cfg.get("fine_feature_goal_px"),
            empty_close_ratio=float(cfg.get("empty_close_ratio", 0.01)),
            active_view_ee_pose=cfg.get("active_view_ee_pose_wxyz_xyz"),
        )

    def save(self, path: str | Path) -> None:
        cfg = {
            "version": 1,
            "reference_object_position_m": self.reference_object_position_m.tolist(),
            "goal_ee_pose_wxyz_xyz": self.goal_ee_pose.tolist(),
            "pregrasp_offset_m": self.pregrasp_offset_m.tolist(),
            "tracked_translation_axes": self.tracked_translation_axes.tolist(),
            "fine_feature_goal_px": (
                None if self.fine_feature_goal is None else self.fine_feature_goal.tolist()
            ),
            "empty_close_ratio": self.empty_close_ratio,
            "active_view_ee_pose_wxyz_xyz": (
                None
                if self.active_view_ee_pose is None
                else self.active_view_ee_pose.tolist()
            ),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2) + "\n")


class ServoPhase(str, Enum):
    OBSERVE = "observe"
    COARSE_MOVE = "coarse_move"
    REOBSERVE = "reobserve"
    WRIST_SERVO = "wrist_servo"
    CONTACT_CONFIRMATION = "contact_confirmation"
    DESCEND = "descend"
    CLOSE_VERIFY = "close_verify"
    RECOVER = "recover"
    COMPLETE = "complete"


class ServoAction(str, Enum):
    HOLD = "hold"
    MOVE = "move"
    CLOSE = "close"
    OPEN = "open"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ServoDecision:
    phase: ServoPhase
    action: ServoAction
    reason: str
    target_pose: np.ndarray | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServoConfig:
    max_observation_age_s: float = 0.5
    min_object_confidence: float = 0.65
    coarse_tolerance_m: float = 0.004
    # Head-camera pose estimates jitter by roughly 4--5 mm even while the lid
    # is stationary.  Keep this above that noise floor, while still resetting
    # after a real post-contact lid displacement.
    object_motion_reset_m: float = 0.006
    fine_tolerance_px: float = 4.0
    fine_probe_m: float = 0.008
    fine_max_step_m: float = 0.008
    fine_z_tolerance_m: float = 0.005
    fine_orientation_tolerance_deg: float = 5.0
    fine_max_excursion_m: float = 0.030
    descend_step_m: float = 0.002
    contact_tolerance_m: float = 0.0015
    require_contact_confirmation: bool = True


class MeasuredImageJacobian:
    """Estimate d(feature_px)/d(actual_robot_xy) from real observations."""

    def __init__(self, damping: float = 1e-3, max_condition: float = 100.0):
        self.damping = damping
        self.max_condition = max_condition
        self._motions: list[np.ndarray] = []
        self._features: list[np.ndarray] = []
        self.matrix: np.ndarray | None = None

    def reset(self) -> None:
        self._motions.clear()
        self._features.clear()
        self.matrix = None

    def update(self, actual_delta_xy, feature_delta, object_moved: bool = False) -> bool:
        if object_moved:
            self.reset()
            return False
        motion = _array(actual_delta_xy, (2,), "actual xy delta")
        feature = _array(feature_delta, (2,), "feature delta")
        if np.linalg.norm(motion) < 2e-4:
            return False
        self._motions.append(motion)
        self._features.append(feature)
        # Image Jacobians are local. Old samples from another wrist pose are
        # actively harmful, even when their least-squares residual looks small.
        if len(self._motions) > 4:
            self._motions.pop(0)
            self._features.pop(0)
        x = np.stack(self._motions, axis=1)
        if np.linalg.matrix_rank(x) < 2:
            return False
        candidate = np.stack(self._features, axis=1) @ np.linalg.pinv(x)
        condition = float(np.linalg.cond(candidate))
        if not np.all(np.isfinite(candidate)) or condition > self.max_condition:
            return False
        self.matrix = candidate
        return True

    @property
    def ready(self) -> bool:
        return self.matrix is not None

    def solve(self, feature_error, max_step_m: float) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("image Jacobian is not observable yet")
        error = _array(feature_error, (2,), "feature error")
        lhs = self.matrix.T @ self.matrix + self.damping * np.eye(2)
        step = np.linalg.solve(lhs, self.matrix.T @ error)
        norm = float(np.linalg.norm(step))
        if norm > max_step_m:
            step *= max_step_m / norm
        return step


def trust_region_step(error_m: float) -> float:
    """Coarse-to-fine Cartesian step schedule."""

    if error_m > 0.030:
        return 0.030
    if error_m > 0.010:
        return 0.010
    return 0.005


def bounded_pose_step(current_pose, target_pose, max_translation_m: float) -> np.ndarray:
    current = _array(current_pose, (7,), "current pose")
    target = _array(target_pose, (7,), "target pose")
    result = target.copy()
    delta = target[4:7] - current[4:7]
    norm = float(np.linalg.norm(delta))
    if norm > max_translation_m:
        result[4:7] = current[4:7] + delta * (max_translation_m / norm)
    return result


def quaternion_error_deg(q0, q1) -> float:
    q0 = _array(q0, (4,), "quaternion").copy()
    q1 = _array(q1, (4,), "quaternion").copy()
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.clip(abs(q0 @ q1), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


class DemoRelativeServo:
    """Pure state machine; a caller executes the returned decisions."""

    def __init__(
        self,
        adapter: TaskAdapter,
        template: ManipulationTemplate,
        config: ServoConfig | None = None,
    ):
        self.adapter = adapter
        self.template = template
        self.config = config or ServoConfig()
        self.phase = ServoPhase.OBSERVE
        self.object_estimate: ObjectEstimate | None = None
        self.jacobian = MeasuredImageJacobian()
        self._last_fine_feature: np.ndarray | None = None
        self._last_fine_xy: np.ndarray | None = None
        self._probe_axis = 0
        self._close_was_commanded = False
        self._wrist_orientation: np.ndarray | None = None
        self._wrist_z_m: float | None = None
        self._previous_fine_error_norm: float | None = None
        self._last_fine_was_control = False
        self._fine_origin_xy: np.ndarray | None = None

    def confirm_contact(self) -> None:
        if self.phase != ServoPhase.CONTACT_CONFIRMATION:
            raise RuntimeError(f"cannot confirm contact during {self.phase.value}")
        self.phase = ServoPhase.DESCEND

    def _decision(self, action, reason, target=None, **diagnostics):
        return ServoDecision(self.phase, action, reason, target, diagnostics)

    def _fresh(self, observation, now):
        return observation.is_fresh(now, self.config.max_observation_age_s)

    def step(
        self, observation: StampedObservation, now: float | None = None
    ) -> ServoDecision:
        now = time.time() if now is None else now
        if not self._fresh(observation, now):
            return self._decision(ServoAction.HOLD, "stale observation")

        if self.phase == ServoPhase.OBSERVE:
            estimate = self.adapter.detect_object(observation)
            if estimate is None or estimate.confidence < self.config.min_object_confidence:
                if self.template.active_view_ee_pose is not None:
                    distance = float(
                        np.linalg.norm(
                            self.template.active_view_ee_pose[4:7]
                            - observation.ee_pose[4:7]
                        )
                    )
                    if distance > self.config.coarse_tolerance_m:
                        target = bounded_pose_step(
                            observation.ee_pose,
                            self.template.active_view_ee_pose,
                            trust_region_step(distance),
                        )
                        return self._decision(
                            ServoAction.MOVE,
                            "move to unoccluded active view",
                            target,
                            error_m=distance,
                        )
                return self._decision(ServoAction.HOLD, "object estimate unavailable")
            self.object_estimate = estimate
            self.jacobian.reset()
            self._probe_axis = 0
            self._last_fine_feature = None
            self._last_fine_xy = None
            self._wrist_orientation = None
            self._wrist_z_m = None
            self._previous_fine_error_norm = None
            self._last_fine_was_control = False
            self._fine_origin_xy = None
            self.phase = ServoPhase.COARSE_MOVE

        if self.phase == ServoPhase.COARSE_MOVE:
            target = self.template.target_pose(self.object_estimate, pregrasp=True)
            error = float(np.linalg.norm(target[4:7] - observation.ee_pose[4:7]))
            if error > self.config.coarse_tolerance_m:
                step = trust_region_step(error)
                bounded = bounded_pose_step(observation.ee_pose, target, step)
                return self._decision(
                    ServoAction.MOVE, "move toward demo-relative pregrasp", bounded, error_m=error
                )
            self.phase = ServoPhase.REOBSERVE

        if self.phase == ServoPhase.REOBSERVE:
            estimate = self.adapter.detect_object(observation)
            if estimate is None or estimate.confidence < self.config.min_object_confidence:
                return self._decision(ServoAction.HOLD, "reobservation unavailable")
            moved = float(np.linalg.norm(estimate.position_m - self.object_estimate.position_m))
            self.object_estimate = estimate
            if moved > self.config.object_motion_reset_m:
                self.jacobian.reset()
                self.phase = ServoPhase.COARSE_MOVE
                return self._decision(
                    ServoAction.HOLD, "object moved; recomputed target", object_motion_m=moved
                )
            self.phase = (
                ServoPhase.WRIST_SERVO
                if self.template.fine_feature_goal is not None
                else ServoPhase.CONTACT_CONFIRMATION
            )
            if self.phase == ServoPhase.WRIST_SERVO:
                # Never feed compensated/undershot wrist orientation back as
                # the next target. Hold successful orientation and pregrasp Z.
                self._wrist_orientation = self.template.goal_ee_pose[:4].copy()
                self._wrist_z_m = float(
                    self.template.target_pose(
                        self.object_estimate, pregrasp=True
                    )[6]
                )
                self._fine_origin_xy = observation.ee_pose[4:6].copy()

        if self.phase == ServoPhase.WRIST_SERVO:
            excursion = float(
                np.linalg.norm(observation.ee_pose[4:6] - self._fine_origin_xy)
            )
            if excursion > self.config.fine_max_excursion_m:
                self.phase = ServoPhase.OBSERVE
                self.jacobian.reset()
                return self._decision(
                    ServoAction.HOLD,
                    "wrist servo excursion exceeded; reobserve from active view",
                    excursion_m=excursion,
                )
            z_error = abs(float(observation.ee_pose[6] - self._wrist_z_m))
            orientation_error = quaternion_error_deg(
                observation.ee_pose[:4], self._wrist_orientation
            )
            if (
                z_error > self.config.fine_z_tolerance_m
                or orientation_error > self.config.fine_orientation_tolerance_deg
            ):
                target = observation.ee_pose.copy()
                target[:4] = self._wrist_orientation
                target[6] = self._wrist_z_m
                return self._decision(
                    ServoAction.MOVE,
                    "stabilize wrist orientation and height before probing",
                    target,
                    z_error_m=z_error,
                    orientation_error_deg=orientation_error,
                )
            fine = self.adapter.fine_observation(observation)
            if fine is None or fine.confidence < self.config.min_object_confidence:
                self.phase = ServoPhase.OBSERVE
                self.jacobian.reset()
                return self._decision(
                    ServoAction.HOLD, "wrist feature unavailable; reobserve from head"
                )

            current_xy = observation.ee_pose[4:6].copy()
            if self._last_fine_feature is not None and self._last_fine_xy is not None:
                self.jacobian.update(
                    current_xy - self._last_fine_xy,
                    fine.feature - self._last_fine_feature,
                )

            error = self.template.fine_feature_goal - fine.feature
            error_norm = float(np.linalg.norm(error))
            if (
                self._last_fine_was_control
                and self._previous_fine_error_norm is not None
                and error_norm > 1.20 * self._previous_fine_error_norm
            ):
                self.jacobian.reset()
                self._probe_axis = 0
            if np.linalg.norm(error) <= self.config.fine_tolerance_px:
                self.phase = ServoPhase.CONTACT_CONFIRMATION
            else:
                target = observation.ee_pose.copy()
                target[:4] = self._wrist_orientation
                target[6] = self._wrist_z_m
                if not self.jacobian.ready:
                    axis = self._probe_axis % 2
                    direction = 1.0 if self._probe_axis < 2 else -1.0
                    target[4 + axis] += direction * self.config.fine_probe_m
                    self._probe_axis += 1
                    reason = f"measure wrist Jacobian axis {axis}"
                    self._last_fine_was_control = False
                else:
                    target[4:6] += self.jacobian.solve(
                        error, self.config.fine_max_step_m
                    )
                    reason = "reduce measured wrist feature error"
                    self._last_fine_was_control = True
                self._last_fine_feature = fine.feature.copy()
                self._last_fine_xy = current_xy
                self._previous_fine_error_norm = error_norm
                return self._decision(
                    ServoAction.MOVE,
                    reason,
                    target,
                    feature_error_px=error.tolist(),
                    jacobian_ready=self.jacobian.ready,
                )

        if self.phase == ServoPhase.CONTACT_CONFIRMATION:
            if self.config.require_contact_confirmation:
                return self._decision(
                    ServoAction.HOLD, "contact confirmation required"
                )
            self.phase = ServoPhase.DESCEND

        if self.phase == ServoPhase.DESCEND:
            target = self.template.target_pose(self.object_estimate, pregrasp=False)
            error = float(np.linalg.norm(target[4:7] - observation.ee_pose[4:7]))
            if error > self.config.contact_tolerance_m:
                bounded = bounded_pose_step(
                    observation.ee_pose, target, self.config.descend_step_m
                )
                return self._decision(
                    ServoAction.MOVE, "descend toward demonstrated contact", bounded, error_m=error
                )
            self.phase = ServoPhase.CLOSE_VERIFY
            self._close_was_commanded = False

        if self.phase == ServoPhase.CLOSE_VERIFY:
            if not self._close_was_commanded:
                self._close_was_commanded = True
                return self._decision(ServoAction.CLOSE, "close gripper")
            success, diagnostics = self.adapter.check_success(observation)
            if success:
                self.phase = ServoPhase.COMPLETE
                return self._decision(
                    ServoAction.COMPLETE, "grasp verified", **dict(diagnostics)
                )
            self.phase = ServoPhase.RECOVER
            return self._decision(
                ServoAction.OPEN, "empty close; reobserve object", **dict(diagnostics)
            )

        if self.phase == ServoPhase.RECOVER:
            self.phase = ServoPhase.OBSERVE
            self.object_estimate = None
            self.jacobian.reset()
            return self._decision(ServoAction.HOLD, "recovery complete; observe again")

        return self._decision(ServoAction.COMPLETE, "already complete")
