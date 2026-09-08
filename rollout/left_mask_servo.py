"""AprilTag-free image-space servoing using a fixed left observer camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MaskServoDecision:
    delta_xy_m: np.ndarray
    error_px: np.ndarray
    jacobian_ready: bool
    reason: str


class LeftMaskServo:
    """Learn the image-to-EE translation Jacobian from measured motion.

    The observer camera can be arbitrarily mounted: no AprilTag or camera
    extrinsic calibration is required.  Only the left-camera mask center,
    the measured right EE XY position, and a teacher image target are used.
    """

    def __init__(
        self,
        target_px,
        *,
        probe_m: float = 0.008,
        max_step_m: float = 0.020,
        damping: float = 1e-3,
    ):
        self.target_px = np.asarray(target_px, dtype=float).reshape(2)
        self.probe_m = float(probe_m)
        self.max_step_m = float(max_step_m)
        self.damping = float(damping)
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._last_xy: np.ndarray | None = None
        self._last_feature: np.ndarray | None = None

    @property
    def jacobian_ready(self) -> bool:
        return len(self._samples) >= 2

    def reset(self) -> None:
        self._samples.clear()
        self._last_xy = None
        self._last_feature = None

    def observe(self, feature_px, ee_xy_m) -> None:
        feature = np.asarray(feature_px, dtype=float).reshape(2)
        xy = np.asarray(ee_xy_m, dtype=float).reshape(2)
        if not np.all(np.isfinite(feature)) or not np.all(np.isfinite(xy)):
            raise ValueError("feature and EE position must be finite")
        if self._last_xy is not None and self._last_feature is not None:
            dxy = xy - self._last_xy
            dfeature = feature - self._last_feature
            if np.linalg.norm(dxy) >= 0.002:
                self._samples.append((dxy, dfeature))
                self._samples = self._samples[-6:]
        self._last_xy = xy.copy()
        self._last_feature = feature.copy()

    def _jacobian(self) -> np.ndarray | None:
        if len(self._samples) < 2:
            return None
        motion = np.vstack([sample[0] for sample in self._samples])
        feature = np.vstack([sample[1] for sample in self._samples])
        if np.linalg.matrix_rank(motion) < 2:
            return None
        # feature_delta = motion_delta @ J.T
        return np.linalg.lstsq(motion, feature, rcond=None)[0].T

    def decide(self, feature_px, ee_xy_m) -> MaskServoDecision:
        self.observe(feature_px, ee_xy_m)
        feature = np.asarray(feature_px, dtype=float).reshape(2)
        error = self.target_px - feature
        if np.linalg.norm(error) <= 3.0:
            return MaskServoDecision(np.zeros(2), error, self.jacobian_ready, "mask aligned")
        jacobian = self._jacobian()
        if jacobian is None:
            axis = len(self._samples) % 2
            delta = np.zeros(2)
            delta[axis] = self.probe_m
            return MaskServoDecision(delta, error, False, f"measure observer Jacobian axis {axis}")
        lhs = jacobian.T @ jacobian + self.damping * np.eye(2)
        delta = np.linalg.solve(lhs, jacobian.T @ error)
        norm = float(np.linalg.norm(delta))
        if norm > self.max_step_m:
            delta *= self.max_step_m / norm
        return MaskServoDecision(delta, error, True, "reduce left-mask image error")
