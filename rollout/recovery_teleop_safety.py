"""Torque stopping for human recovery teleoperation.

The Quest jump filter rejects discontinuous Cartesian commands.  This module
independently monitors measured joint torque and latches an affected arm into a
measured joint hold.  It deliberately contains no workspace envelope: absence
from a demonstration is not evidence that a Cartesian location is unsafe.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

def extend_fallback_threshold_for_stationary_pose(
    rpc: Any,
    thresholds: dict[str, np.ndarray],
    *,
    arm: str,
    sample_count: int = 25,
    sample_interval_s: float = 0.01,
    margin: float = 1.20,
) -> dict[str, Any]:
    """Extend a mirrored fallback envelope to cover the current static pose.

    Absolute Piper torque contains a pose-dependent gravity/load component.
    A threshold copied from the other arm can therefore reject a harmless
    static pose.  This adjustment is only for an explicitly selected fallback;
    dedicated per-arm calibration must be used unchanged.
    """

    if arm not in thresholds:
        raise ValueError(f"unknown arm {arm!r}")
    if sample_count < 5 or not 1.0 < margin <= 1.5:
        raise ValueError("invalid stationary fallback calibration settings")
    samples = []
    for index in range(sample_count):
        torque = np.asarray(
            getattr(rpc, f"get_{arm}_joint_torque")(),
            dtype=float,
        )
        if torque.shape != (6,) or not np.all(np.isfinite(torque)):
            raise ValueError(f"invalid stationary {arm} torque sample")
        samples.append(np.abs(torque))
        if index + 1 < sample_count and sample_interval_s > 0.0:
            time.sleep(sample_interval_s)
    observed = np.max(np.stack(samples), axis=0)
    original = np.asarray(thresholds[arm], dtype=float).copy()
    adjusted = np.maximum(original, observed * margin)
    thresholds[arm] = adjusted
    return {
        "arm": arm,
        "sample_count": sample_count,
        "sample_interval_s": sample_interval_s,
        "margin": margin,
        "observed_max_abs_nm": observed.tolist(),
        "original_threshold_nm": original.tolist(),
        "adjusted_threshold_nm": adjusted.tolist(),
        "changed_joints": np.flatnonzero(adjusted > original).astype(int).tolist(),
    }


class RecoveryTorqueGuard:
    """Monitor each arm independently and hold on sustained excess torque."""

    def __init__(
        self,
        rpc: Any,
        thresholds: dict[str, np.ndarray],
        *,
        consecutive_samples: int = 5,
        preview_time_s: float = 0.05,
        residual_fraction: float = 0.30,
        residual_floor_nm: float = 0.10,
        residual_ceiling_nm: float = 0.30,
        residual_duration_s: float = 0.35,
        baseline_slew_nm_per_s: float = 0.50,
        hard_limit_multiplier: float = 2.0,
        enforce: bool = True,
        warning_interval_s: float = 2.0,
        audit_path: str | Path | None = None,
        provenance: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.rpc = rpc
        self.thresholds = {
            arm: np.asarray(value, dtype=float).copy()
            for arm, value in thresholds.items()
        }
        if set(self.thresholds) != {"left", "right"}:
            raise ValueError("left and right torque thresholds are required")
        if any(
            value.shape != (6,)
            or not np.all(np.isfinite(value))
            or np.any(value <= 0.0)
            for value in self.thresholds.values()
        ):
            raise ValueError("torque thresholds must be finite positive shape-(6,) arrays")
        self.consecutive_samples = int(consecutive_samples)
        self.preview_time_s = float(preview_time_s)
        self.residual_fraction = float(residual_fraction)
        self.residual_floor_nm = float(residual_floor_nm)
        self.residual_ceiling_nm = float(residual_ceiling_nm)
        self.residual_duration_s = float(residual_duration_s)
        self.baseline_slew_nm_per_s = float(baseline_slew_nm_per_s)
        self.hard_limit_multiplier = float(hard_limit_multiplier)
        self.enforce = bool(enforce)
        self.warning_interval_s = float(warning_interval_s)
        self.clock = clock
        if (
            self.consecutive_samples < 1
            or not 0.0 < self.preview_time_s <= 0.25
            or not 0.0 < self.residual_fraction <= 1.0
            or not 0.0 < self.residual_floor_nm <= self.residual_ceiling_nm
            or not 0.05 <= self.residual_duration_s <= 2.0
            or not 0.0 < self.baseline_slew_nm_per_s <= 2.0
            or not 1.0 < self.hard_limit_multiplier <= 5.0
            or not 0.1 <= self.warning_interval_s <= 60.0
        ):
            raise ValueError("invalid recovery torque guard configuration")
        self.residual_limits = {
            arm: np.clip(
                values * self.residual_fraction,
                self.residual_floor_nm,
                self.residual_ceiling_nm,
            )
            for arm, values in self.thresholds.items()
        }
        self.audit_path = Path(audit_path).resolve() if audit_path else None
        self.provenance = dict(provenance or {})
        self.latched: dict[str, dict[str, Any]] = {}
        self._baseline: dict[str, np.ndarray] = {}
        self._last_check_s: dict[str, float] = {}
        self._residual_started_s: dict[str, np.ndarray] = {}
        self._hard_counts: dict[str, np.ndarray] = {}
        self._last_warning_s: dict[str, float] = {
            "left": float("-inf"),
            "right": float("-inf"),
        }
        for arm in ("left", "right"):
            self.reset(arm)
        self._audit(
            {
                "event": "torque_guard_initialized",
                "thresholds_nm": {
                    arm: values.tolist()
                    for arm, values in self.thresholds.items()
                },
                "consecutive_samples": self.consecutive_samples,
                "residual_limits_nm": {
                    arm: values.tolist()
                    for arm, values in self.residual_limits.items()
                },
                "residual_duration_s": self.residual_duration_s,
                "baseline_slew_nm_per_s": self.baseline_slew_nm_per_s,
                "hard_limit_multiplier": self.hard_limit_multiplier,
                "enforce": self.enforce,
                "provenance": self.provenance,
            }
        )

    def reset(self, arm: str) -> None:
        """Reset pre-trip strike counts when an arm is newly engaged."""

        if arm not in self.thresholds:
            raise ValueError(f"unknown arm {arm!r}")
        if arm in self.latched:
            return
        torque = np.asarray(
            getattr(self.rpc, f"get_{arm}_joint_torque")(),
            dtype=float,
        )
        if torque.shape != (6,) or not np.all(np.isfinite(torque)):
            self._baseline.pop(arm, None)
        else:
            self._baseline[arm] = torque.copy()
        self._last_check_s[arm] = self.clock()
        self._residual_started_s[arm] = np.full(6, np.nan)
        self._hard_counts[arm] = np.zeros(6, dtype=int)

    def check(self, arm: str, *, torque_sample: Any | None = None) -> bool:
        """Return True when safe; otherwise latch a measured joint hold."""

        if arm in self.latched:
            return False
        try:
            now = self.clock()
            torque = np.asarray(
                getattr(self.rpc, f"get_{arm}_joint_torque")()
                if torque_sample is None else torque_sample,
                dtype=float,
            )
            if torque.shape != (6,) or not np.all(np.isfinite(torque)):
                raise ValueError("invalid torque sample")
            baseline = self._baseline.get(arm)
            if baseline is None:
                raise ValueError("invalid engagement torque baseline")
            dt = float(np.clip(now - self._last_check_s[arm], 0.0, 0.2))
            self._last_check_s[arm] = now
            maximum_baseline_step = self.baseline_slew_nm_per_s * dt
            baseline += np.clip(
                torque - baseline,
                -maximum_baseline_step,
                maximum_baseline_step,
            )
            residual = np.abs(torque - baseline)
            residual_over = residual > self.residual_limits[arm]
            started = self._residual_started_s[arm]
            started = np.where(
                residual_over & np.isnan(started),
                now,
                np.where(residual_over, started, np.nan),
            )
            self._residual_started_s[arm] = started
            residual_hit = np.flatnonzero(
                residual_over & ((now - started) >= self.residual_duration_s)
            )

            hard_limits = self.thresholds[arm] * self.hard_limit_multiplier
            hard_over = np.abs(torque) > hard_limits
            hard_counts = np.where(
                hard_over,
                self._hard_counts[arm] + 1,
                0,
            )
            self._hard_counts[arm] = hard_counts
            hard_hit = np.flatnonzero(
                hard_counts >= self.consecutive_samples
            )

            trip = None
            if hard_hit.size:
                joint = int(hard_hit[0])
                trip = {
                    "arm": arm,
                    "reason": "hard_absolute_torque",
                    "joint": joint,
                    "torque": float(abs(torque[joint])),
                    "threshold": float(hard_limits[joint]),
                    "consecutive_samples": int(hard_counts[joint]),
                    "baseline_nm": baseline.tolist(),
                    "residual_nm": residual.tolist(),
                }
            elif residual_hit.size:
                joint = int(residual_hit[0])
                trip = {
                    "arm": arm,
                    "reason": "sustained_torque_residual",
                    "joint": joint,
                    "torque": float(abs(torque[joint])),
                    "threshold": float(self.residual_limits[arm][joint]),
                    "residual_duration_s": float(now - started[joint]),
                    "baseline_nm": baseline.tolist(),
                    "residual_nm": residual.tolist(),
                }
            safe = trip is None
        except Exception as exc:
            torque = np.asarray([], dtype=float)
            safe = False
            trip = {
                "arm": arm,
                "reason": f"torque read failed: {type(exc).__name__}: {exc}",
            }
        if safe:
            return True
        if not self.enforce:
            now = self.clock()
            record = {
                **dict(trip or {"arm": arm}),
                "arm": arm,
                "torque_nm": torque.tolist(),
                "hold_sent": False,
                "observer_only": True,
            }
            if now - self._last_warning_s[arm] >= self.warning_interval_s:
                self._audit({"event": "torque_warning", "warning": record})
                self._last_warning_s[arm] = now
            self.reset(arm)
            return True
        self._latch_hold(arm, torque, dict(trip or {"arm": arm}))
        return False

    def _latch_hold(
        self,
        arm: str,
        torque: np.ndarray,
        trip: dict[str, Any],
    ) -> None:
        hold_sent = False
        hold_error = None
        qpos = None
        try:
            qpos = np.asarray(
                getattr(self.rpc, f"get_{arm}_joint_positions")(),
                dtype=float,
            )
            if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
                raise ValueError("measured qpos is invalid")
            getattr(self.rpc, f"set_{arm}_joint_target")(
                qpos,
                preview_time=self.preview_time_s,
            )
            hold_sent = True
        except Exception as exc:
            hold_error = f"{type(exc).__name__}: {exc}"
        record = {
            **trip,
            "arm": arm,
            "torque_nm": torque.tolist(),
            "measured_qpos_rad": None if qpos is None else qpos.tolist(),
            "hold_sent": hold_sent,
            "hold_error": hold_error,
        }
        self.latched[arm] = record
        self._audit({"event": "torque_stop", "trip": record})

    def _audit(self, record: dict[str, Any]) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
