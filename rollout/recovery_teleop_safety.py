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
from typing import Any

import numpy as np

from rollout.torque_safety import TorqueWatchdog


class RecoveryTorqueGuard:
    """Monitor each arm independently and hold on sustained excess torque."""

    def __init__(
        self,
        rpc: Any,
        thresholds: dict[str, np.ndarray],
        *,
        consecutive_samples: int = 5,
        preview_time_s: float = 0.05,
        audit_path: str | Path | None = None,
        provenance: dict[str, Any] | None = None,
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
        if self.consecutive_samples < 1 or not 0.0 < self.preview_time_s <= 0.25:
            raise ValueError("invalid recovery torque guard configuration")
        self.audit_path = Path(audit_path).resolve() if audit_path else None
        self.provenance = dict(provenance or {})
        self.latched: dict[str, dict[str, Any]] = {}
        self._watchdogs: dict[str, TorqueWatchdog] = {}
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
                "provenance": self.provenance,
            }
        )

    def reset(self, arm: str) -> None:
        """Reset pre-trip strike counts when an arm is newly engaged."""

        if arm not in self.thresholds:
            raise ValueError(f"unknown arm {arm!r}")
        if arm in self.latched:
            return
        self._watchdogs[arm] = TorqueWatchdog(
            {arm: self.thresholds[arm]},
            consecutive_limit=self.consecutive_samples,
        )

    def check(self, arm: str) -> bool:
        """Return True when safe; otherwise latch a measured joint hold."""

        if arm in self.latched:
            return False
        try:
            torque = np.asarray(
                getattr(self.rpc, f"get_{arm}_joint_torque")(),
                dtype=float,
            )
            safe = self._watchdogs[arm].check(arm, torque)
            trip = self._watchdogs[arm].tripped
        except Exception as exc:
            torque = np.asarray([], dtype=float)
            safe = False
            trip = {
                "arm": arm,
                "reason": f"torque read failed: {type(exc).__name__}: {exc}",
            }
        if safe:
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
