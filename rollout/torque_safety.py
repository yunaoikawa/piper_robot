"""Joint-torque calibration and sustained-exceedance watchdog."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def torque_stop_enabled_from_config(config: dict) -> bool:
    """Return whether torque telemetry alone may stop robot motion."""

    mode = config.get("motion_torque_policy", "enforce")
    if mode not in {"enforce", "observe_only"}:
        raise ValueError(
            "motion_torque_policy must be 'enforce' or 'observe_only'"
        )
    return mode == "enforce"


@dataclass
class TorqueWatchdog:
    thresholds: dict[str, np.ndarray]
    consecutive_limit: int = 5
    _counts: dict[str, np.ndarray] = field(default_factory=dict)
    tripped: dict | None = None

    def __post_init__(self):
        self.thresholds = {
            arm: np.asarray(values, dtype=float) for arm, values in self.thresholds.items()
        }
        self._counts = {
            arm: np.zeros(values.shape, dtype=int) for arm, values in self.thresholds.items()
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "TorqueWatchdog":
        with open(path) as f:
            cfg = json.load(f)
        return cls(cfg["thresholds"], int(cfg.get("consecutive_samples", 5)))

    def check(self, arm: str, torque) -> bool:
        """Return True while safe; latch details and return False on a trip."""
        if self.tripped is not None:
            return False
        values = np.abs(np.asarray(torque, dtype=float))
        limits = self.thresholds[arm]
        if values.shape != limits.shape or not np.all(np.isfinite(values)):
            self.tripped = {"arm": arm, "reason": "invalid torque sample"}
            return False
        counts = np.where(values > limits, self._counts[arm] + 1, 0)
        self._counts[arm] = counts
        hit = np.flatnonzero(counts >= self.consecutive_limit)
        if hit.size:
            joint = int(hit[0])
            self.tripped = {
                "arm": arm,
                "joint": joint,
                "torque": float(values[joint]),
                "threshold": float(limits[joint]),
                "consecutive_samples": int(counts[joint]),
            }
            return False
        return True


@dataclass
class TorqueCalibrator:
    samples: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, arm: str, torque) -> None:
        values = np.abs(np.asarray(torque, dtype=float))
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid {arm} torque sample: {values}")
        self.samples.setdefault(arm, []).append(values)

    def build_config(self) -> dict:
        thresholds = {}
        stats = {}
        for arm, samples in self.samples.items():
            values = np.stack(samples)
            maximum = values.max(axis=0)
            median = np.median(values, axis=0)
            mad = np.median(np.abs(values - median), axis=0)
            margin = np.maximum(0.20 * maximum, 6.0 * mad)
            thresholds[arm] = (maximum + margin).tolist()
            stats[arm] = {
                "samples": len(samples),
                "observed_max_abs": maximum.tolist(),
                "median_abs": median.tolist(),
                "mad_abs": mad.tolist(),
            }
        if not thresholds:
            raise ValueError("no torque samples collected")
        return {
            "version": 1,
            "method": "max_abs + max(20% max_abs, 6 MAD)",
            "consecutive_samples": 5,
            "thresholds": thresholds,
            "calibration": stats,
        }

    def save(self, path: str | Path) -> dict:
        cfg = self.build_config()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        return cfg
