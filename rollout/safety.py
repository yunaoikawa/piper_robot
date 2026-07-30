"""Preventive geometric checks applied to streamed EE targets.

The demo-derived Cartesian workspace box was removed because absence from a
demonstration does not identify an unsafe location.  This layer now rejects
only explicitly measured keep-out volumes and discontinuous Quest targets.
Measured joint torque is monitored independently by the active motion
controller (for recovery teleoperation, by
``rollout.recovery_teleop_safety``).

Limits are grounded in measurement, not taste:
  per-step EE displacement over all 360,863 demo transitions at 30 Hz
    p50 1.54 mm   p95 8.24 mm   p99.9 15.17 mm   max 29.93 mm (0.9 m/s)
so MAX_STEP_M below sits ~33% above the largest motion ever demonstrated -- it
cannot fire on normal operation, only on a genuine runaway.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

# Above the 29.93 mm demo maximum with headroom. Applied between *consecutive
# commanded targets*, which is the quantity the demo statistic measures -- not
# target-vs-current-pose, which lags legitimately.
MAX_STEP_M = 0.040


@dataclass
class KeepOutZone:
    """An axis-aligned box the EE must not enter, in robot frame metres."""

    name: str
    lo: np.ndarray
    hi: np.ndarray

    def contains(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))


@dataclass
class SafetyLayer:
    """Vets EE targets. Returns None for a target that must not be sent.

    A rejected target means the caller holds the previous pose. Holding is the
    safe failure here: the arm is position-controlled with a preview time, so
    simply not issuing a new target leaves it where it is.
    """

    zones: list[KeepOutZone] = field(default_factory=list)
    max_step_m: float = MAX_STEP_M
    _prev: dict[str, np.ndarray] = field(default_factory=dict)
    _last_log: float = 0.0
    _rejected: int = 0

    @classmethod
    def from_config(cls, path: str | None) -> "SafetyLayer":
        """Load keep-out zones from JSON. No config = bounds-only (zones empty).

        Expected shape:
            {"max_step_m": 0.04,
             "keep_out": [{"name": "reagent_shelf",
                           "lo": [x, y, z], "hi": [x, y, z]}]}
        """
        if not path:
            return cls()
        with open(path) as f:
            cfg = json.load(f)
        zones = [
            KeepOutZone(z["name"], np.asarray(z["lo"], float), np.asarray(z["hi"], float))
            for z in cfg.get("keep_out", [])
        ]
        layer = cls(zones=zones, max_step_m=float(cfg.get("max_step_m", MAX_STEP_M)))
        print(f"[safety] loaded {len(zones)} keep-out zone(s) from {path}; "
              f"max_step={layer.max_step_m * 1000:.0f}mm")
        return layer

    def reset(self, arm: str | None = None) -> None:
        """Forget the previous target so the next one skips the step check.

        Call on episode start and after any pause -- the first target of an
        episode has no predecessor, and stale state would false-trigger.
        """
        if arm is None:
            self._prev.clear()
        else:
            self._prev.pop(arm, None)

    def check(self, arm: str, p_target: np.ndarray) -> np.ndarray | None:
        """Vet one target position. Returns it unchanged, or None to reject."""
        p = np.asarray(p_target, dtype=float)

        for z in self.zones:
            if z.contains(p):
                self._reject(f"{arm} target {np.round(p, 3)} inside keep-out '{z.name}'")
                return None

        prev = self._prev.get(arm)
        if prev is not None:
            step = float(np.linalg.norm(p - prev))
            if step > self.max_step_m:
                self._reject(
                    f"{arm} step {step * 1000:.0f}mm > {self.max_step_m * 1000:.0f}mm "
                    f"limit ({np.round(prev, 3)} -> {np.round(p, 3)})"
                )
                return None

        self._prev[arm] = p
        return p

    def _reject(self, msg: str) -> None:
        self._rejected += 1
        now = time.time()
        if now - self._last_log >= 1.0:
            extra = f"  ({self._rejected} rejected so far)" if self._rejected > 1 else ""
            print(f"[safety] REJECTED {msg}{extra}", flush=True)
            self._last_log = now

    @property
    def rejected_count(self) -> int:
        return self._rejected
