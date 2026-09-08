"""Persistent checkpoint-to-cached-continuous orientation escalation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
import time

from rollout.gripper_level import JawLevelReference, assess_jaw_level


ESCALATING_REASONS = frozenset(
    {
        "jaw_tilt",
        "fingertip_height_asymmetry",
        "lid_lateral_motion",
        "asymmetric_contact",
    }
)


@dataclass(frozen=True)
class OrientationMonitoringPolicy:
    mode: str = "checkpoint"
    reason: str | None = None
    updated_at_unix_s: float | None = None

    def __post_init__(self):
        if self.mode not in {"checkpoint", "continuous_cached"}:
            raise ValueError("unknown orientation monitoring mode")


class OrientationMonitoringPolicyStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> OrientationMonitoringPolicy:
        if not self.path.exists():
            return OrientationMonitoringPolicy()
        return OrientationMonitoringPolicy(**json.loads(self.path.read_text()))

    def record_failure(self, reason: str) -> OrientationMonitoringPolicy:
        current = self.load()
        if str(reason) not in ESCALATING_REASONS:
            return current
        policy = OrientationMonitoringPolicy(
            mode="continuous_cached",
            reason=str(reason),
            updated_at_unix_s=time.time(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(policy), indent=2) + "\n")
        temporary.replace(self.path)
        return policy


class CachedOrientationMonitor:
    """Assess state pushed by the controller receiver; never call robot RPC."""

    def __init__(self, reference: JawLevelReference):
        self.reference = reference
        self._lock = threading.Lock()
        self._latest = None

    def update(self, pose_wxyz_xyz, *, observed_at_monotonic_s=None) -> None:
        assessment = assess_jaw_level(pose_wxyz_xyz, self.reference)
        with self._lock:
            self._latest = {
                "observed_at_monotonic_s": float(
                    time.monotonic()
                    if observed_at_monotonic_s is None
                    else observed_at_monotonic_s
                ),
                "assessment": assessment,
            }

    def require_level(self, *, maximum_age_s: float = 0.2):
        with self._lock:
            latest = self._latest
        if latest is None:
            raise RuntimeError("cached orientation state is unavailable")
        age = time.monotonic() - latest["observed_at_monotonic_s"]
        if age > maximum_age_s:
            raise RuntimeError(f"cached orientation state is stale: {age:.3f}s")
        assessment = latest["assessment"]
        if not assessment.accepted:
            raise RuntimeError(
                f"cached jaw orientation rejected: {assessment.reasons}"
            )
        return assessment
