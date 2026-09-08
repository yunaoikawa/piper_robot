"""Non-blocking, scale-normalized lid displacement watchdog primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MotionWatchdogState:
    ready: bool
    visible: bool
    triggered: bool
    displacement_scale: float | None
    threshold_scale: float | None
    consecutive_exceedances: int
    reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


class RobustLidMotionWatchdog:
    """Detect lateral target motion relative to a stationary baseline.

    Coordinates and target diameter must use the same units: pixels for a
    fixed head image, or tool-frame units for the wrist image.  Thresholds are
    derived from stationary MAD and object size; no fixed pixel count exists.
    """

    def __init__(
        self,
        *,
        minimum_baseline_samples: int = 15,
        noise_multiplier: float = 6.0,
        object_motion_fraction: float = 0.05,
        consecutive_exceedances: int = 2,
    ):
        if minimum_baseline_samples < 3 or consecutive_exceedances < 1:
            raise ValueError("watchdog sample counts are invalid")
        if noise_multiplier <= 0 or object_motion_fraction <= 0:
            raise ValueError("watchdog scale factors must be positive")
        self.minimum_baseline_samples = int(minimum_baseline_samples)
        self.noise_multiplier = float(noise_multiplier)
        self.object_motion_fraction = float(object_motion_fraction)
        self.required_exceedances = int(consecutive_exceedances)
        self._baseline: list[np.ndarray] = []
        self._center = None
        self._threshold = None
        self._strikes = 0
        self._latest = MotionWatchdogState(False, False, False, None, None, 0, None)
        self._lock = threading.Lock()

    def add_baseline(self, center_xy, *, object_diameter: float) -> None:
        point = np.asarray(center_xy, dtype=float).reshape(2)
        if not np.all(np.isfinite(point)) or not np.isfinite(object_diameter) or object_diameter <= 0:
            raise ValueError("baseline observation is invalid")
        with self._lock:
            if self._center is not None:
                raise RuntimeError("watchdog baseline is already finalized")
            self._baseline.append(point)
            if len(self._baseline) < self.minimum_baseline_samples:
                self._latest = MotionWatchdogState(False, True, False, None, None, 0, None)
                return
            samples = np.vstack(self._baseline)
            center = np.median(samples, axis=0)
            radial = np.linalg.norm(samples - center, axis=1)
            radial_center = float(np.median(radial))
            mad = float(np.median(np.abs(radial - radial_center)))
            self._center = center
            self._threshold = max(
                self.noise_multiplier * 1.4826 * mad,
                self.object_motion_fraction * float(object_diameter),
            )
            self._latest = MotionWatchdogState(
                True, True, False, 0.0, self._threshold / object_diameter, 0, None
            )

    def observe(self, center_xy, *, object_diameter: float) -> MotionWatchdogState:
        point = np.asarray(center_xy, dtype=float).reshape(2)
        if not np.all(np.isfinite(point)) or object_diameter <= 0:
            raise ValueError("watchdog observation is invalid")
        with self._lock:
            if self._center is None or self._threshold is None:
                raise RuntimeError("watchdog needs a stationary baseline")
            displacement = float(np.linalg.norm(point - self._center))
            self._strikes = self._strikes + 1 if displacement > self._threshold else 0
            triggered = self._strikes >= self.required_exceedances
            self._latest = MotionWatchdogState(
                True,
                True,
                triggered,
                displacement / float(object_diameter),
                self._threshold / float(object_diameter),
                self._strikes,
                "lid_lateral_motion" if triggered else None,
            )
            return self._latest

    def missing(self) -> MotionWatchdogState:
        with self._lock:
            self._latest = MotionWatchdogState(
                self._center is not None,
                False,
                False,
                None,
                None,
                self._strikes,
                "target_not_visible",
            )
            return self._latest

    def latest(self) -> MotionWatchdogState:
        with self._lock:
            return self._latest


class DualCameraLidMotionGuard:
    """A motion veto is enough from either camera; loss requires both."""

    def __init__(self, head: RobustLidMotionWatchdog, wrist: RobustLidMotionWatchdog):
        self.head = head
        self.wrist = wrist

    def require_motion_safe(self) -> dict:
        head = self.head.latest()
        wrist = self.wrist.latest()
        if head.triggered or wrist.triggered:
            raise RuntimeError("lid motion watchdog triggered")
        if not head.visible and not wrist.visible:
            raise RuntimeError("both lid motion cameras lost the target")
        return {"head": head.to_dict(), "wrist": wrist.to_dict()}


class AsyncMotionWatchdogWorker:
    """Feed a watchdog from camera inference outside the motion thread.

    ``observe`` returns ``(center_xy, object_diameter)`` in any consistent
    image/tool coordinate system, or ``None`` when the selected target is not
    visible. The 30 Hz command thread only calls ``latest`` through the guard.
    """

    def __init__(
        self,
        watchdog: RobustLidMotionWatchdog,
        observe: Callable[[], tuple[object, float] | None],
        *,
        interval_s: float = 1.0 / 30.0,
    ):
        if interval_s <= 0:
            raise ValueError("watchdog worker interval must be positive")
        self.watchdog = watchdog
        self.observe = observe
        self.interval_s = float(interval_s)
        self.stop_event = threading.Event()
        self._thread = None
        self._error = None

    def sample_once(self) -> MotionWatchdogState:
        observation = self.observe()
        if observation is None:
            return self.watchdog.missing()
        center, diameter = observation
        if not self.watchdog.latest().ready:
            self.watchdog.add_baseline(center, object_diameter=diameter)
            return self.watchdog.latest()
        return self.watchdog.observe(center, object_diameter=diameter)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.sample_once()
                self._error = None
            except BaseException as error:
                self._error = error
            self.stop_event.wait(self.interval_s)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("watchdog worker was already started")
        self._thread = threading.Thread(
            target=self._run,
            name="lid-motion-watchdog",
            daemon=True,
        )
        self._thread.start()

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.watchdog.latest().ready:
                return
            if self._error is not None:
                raise RuntimeError(
                    f"motion watchdog observation failed: {self._error}"
                ) from self._error
            time.sleep(min(self.interval_s, 0.02))
        raise TimeoutError("motion watchdog stationary baseline timed out")

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
