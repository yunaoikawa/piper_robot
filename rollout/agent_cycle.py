"""Deterministic alternating ACT episode orchestration.

The VLA controller owns real-time motion.  This module owns only the small,
testable state machine that decides whether an episode may continue, completed
after a verified final release, or must stop for operator inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CycleDecision:
    action: str
    reason: str
    task: str
    next_task: str | None = None
    bias_attempt_index: int | None = None
    bias_offset_m: tuple[float, float, float] | None = None


class AlternatingAgentCycle:
    """Alternate named tasks and fail closed on missing runtime evidence.

    A successful terminal release is supplied by the transport-aware gripper
    latch.  Camera and controller-health inputs are deliberately booleans or
    counters so this class never reaches into hardware or performs motion.
    """

    def __init__(
        self,
        tasks: Sequence[str],
        *,
        initial_task: str,
        required_cameras: Sequence[str] = ("head", "right"),
        camera_failure_grace_s: float = 1.0,
        inference_failure_grace_s: float = 10.0,
        maximum_episode_s: float = 60.0,
        bias_retry_offsets_m: Mapping[
            str, Sequence[Sequence[float]]
        ] | None = None,
    ):
        normalized = tuple(str(task) for task in tasks)
        if len(normalized) < 2 or len(set(normalized)) != len(normalized):
            raise ValueError("cycle requires at least two unique tasks")
        if initial_task not in normalized:
            raise ValueError("initial task must belong to the cycle")
        if camera_failure_grace_s < 0 or inference_failure_grace_s <= 0:
            raise ValueError("failure grace periods must be non-negative")
        if maximum_episode_s <= 0:
            raise ValueError("maximum episode duration must be positive")
        self.tasks = normalized
        self.current_task = str(initial_task)
        self.required_cameras = tuple(str(name) for name in required_cameras)
        self.camera_failure_grace_s = float(camera_failure_grace_s)
        self.inference_failure_grace_s = float(inference_failure_grace_s)
        self.maximum_episode_s = float(maximum_episode_s)
        raw_schedules = dict(bias_retry_offsets_m or {})
        unknown = set(raw_schedules) - set(normalized)
        if unknown:
            raise ValueError(
                f"bias retry schedule has unknown tasks: {sorted(unknown)}"
            )
        self.bias_retry_offsets_m = {}
        for task in normalized:
            raw = raw_schedules.get(task, ((0.0, 0.0, 0.0),))
            values = np.asarray(raw, dtype=float)
            if (values.ndim != 2 or values.shape[1:] != (3,)
                    or len(values) < 1 or not np.all(np.isfinite(values))):
                raise ValueError(
                    f"{task} bias retry offsets must be a non-empty finite Nx3 list"
                )
            if not np.allclose(values[0], 0.0):
                raise ValueError(
                    f"{task} first bias retry offset must be [0, 0, 0]"
                )
            self.bias_retry_offsets_m[task] = tuple(
                tuple(float(axis) for axis in row) for row in values
            )
        self.bias_attempt_index = {task: 0 for task in normalized}
        self.enabled = True
        self.stop_reason: str | None = None
        self.episode_started_at: float | None = None
        self.safety_rejected_at_start = 0
        self._camera_missing_since: dict[str, float] = {}
        self._inference_missing_since: float | None = None

    @property
    def next_task(self) -> str:
        index = self.tasks.index(self.current_task)
        return self.tasks[(index + 1) % len(self.tasks)]

    @property
    def current_bias_attempt_index(self) -> int:
        return self.bias_attempt_index[self.current_task]

    @property
    def current_bias_offset_m(self) -> tuple[float, float, float]:
        return self.bias_retry_offsets_m[self.current_task][
            self.current_bias_attempt_index
        ]

    def begin_episode(self, *, safety_rejected_count: int,
                      now: float | None = None) -> None:
        if not self.enabled:
            raise RuntimeError(f"cycle is stopped: {self.stop_reason}")
        self.episode_started_at = time.monotonic() if now is None else float(now)
        self.safety_rejected_at_start = int(safety_rejected_count)
        self._camera_missing_since.clear()
        self._inference_missing_since = None

    def note_inference(self, available: bool, *, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        if available:
            self._inference_missing_since = None
        elif self._inference_missing_since is None:
            self._inference_missing_since = now

    def evaluate(
        self,
        *,
        terminal_release: bool,
        camera_ready: Mapping[str, bool],
        safety_rejected_count: int,
        pressure_stop: bool = False,
        explicit_failure: str | None = None,
        now: float | None = None,
    ) -> CycleDecision | None:
        if not self.enabled or self.episode_started_at is None:
            return None
        now = time.monotonic() if now is None else float(now)

        if explicit_failure:
            reason = str(explicit_failure)
            if reason == "grasp_miss":
                return self._bias_retry_decision(reason)
            return self._stop(reason)
        if pressure_stop:
            return self._stop("pressure_stop")
        if int(safety_rejected_count) > self.safety_rejected_at_start:
            return self._stop("safety_rejected")

        for camera in self.required_cameras:
            if bool(camera_ready.get(camera, False)):
                self._camera_missing_since.pop(camera, None)
                continue
            missing_since = self._camera_missing_since.setdefault(camera, now)
            if now - missing_since >= self.camera_failure_grace_s:
                return self._stop(f"camera_missing:{camera}")

        if (self._inference_missing_since is not None
                and now - self._inference_missing_since
                >= self.inference_failure_grace_s):
            return self._stop("inference_unavailable")
        if now - self.episode_started_at >= self.maximum_episode_s:
            return self._bias_retry_decision("episode_timeout")
        if terminal_release:
            return CycleDecision(
                action="complete_success",
                reason="released_after_transport",
                task=self.current_task,
                next_task=self.next_task,
            )
        return None

    def advance_after_success(self) -> str:
        if not self.enabled:
            raise RuntimeError(f"cycle is stopped: {self.stop_reason}")
        completed_task = self.current_task
        self.current_task = self.next_task
        # A later visit to either task starts from that task's configured base
        # bias.  A successful scheduled bias can separately be promoted to the
        # controller's base before this reset.
        self.bias_attempt_index[completed_task] = 0
        self.bias_attempt_index[self.current_task] = 0
        self.episode_started_at = None
        self._camera_missing_since.clear()
        self._inference_missing_since = None
        return self.current_task

    def advance_after_retry(self, decision: CycleDecision) -> int:
        """Commit a retry decision after failure save + guarded home succeed."""
        if not self.enabled:
            raise RuntimeError(f"cycle is stopped: {self.stop_reason}")
        if (decision.action != "retry_failure"
                or decision.task != self.current_task
                or decision.bias_attempt_index is None):
            raise ValueError("decision is not a retry for the current task")
        expected = self.current_bias_attempt_index + 1
        if decision.bias_attempt_index != expected:
            raise ValueError("retry decision is stale or out of sequence")
        self.bias_attempt_index[self.current_task] = expected
        self.episode_started_at = None
        self._camera_missing_since.clear()
        self._inference_missing_since = None
        return expected

    def stop_after_retry_exhausted(self, decision: CycleDecision) -> CycleDecision:
        if (decision.action != "stop_retry_exhausted"
                or decision.task != self.current_task):
            raise ValueError("decision is not exhausted for the current task")
        return self._stop(f"bias_schedule_exhausted:{decision.reason}")

    def stop(self, reason: str) -> CycleDecision:
        return self._stop(reason)

    def _stop(self, reason: str) -> CycleDecision:
        self.enabled = False
        self.stop_reason = str(reason)
        return CycleDecision(
            action="stop_failure",
            reason=self.stop_reason,
            task=self.current_task,
        )

    def _bias_retry_decision(self, reason: str) -> CycleDecision:
        next_index = self.current_bias_attempt_index + 1
        schedule = self.bias_retry_offsets_m[self.current_task]
        if next_index >= len(schedule):
            # Do not disable yet: the controller must first save the last
            # failure and pressure-home.  It calls stop_after_retry_exhausted
            # only after those two operations complete.
            return CycleDecision(
                action="stop_retry_exhausted",
                reason=str(reason),
                task=self.current_task,
                bias_attempt_index=self.current_bias_attempt_index,
                bias_offset_m=self.current_bias_offset_m,
            )
        return CycleDecision(
            action="retry_failure",
            reason=str(reason),
            task=self.current_task,
            next_task=self.current_task,
            bias_attempt_index=next_index,
            bias_offset_m=schedule[next_index],
        )
