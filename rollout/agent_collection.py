"""30 Hz ACT intervention collection primitives.

This module is deliberately independent from the ordinary demonstration
recorder.  Agent episodes retain the model action, the command after bias, and
the measured trajectory while keeping failures outside the training pool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Mapping

import cv2
import h5py
import numpy as np

from rollout.recorder import StreamingVideoWriter


SCHEMA = "piper_robot.agent_intervention_episode/v1"
FAILURE_REASONS = {
    "grasp_miss",
    "jam",
    "drop",
    "wrong_placement",
    "abort",
}


def summarize_sampling_timing(
    timestamps: Any,
    *,
    target_hz: float,
    deadline_factor: float = 1.5,
    minimum_rate_ratio: float = 0.9,
) -> dict[str, Any]:
    """Summarize measured cadence on a pause-excluding active clock.

    This deliberately uses timestamps instead of the claimed output FPS.  A
    slow producer must not turn ten physical samples per second into apparent
    30 Hz data merely by assigning ``frame_index / 30`` timestamps.
    """

    values = np.asarray(timestamps, dtype=float).reshape(-1)
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")
    if deadline_factor <= 1:
        raise ValueError("deadline_factor must be greater than one")
    if not 0 < minimum_rate_ratio <= 1:
        raise ValueError("minimum_rate_ratio must be in (0, 1]")

    empty = {
        "eligible": False,
        "target_hz": float(target_hz),
        "effective_hz": 0.0,
        "active_duration_s": 0.0,
        "median_period_s": None,
        "p95_period_s": None,
        "interval_deadline_misses": 0,
        "invalid_intervals": 0,
    }
    if len(values) < 2:
        return empty

    intervals = np.diff(values)
    valid = np.isfinite(intervals) & (intervals > 0)
    valid_intervals = intervals[valid]
    invalid_intervals = int((~valid).sum())
    if not len(valid_intervals):
        return {**empty, "invalid_intervals": invalid_intervals}

    target_period_s = 1.0 / float(target_hz)
    duration_s = float(valid_intervals.sum())
    effective_hz = float(len(valid_intervals) / duration_s)
    median_period_s = float(np.median(valid_intervals))
    p95_period_s = float(np.percentile(valid_intervals, 95))
    interval_deadline_misses = int(
        np.count_nonzero(valid_intervals > deadline_factor * target_period_s)
    )
    eligible = bool(
        invalid_intervals == 0
        and effective_hz >= minimum_rate_ratio * float(target_hz)
        and p95_period_s <= deadline_factor * target_period_s
    )
    return {
        "eligible": eligible,
        "target_hz": float(target_hz),
        "effective_hz": effective_hz,
        "active_duration_s": duration_s,
        "median_period_s": median_period_s,
        "p95_period_s": p95_period_s,
        "interval_deadline_misses": interval_deadline_misses,
        "invalid_intervals": invalid_intervals,
    }


class GripperCloseLatch:
    """Hold a confirmed grasp until measured transport makes release plausible.

    ACT is time-agnostic in this deployment.  Replanning after the demonstrated
    trajectory can therefore start another open/close cycle.  The latch blocks
    those transient open commands while the end effector remains near the
    confirmed grasp pose, but it must not suppress the demonstrated final
    release.  Release is enabled from measured Cartesian displacement relative
    to that pose; it does not depend on a task-specific image pixel or robot
    coordinate.  The latch is armed only after an explicitly open command has
    been seen, so a noisy first prediction cannot engage it accidentally.
    """

    def __init__(self, *, open_threshold: float = 0.75,
                 close_threshold: float = 0.35,
                 empty_closed_threshold: float = 0.08,
                 minimum_close_time_s: float = 1.0,
                 stall_window_s: float = 0.35,
                 stall_range: float = 0.02,
                 minimum_transport_distance_m: float = 0.03,
                 minimum_release_command_time_s: float = 0.15):
        if not 0.0 <= close_threshold < open_threshold <= 1.0:
            raise ValueError("gripper thresholds must satisfy 0 <= close < open <= 1")
        if minimum_transport_distance_m <= 0:
            raise ValueError("minimum transport distance must be positive")
        if minimum_release_command_time_s < 0:
            raise ValueError("minimum release command time cannot be negative")
        self.open_threshold = float(open_threshold)
        self.close_threshold = float(close_threshold)
        self.empty_closed_threshold = float(empty_closed_threshold)
        self.minimum_close_time_s = float(minimum_close_time_s)
        self.stall_window_s = float(stall_window_s)
        self.stall_range = float(stall_range)
        self.minimum_transport_distance_m = float(minimum_transport_distance_m)
        self.minimum_release_command_time_s = float(minimum_release_command_time_s)
        self.reset()

    def reset(self) -> None:
        self.seen_open = False
        self.latched = False
        self.target = 1.0
        self.closing_started_at: float | None = None
        self.measured_history: list[tuple[float, float]] = []
        self.grasp_xyz_m: np.ndarray | None = None
        self.latest_xyz_m: np.ndarray | None = None
        self.transport_displacement_m = 0.0
        self.release_enabled = False
        self.release_requested_at: float | None = None
        self.release_commanded = False
        self.released = False

    @staticmethod
    def _validated_xyz(measured_xyz: Any | None) -> np.ndarray | None:
        if measured_xyz is None:
            return None
        xyz = np.asarray(measured_xyz, dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            return None
        return xyz.copy()

    def _update_transport(self, measured_xyz: Any | None) -> None:
        xyz = self._validated_xyz(measured_xyz)
        if xyz is None:
            return
        self.latest_xyz_m = xyz
        if not self.latched or self.release_enabled:
            return
        # A controller can receive its first valid observation just after the
        # stall was confirmed.  Capture that first measured pose rather than
        # turning one missing sample into a permanent no-release state.
        if self.grasp_xyz_m is None:
            self.grasp_xyz_m = xyz.copy()
            return
        self.transport_displacement_m = float(np.linalg.norm(xyz - self.grasp_xyz_m))
        if self.transport_displacement_m >= self.minimum_transport_distance_m:
            self.release_enabled = True

    def apply(self, command: float, measured: float,
              *, measured_xyz: Any | None = None,
              now: float | None = None) -> tuple[float, bool]:
        value = float(np.clip(command, 0.0, 1.0))
        measured = float(np.clip(measured, 0.0, 1.0))
        now = time.monotonic() if now is None else float(now)
        newly_latched = False
        if self.released:
            # Release is terminal for this episode.  Keep the jaws open while
            # the episode manager records/promotes the completed transition;
            # a subsequent time-agnostic replan must not close them again.
            self.target = max(self.target, value, self.open_threshold)
            return self.target, False
        if self.latched:
            self._update_transport(measured_xyz)
            if self.release_commanded:
                self.target = max(self.target, value, self.open_threshold)
                if measured >= self.open_threshold:
                    self.released = True
                    self.latched = False
                return self.target, False
            if self.release_enabled and value >= self.open_threshold:
                if self.release_requested_at is None:
                    self.release_requested_at = now
                if now - self.release_requested_at >= self.minimum_release_command_time_s:
                    self.release_commanded = True
                    self.target = max(value, self.open_threshold)
                    return self.target, False
            else:
                self.release_requested_at = None
            self.target = min(self.target, value)
            return self.target, False

        if value >= self.open_threshold:
            self.seen_open = True
            self.closing_started_at = None
            self.measured_history.clear()
            return value, False

        if self.seen_open and value <= self.close_threshold:
            if self.closing_started_at is None:
                self.closing_started_at = now
                self.measured_history.clear()
            self.measured_history.append((now, measured))
            cutoff = now - self.stall_window_s
            self.measured_history = [
                sample for sample in self.measured_history if sample[0] >= cutoff
            ]
            close_age = now - self.closing_started_at
            recent = [sample[1] for sample in self.measured_history]
            blocked = (
                close_age >= self.minimum_close_time_s
                and len(recent) >= 2
                and measured > self.empty_closed_threshold
                and max(recent) - min(recent) <= self.stall_range
            )
            if blocked:
                self.latched = True
                self.target = value
                self.grasp_xyz_m = self._validated_xyz(measured_xyz)
                self.latest_xyz_m = (
                    None if self.grasp_xyz_m is None else self.grasp_xyz_m.copy()
                )
                newly_latched = True
        elif self.closing_started_at is not None:
            # An opening/mid-range command before a confirmed stall means the
            # grasp was not proven.  Do not trap an empty closed gripper.
            self.closing_started_at = None
            self.measured_history.clear()

        return value, newly_latched


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class ControllerClaim:
    """Cross-worktree exclusive ownership of the physical right controller."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.stream = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError(
                f"right-arm controller already owned: {self.path}"
            ) from error
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(
            json.dumps({"pid": os.getpid(), "acquired_at": utc_now()}) + "\n"
        )
        self.stream.flush()

    def release(self) -> None:
        if self.stream is None:
            return
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None


@dataclass(frozen=True)
class AgentRecordingSample:
    wall_timestamp: float
    active_timestamp: float
    left_ee_pose: Any
    right_ee_pose: Any
    left_gripper_exact: float
    right_gripper_exact: float
    left_gripper: float
    right_gripper: float
    left_joint_positions: np.ndarray
    right_joint_positions: np.ndarray
    head_rgb: np.ndarray | None
    left_rgb: np.ndarray | None
    right_rgb: np.ndarray | None
    camera_timestamps: tuple[float, float, float]
    camera_frame_ids: tuple[int, int, int]
    policy_action_quat16: np.ndarray
    commanded_target_quat16: np.ndarray
    xyz_bias_left_right: np.ndarray
    chunk_index: int
    action_generation: int
    action_observation_timestamp: float
    intervention_revision: int
    safety_rejected_count: int


class AgentEpisodeRecorder:
    """Crash-tolerant agent recorder with explicit outcome promotion."""

    def __init__(self, root: str | Path, stop_event: threading.Event, fps: int = 30):
        self.root = Path(root).resolve()
        self.stop_event = stop_event
        self.worker_stop = threading.Event()
        self.fps = int(fps)
        if self.fps != 30:
            raise ValueError("agent collection is fixed at 30 Hz")
        self.pending_root = self.root / "pending"
        self.pending_root.mkdir(parents=True, exist_ok=True)
        self.queue: queue.Queue[AgentRecordingSample] = queue.Queue(maxsize=600)
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.lock = threading.Lock()
        self.is_recording = False
        self.accepting_samples = False
        self.episode_count = 0
        self.context: dict[str, Any] = {}
        self.data: dict[str, list] = {}
        self.episode_dir: Path | None = None
        self.episode_name: str | None = None
        self.audit_stream = None
        self.writers: dict[str, StreamingVideoWriter] = {}
        self.dropped_samples = 0
        self.deadline_misses = 0
        self._sampling_active = False
        self._active_elapsed_s = 0.0
        self._active_segment_started_at: float | None = None
        self.worker.start()

    @property
    def save_dir(self) -> Path:
        return self.root

    def success_count(self, task: str) -> int:
        directory = self.root / "success" / task
        return sum(1 for path in directory.glob("agent_*") if path.is_dir()) if directory.exists() else 0

    def configure_episode(self, context: Mapping[str, Any]) -> None:
        required = {"task", "target_selection", "initial_bias_m"}
        missing = required - set(context)
        if missing:
            raise ValueError(f"agent episode context missing {sorted(missing)}")
        if context["task"] not in {"lid_open", "lid_close"}:
            raise ValueError("agent pilot only supports lid_open and lid_close")
        self.context = dict(context)

    @staticmethod
    def _empty_data() -> dict[str, list]:
        names = (
            "wall_timestamps",
            "active_timestamps",
            "left_ee_pos",
            "left_ee_quat",
            "left_gripper_exact",
            "left_gripper",
            "left_joint_positions",
            "right_ee_pos",
            "right_ee_quat",
            "right_gripper_exact",
            "right_gripper",
            "right_joint_positions",
            "camera_timestamps",
            "camera_frame_ids",
            "policy_action_quat16",
            "commanded_target_quat16",
            "xyz_bias_left_right",
            "chunk_index",
            "action_generation",
            "action_observation_timestamp",
            "intervention_revision",
            "safety_rejected_count",
        )
        return {name: [] for name in names}

    def start_episode(self) -> None:
        if self.is_recording:
            raise RuntimeError("agent episode already recording")
        if not self.context:
            raise RuntimeError("configure_episode must precede start_episode")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_name = f"agent_{self.context['task']}_{stamp}_{self.episode_count:04d}"
        self.episode_dir = self.pending_root / self.episode_name
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.data = self._empty_data()
        self.dropped_samples = 0
        self.deadline_misses = 0
        with self.lock:
            self._sampling_active = False
            self._active_elapsed_s = 0.0
            self._active_segment_started_at = None
        self.writers = {
            name: StreamingVideoWriter(
                self.episode_dir / f"{self.episode_name}_{name}.mp4", fps=self.fps
            )
            for name in ("head", "left", "right")
        }
        self.audit_stream = (self.episode_dir / "interventions.jsonl").open(
            "a", encoding="utf-8"
        )
        self.log_event("episode_started", context=self.context)
        self.is_recording = True
        self.accepting_samples = True

    def set_sampling_active(self, active: bool, *, now: float | None = None) -> None:
        """Start or stop the clock used for recorded training timestamps."""

        now = time.monotonic() if now is None else float(now)
        with self.lock:
            active = bool(active)
            if active == self._sampling_active:
                return
            if active:
                self._active_segment_started_at = now
            else:
                if self._active_segment_started_at is not None:
                    self._active_elapsed_s += max(
                        0.0, now - self._active_segment_started_at
                    )
                self._active_segment_started_at = None
            self._sampling_active = active

    def active_timestamp(self, *, now: float | None = None) -> float:
        """Return elapsed running time, excluding pause/inference-wait spans."""

        now = time.monotonic() if now is None else float(now)
        with self.lock:
            elapsed = self._active_elapsed_s
            if self._sampling_active and self._active_segment_started_at is not None:
                elapsed += max(0.0, now - self._active_segment_started_at)
            return float(elapsed)

    def log_event(self, event: str, **payload: Any) -> None:
        record = {"timestamp_utc": utc_now(), "event": event, **payload}
        stream = self.audit_stream
        if stream is not None:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()

    def record_sample(self, sample: AgentRecordingSample) -> None:
        if not self.accepting_samples:
            return
        try:
            self.queue.put_nowait(sample)
        except queue.Full:
            self.dropped_samples += 1

    def note_deadline_miss(self) -> None:
        self.deadline_misses += 1

    def _worker(self) -> None:
        while (not self.stop_event.is_set() and not self.worker_stop.is_set()) or not self.queue.empty():
            try:
                sample = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                frames = {
                    "head": sample.head_rgb,
                    "left": sample.left_rgb,
                    "right": sample.right_rgb,
                }
                for name, frame in frames.items():
                    if frame is not None:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    self.writers[name].write(frame)
                with self.lock:
                    left_pos = sample.left_ee_pose.translation()
                    left_quat = sample.left_ee_pose.rotation().wxyz
                    right_pos = sample.right_ee_pose.translation()
                    right_quat = sample.right_ee_pose.rotation().wxyz
                    values = {
                        "wall_timestamps": sample.wall_timestamp,
                        "active_timestamps": sample.active_timestamp,
                        "left_ee_pos": left_pos,
                        "left_ee_quat": left_quat,
                        "left_gripper_exact": sample.left_gripper_exact,
                        "left_gripper": sample.left_gripper,
                        "left_joint_positions": sample.left_joint_positions,
                        "right_ee_pos": right_pos,
                        "right_ee_quat": right_quat,
                        "right_gripper_exact": sample.right_gripper_exact,
                        "right_gripper": sample.right_gripper,
                        "right_joint_positions": sample.right_joint_positions,
                        "camera_timestamps": sample.camera_timestamps,
                        "camera_frame_ids": sample.camera_frame_ids,
                        "policy_action_quat16": sample.policy_action_quat16,
                        "commanded_target_quat16": sample.commanded_target_quat16,
                        "xyz_bias_left_right": sample.xyz_bias_left_right,
                        "chunk_index": sample.chunk_index,
                        "action_generation": sample.action_generation,
                        "action_observation_timestamp": sample.action_observation_timestamp,
                        "intervention_revision": sample.intervention_revision,
                        "safety_rejected_count": sample.safety_rejected_count,
                    }
                    for name, value in values.items():
                        self.data[name].append(value)
            finally:
                self.queue.task_done()

    def _save_hdf5(self) -> Path:
        assert self.episode_dir is not None and self.episode_name is not None
        path = self.episode_dir / f"{self.episode_name}.hdf5"
        with h5py.File(path, "w") as output:
            with self.lock:
                for name, values in self.data.items():
                    output.create_dataset(name, data=np.asarray(values))
                # Existing VLA converters expect `timestamps`.
                output["timestamps"] = np.asarray(self.data["active_timestamps"])
                output.attrs["num_samples"] = len(self.data["active_timestamps"])
            output.attrs["schema"] = SCHEMA
            output.attrs["control_frequency_hz"] = self.fps
            output.attrs["task"] = self.context["task"]
            output.attrs["pause_samples_excluded"] = True
        return path

    def end_episode(self) -> None:
        """Stop active sampling but retain the episode under pending."""
        if not self.is_recording:
            return
        self.set_sampling_active(False)
        self.accepting_samples = False
        self.queue.join()
        self.is_recording = False
        for writer in self.writers.values():
            writer.release()
        self.writers = {}
        hdf5_path = self._save_hdf5()
        self.log_event("episode_stopped", hdf5=str(hdf5_path))
        if self.audit_stream is not None:
            self.audit_stream.close()
            self.audit_stream = None

    def finalize(self, outcome: str, *, reason: str | None = None) -> Path:
        if self.is_recording:
            raise RuntimeError("end episode before finalizing outcome")
        if self.episode_dir is None or self.episode_name is None:
            raise RuntimeError("no pending agent episode")
        if outcome not in {"success", "failure"}:
            raise ValueError("outcome must be success or failure")
        if outcome == "failure" and reason not in FAILURE_REASONS:
            raise ValueError(f"unsupported failure reason {reason!r}")
        sample_count = len(self.data.get("active_timestamps", ()))
        timing = summarize_sampling_timing(
            self.data.get("active_timestamps", ()), target_hz=self.fps
        )
        manifest = {
            "schema": SCHEMA,
            "episode_name": self.episode_name,
            "task": self.context["task"],
            "outcome": outcome,
            "failure_reason": reason,
            "training_eligible": (
                outcome == "success" and sample_count > 1
                and self.dropped_samples == 0
                and self.deadline_misses <= max(1, sample_count // 20)
                and timing["eligible"]
            ),
            "sample_count": sample_count,
            "control_frequency_hz": self.fps,
            "dropped_samples": self.dropped_samples,
            "deadline_misses": self.deadline_misses,
            "sampling_timing": timing,
            "context": self.context,
            "finalized_at": utc_now(),
        }
        hashes = {}
        for path in self.episode_dir.iterdir():
            if path.is_file():
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["sha256"] = hashes
        _atomic_json(self.episode_dir / "manifest.json", manifest)
        destination_root = (
            self.root / "success" / self.context["task"]
            if outcome == "success"
            else self.root / "failures" / self.context["task"]
        )
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / self.episode_name
        os.replace(self.episode_dir, destination)
        self.episode_count += 1
        self.context = {}
        self.data = {}
        self.episode_dir = None
        self.episode_name = None
        return destination

    def stop(self) -> None:
        if self.is_recording:
            self.end_episode()
        self.worker_stop.set()
        self.worker.join(timeout=2.0)


@dataclass
class InterventionState:
    """Thread-safe command mailbox consumed only by the 30 Hz control loop."""

    maximum_bias_m: float = 0.06
    lock: threading.Lock = field(default_factory=threading.Lock)
    condition: threading.Condition = field(init=False)
    revision: int = 0
    correction_revision: int = 0
    mode: str = "idle"
    requested: str | None = None
    request_payload: dict[str, Any] = field(default_factory=dict)
    request_id: int = 0
    completed_request_id: int = 0
    bias: dict[str, np.ndarray] = field(
        default_factory=lambda: {"left": np.zeros(3), "right": np.zeros(3)}
    )
    target_selection: dict[str, Any] | None = None
    latest_metrics: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None

    def __post_init__(self):
        self.condition = threading.Condition(self.lock)

    def submit(self, command: str, payload: Mapping[str, Any] | None = None) -> int:
        with self.condition:
            if self.requested is not None:
                raise RuntimeError("another UI command is still pending")
            self.request_id += 1
            self.requested = command
            self.request_payload = dict(payload or {})
            self.condition.notify_all()
            return self.request_id

    def complete(
        self, request_id: int, *, mode: str | None = None,
        error: str | None = None, **metrics: Any
    ) -> None:
        with self.condition:
            if mode is not None:
                self.mode = mode
            self.latest_metrics.update(metrics)
            self.last_error = error
            self.revision += 1
            self.completed_request_id = request_id
            self.requested = None
            self.request_payload = {}
            self.condition.notify_all()

    def wait(self, request_id: int, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while self.completed_request_id < request_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("control loop did not finish UI command")
                self.condition.wait(remaining)
            return self.snapshot_unlocked()

    def snapshot_unlocked(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "correction_revision": self.correction_revision,
            "mode": self.mode,
            "pending_command": self.requested,
            "bias": {arm: values.tolist() for arm, values in self.bias.items()},
            "target_selection": self.target_selection,
            "metrics": dict(self.latest_metrics),
            "last_error": self.last_error,
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return self.snapshot_unlocked()

    def set_bias(self, arm: str, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float).reshape(3)
        if arm not in self.bias or not np.all(np.isfinite(value)):
            raise ValueError("invalid arm bias")
        if np.any(np.abs(value) > self.maximum_bias_m + 1e-12):
            raise ValueError("agent bias exceeds configured ±0.06 m limit")
        self.bias[arm] = value.copy()


def intervention_slice_mask(revisions: np.ndarray, mode: str) -> np.ndarray:
    revisions = np.asarray(revisions, dtype=int).reshape(-1)
    if mode == "all":
        return np.ones(len(revisions), dtype=bool)
    if mode != "post-intervention":
        raise ValueError("intervention slice must be all or post-intervention")
    indices = np.flatnonzero(revisions > 0)
    if not len(indices):
        return np.zeros(len(revisions), dtype=bool)
    mask = np.zeros(len(revisions), dtype=bool)
    mask[int(indices[0]) :] = True
    return mask
