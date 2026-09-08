"""Agentic supervision and audit primitives for policy data collection.

The policy remains responsible for continuous control.  This module only runs
at semantic checkpoints, records why execution continued or stopped, and
classifies the resulting episode.  It deliberately contains no robot RPCs so
the same decisions can be replayed from recorded observations in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable, Mapping

import cv2
import h5py
import numpy as np


SCHEMA = "piper_robot.agentic_collection/v1"


class CollectionMode(str, Enum):
    SHADOW = "shadow"
    SUPERVISED = "supervised"
    AUTO = "auto"


class Checkpoint(str, Enum):
    INITIAL = "initial"
    PRE_GRASP = "pre_grasp"
    POST_GRASP = "post_grasp_lift"
    PRE_PLACE = "pre_place"
    POST_RELEASE = "post_release"


class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


class EpisodeClass(str, Enum):
    CLEAN_SUCCESS = "clean_success"
    RECOVERY = "recovery"
    FAILURE = "failure"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    instruction: str
    source: str
    destination: str
    arm: str
    reference_glob: str | None = None

    def __post_init__(self) -> None:
        if self.arm not in {"left", "right"}:
            raise ValueError("task arm must be left or right")
        if self.source == self.destination:
            raise ValueError("task source and destination must differ")


@dataclass(frozen=True)
class VerifierConfig:
    object_diameter_m: float
    maximum_observation_age_s: float = 0.75
    maximum_camera_skew_s: float = 0.20
    reference_position_tolerance_diameters: float = 0.75
    minimum_lift_diameters: float = 0.12
    minimum_visual_change_fraction: float = 0.002
    gripper_open_threshold: float = 0.65
    gripper_closed_threshold: float = 0.35
    maximum_closed_reference_error: float = 0.15
    stable_observations: int = 2
    required_cameras: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.object_diameter_m) or self.object_diameter_m <= 0:
            raise ValueError("object_diameter_m must be finite and positive")
        if self.stable_observations < 1:
            raise ValueError("stable_observations must be positive")


@dataclass(frozen=True)
class ReferenceEnvelope:
    """Task geometry distilled from verified demonstrations.

    Position tolerances are expressed later in object diameters rather than
    pixels.  Exact gripper ranges are optional because older recordings only
    stored a binary goal and cannot prove that an object obstructed closure.
    """

    close_position_xyz: tuple[float, float, float]
    release_position_xyz: tuple[float, float, float]
    closed_gripper_median: float | None
    open_gripper_median: float | None
    evidence_files: tuple[str, ...]

    @classmethod
    def from_hdf5(cls, paths: Iterable[str | Path], *, arm: str) -> "ReferenceEnvelope":
        close_positions: list[np.ndarray] = []
        release_positions: list[np.ndarray] = []
        closed_exact: list[float] = []
        open_exact: list[float] = []
        evidence: list[str] = []
        for raw_path in paths:
            path = Path(raw_path).resolve()
            with h5py.File(path, "r") as recording:
                pos = np.asarray(recording[f"{arm}_ee_pos"], dtype=float)
                binary = np.asarray(recording[f"{arm}_gripper"], dtype=float)
                exact = (
                    np.asarray(recording[f"{arm}_gripper_exact"], dtype=float)
                    if f"{arm}_gripper_exact" in recording
                    else None
                )
            if pos.ndim != 2 or pos.shape[1] != 3 or binary.shape != (len(pos),):
                raise ValueError(f"{path}: unsupported demonstration shape")
            is_open = binary > 0.5
            transitions = np.flatnonzero(is_open[1:] != is_open[:-1]) + 1
            if len(transitions) != 2:
                continue
            close, release = (int(value) for value in transitions)
            if is_open[close] or not is_open[release]:
                continue
            close_positions.append(pos[close])
            release_positions.append(pos[release])
            if exact is not None and exact.shape == binary.shape:
                closed_exact.append(float(np.median(exact[close:release])))
                open_exact.extend([float(exact[max(0, close - 1)]), float(exact[release])])
            evidence.append(str(path))
        if not evidence:
            raise ValueError("no verified open-close-open reference demonstrations")
        return cls(
            close_position_xyz=tuple(np.median(close_positions, axis=0).tolist()),
            release_position_xyz=tuple(np.median(release_positions, axis=0).tolist()),
            closed_gripper_median=(float(np.median(closed_exact)) if closed_exact else None),
            open_gripper_median=(float(np.median(open_exact)) if open_exact else None),
            evidence_files=tuple(evidence),
        )


@dataclass(frozen=True)
class SemanticObservation:
    timestamp: float
    ee_position_xyz: tuple[float, float, float]
    gripper_open_fraction: float
    camera_timestamps: Mapping[str, float] = field(default_factory=dict)
    visual_change_fraction: Mapping[str, float] = field(default_factory=dict)
    target_visible: bool | None = None
    target_station: str | None = None
    target_position_xyz: tuple[float, float, float] | None = None
    target_attached: bool | None = None
    target_supported: bool | None = None
    action_stream_healthy: bool = True
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointDecision:
    checkpoint: Checkpoint
    verdict: Verdict
    reasons: tuple[str, ...]
    metrics: Mapping[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["checkpoint"] = self.checkpoint.value
        value["verdict"] = self.verdict.value
        return value


def _station_distance(
    position: tuple[float, float, float],
    reference: tuple[float, float, float],
    diameter: float,
) -> float:
    return float(np.linalg.norm(np.asarray(position) - np.asarray(reference)) / diameter)


def camera_health(
    observation: SemanticObservation,
    config: VerifierConfig,
    *,
    now: float,
) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    metrics: dict[str, Any] = {}
    age = float(now - observation.timestamp)
    metrics["observation_age_s"] = age
    if age < -0.1 or age > config.maximum_observation_age_s:
        reasons.append("stale_observation")
    timestamps = np.asarray(list(observation.camera_timestamps.values()), dtype=float)
    missing_cameras = sorted(
        set(config.required_cameras) - set(observation.camera_timestamps)
    )
    if missing_cameras:
        reasons.append("missing_required_cameras")
        metrics["missing_required_cameras"] = missing_cameras
    if timestamps.size:
        skew = float(np.max(timestamps) - np.min(timestamps))
        camera_age = float(now - np.max(timestamps))
        metrics["camera_timestamp_skew_s"] = skew
        metrics["newest_camera_age_s"] = camera_age
        if skew > config.maximum_camera_skew_s:
            reasons.append("camera_timestamp_skew")
        if camera_age < -0.1 or camera_age > config.maximum_observation_age_s:
            reasons.append("stale_camera")
    else:
        reasons.append("missing_camera_timestamps")
    if not observation.action_stream_healthy:
        reasons.append("action_stream_unhealthy")
    return reasons, metrics


class CheckpointVerifier:
    def __init__(self, config: VerifierConfig):
        self.config = config

    def evaluate(
        self,
        checkpoint: Checkpoint,
        observation: SemanticObservation,
        *,
        task: TaskSpec,
        reference: ReferenceEnvelope | None,
        initial: SemanticObservation,
        now: float | None = None,
    ) -> CheckpointDecision:
        now = time.time() if now is None else float(now)
        hard_reasons, metrics = camera_health(observation, self.config, now=now)
        unknown: list[str] = []
        position = observation.ee_position_xyz
        diameter = self.config.object_diameter_m

        if checkpoint is Checkpoint.INITIAL:
            if observation.gripper_open_fraction < self.config.gripper_open_threshold:
                hard_reasons.append("gripper_not_open_at_initial_checkpoint")
            if observation.target_visible is False:
                hard_reasons.append("target_not_visible")
            elif observation.target_visible is None:
                unknown.append("target_visibility_unknown")
            if observation.target_station is not None and observation.target_station != task.source:
                hard_reasons.append("target_not_at_expected_source")
            elif observation.target_station is None:
                unknown.append("target_station_unknown")

        elif checkpoint is Checkpoint.PRE_GRASP:
            if observation.target_visible is False:
                hard_reasons.append("target_not_visible_at_pregrasp")
            elif observation.target_visible is None:
                unknown.append("target_visibility_unknown_at_pregrasp")
            if reference is None:
                unknown.append("missing_reference_envelope")
            else:
                distance = _station_distance(position, reference.close_position_xyz, diameter)
                metrics["reference_close_distance_diameters"] = distance
                if distance > self.config.reference_position_tolerance_diameters:
                    hard_reasons.append("pregrasp_outside_reference_envelope")
            if observation.target_position_xyz is not None:
                distance = float(
                    np.linalg.norm(np.asarray(position) - np.asarray(observation.target_position_xyz))
                    / diameter
                )
                metrics["target_gripper_distance_diameters"] = distance

        elif checkpoint is Checkpoint.POST_GRASP:
            lift = float((position[2] - initial.ee_position_xyz[2]) / diameter)
            metrics["lift_diameters"] = lift
            if lift < self.config.minimum_lift_diameters:
                hard_reasons.append("insufficient_lift")
            if observation.target_attached is False:
                hard_reasons.append("target_not_attached")
            elif observation.target_attached is None:
                if reference is None or reference.closed_gripper_median is None:
                    unknown.append("attachment_not_observed")
                else:
                    error = abs(
                        observation.gripper_open_fraction - reference.closed_gripper_median
                    )
                    metrics["gripper_vs_success_reference"] = error
                    if error > self.config.maximum_closed_reference_error:
                        hard_reasons.append("gripper_closure_differs_from_success_reference")
                    else:
                        # Matching a successful finger opening is useful
                        # evidence, but cannot prove that the intended object
                        # (rather than another obstruction) is being carried.
                        unknown.append("attachment_only_inferred_from_gripper")

        elif checkpoint is Checkpoint.PRE_PLACE:
            if observation.target_attached is False:
                hard_reasons.append("target_lost_before_place")
            elif observation.target_attached is None:
                unknown.append("attachment_unknown_before_place")
            if reference is None:
                unknown.append("missing_reference_envelope")
            else:
                distance = _station_distance(position, reference.release_position_xyz, diameter)
                metrics["reference_release_distance_diameters"] = distance
                if distance > self.config.reference_position_tolerance_diameters:
                    hard_reasons.append("preplace_outside_reference_envelope")

        elif checkpoint is Checkpoint.POST_RELEASE:
            if observation.gripper_open_fraction < self.config.gripper_open_threshold:
                hard_reasons.append("gripper_not_open_after_release")
            if observation.target_attached is True:
                hard_reasons.append("target_still_attached")
            if observation.target_supported is False:
                hard_reasons.append("target_not_supported")
            elif observation.target_supported is None:
                unknown.append("support_state_unknown")
            if observation.target_station is not None and observation.target_station != task.destination:
                hard_reasons.append("target_not_at_destination")
            elif observation.target_station is None:
                unknown.append("destination_state_unknown")
            maximum_change = max(observation.visual_change_fraction.values(), default=0.0)
            metrics["maximum_visual_change_fraction"] = maximum_change
            if maximum_change < self.config.minimum_visual_change_fraction:
                unknown.append("no_task_relevant_visual_change")

        if hard_reasons:
            verdict = Verdict.REJECT
            reasons = tuple(hard_reasons + unknown)
        elif unknown:
            verdict = Verdict.UNCERTAIN
            reasons = tuple(unknown)
        else:
            verdict = Verdict.ACCEPT
            reasons = ("all_checkpoint_predicates_satisfied",)
        return CheckpointDecision(checkpoint, verdict, reasons, metrics, now)


def normalized_visual_difference(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    """Return resolution-independent visual change metrics."""

    first = np.asarray(previous)
    second = np.asarray(current)
    if first.ndim != 3 or second.ndim != 3:
        raise ValueError("visual difference expects HxWxC images")
    size = (160, 120)
    first_gray = cv2.resize(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), size)
    second_gray = cv2.resize(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), size)
    delta = cv2.absdiff(first_gray, second_gray).astype(np.float32) / 255.0
    return {
        "mean_absolute_fraction": float(np.mean(delta)),
        "changed_pixel_fraction": float(np.mean(delta >= 0.08)),
        "brightness_before": float(np.mean(first_gray) / 255.0),
        "brightness_after": float(np.mean(second_gray) / 255.0),
    }


class PetriTaskScheduler:
    """Cycle through reversible petri moves without a separate reset action."""

    def __init__(self, tasks: Iterable[TaskSpec], *, initial_station: str):
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise ValueError("at least one task is required")
        self.station = initial_station
        self.counts = {task.name: 0 for task in self.tasks}
        self.order = {task.name: index for index, task in enumerate(self.tasks)}

    def next_task(self) -> TaskSpec:
        candidates = [task for task in self.tasks if task.source == self.station]
        if not candidates:
            raise RuntimeError(f"no task leaves current station {self.station!r}")
        return min(
            candidates,
            key=lambda task: (self.counts[task.name], self.order[task.name]),
        )

    def record(self, task: TaskSpec, *, success: bool) -> None:
        self.counts[task.name] += 1
        if success:
            self.station = task.destination


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


class AuditTrail:
    def __init__(self, directory: str | Path, session: Mapping[str, Any]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "events.jsonl"
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = {"schema": SCHEMA, "session": dict(session), "episodes": []}
        _atomic_json(self.manifest_path, self.manifest)

    def event(self, kind: str, **payload: Any) -> None:
        value = {"schema": SCHEMA, "timestamp": time.time(), "kind": kind, **payload}
        with self.events_path.open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")

    def finish_episode(self, summary: Mapping[str, Any]) -> None:
        self.manifest["episodes"].append(dict(summary))
        _atomic_json(self.manifest_path, self.manifest)


class AgenticEpisode:
    """Pure checkpoint state machine with an auditable terminal class."""

    ORDER = (
        Checkpoint.INITIAL,
        Checkpoint.PRE_GRASP,
        Checkpoint.POST_GRASP,
        Checkpoint.PRE_PLACE,
        Checkpoint.POST_RELEASE,
    )

    def __init__(
        self,
        task: TaskSpec,
        verifier: CheckpointVerifier,
        reference: ReferenceEnvelope | None,
        initial: SemanticObservation,
    ):
        self.task = task
        self.verifier = verifier
        self.reference = reference
        self.initial = initial
        self.decisions: list[CheckpointDecision] = []
        self.interventions = 0
        self.recoveries = 0
        self.invalid_reasons: list[str] = []

    @property
    def next_checkpoint(self) -> Checkpoint | None:
        index = len(self.decisions)
        return self.ORDER[index] if index < len(self.ORDER) else None

    def check(
        self,
        checkpoint: Checkpoint,
        observation: SemanticObservation,
        *,
        now: float | None = None,
    ) -> CheckpointDecision:
        expected = self.next_checkpoint
        if checkpoint is not expected:
            raise ValueError(f"expected checkpoint {expected}, got {checkpoint}")
        decision = self.verifier.evaluate(
            checkpoint,
            observation,
            task=self.task,
            reference=self.reference,
            initial=self.initial,
            now=now,
        )
        self.decisions.append(decision)
        return decision

    def mark_intervention(self) -> None:
        self.interventions += 1

    def mark_recovery(self) -> None:
        self.recoveries += 1

    def mark_invalid(self, reason: str) -> None:
        self.invalid_reasons.append(str(reason))

    def classify(self) -> EpisodeClass:
        if self.invalid_reasons:
            return EpisodeClass.INVALID
        if any(value.verdict is Verdict.REJECT for value in self.decisions):
            return EpisodeClass.FAILURE
        if len(self.decisions) != len(self.ORDER) or any(
            value.verdict is Verdict.UNCERTAIN for value in self.decisions
        ):
            return EpisodeClass.UNCERTAIN
        if self.interventions or self.recoveries:
            return EpisodeClass.RECOVERY
        return EpisodeClass.CLEAN_SUCCESS

    def summary(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "task": asdict(self.task),
            "classification": self.classify().value,
            "interventions": self.interventions,
            "recoveries": self.recoveries,
            "invalid_reasons": self.invalid_reasons,
            "decisions": [value.to_dict() for value in self.decisions],
        }


@dataclass
class SkillEvidence:
    episode_id: str
    task: str
    start_bin: str
    condition: str
    verified: bool


class SkillRegistry:
    """Evidence-gated registry; occurrence count alone can never promote."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.value = (
            json.loads(self.path.read_text())
            if self.path.exists()
            else {"schema": "piper_robot.agentic_skill_registry/v1", "skills": {}}
        )

    def add_candidate(
        self,
        name: str,
        *,
        primitive_trace: list[Mapping[str, Any]],
        preconditions: Mapping[str, Any],
        postconditions: Mapping[str, Any],
        evidence: SkillEvidence,
    ) -> None:
        canonical = json.dumps(primitive_trace, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        skill = self.value["skills"].setdefault(
            name,
            {
                "trace_sha256": digest,
                "primitive_trace": primitive_trace,
                "preconditions": dict(preconditions),
                "postconditions": dict(postconditions),
                "evidence": [],
                "simulation_passed": False,
                "shadow_passed": False,
                "promoted": False,
            },
        )
        if skill["trace_sha256"] != digest:
            raise ValueError(f"skill {name!r} changed trace; create a new version")
        item = asdict(evidence)
        if item not in skill["evidence"]:
            skill["evidence"].append(item)
        _atomic_json(self.path, self.value)

    def mark_validation(self, name: str, *, simulation: bool, shadow: bool) -> None:
        skill = self.value["skills"][name]
        skill["simulation_passed"] = bool(simulation)
        skill["shadow_passed"] = bool(shadow)
        _atomic_json(self.path, self.value)

    def promote(self, name: str, *, minimum_clean_successes: int = 5) -> None:
        skill = self.value["skills"][name]
        verified = [item for item in skill["evidence"] if item["verified"]]
        bins = {item["start_bin"] for item in verified}
        if len(verified) < minimum_clean_successes:
            raise ValueError("skill has insufficient verified clean successes")
        if len(bins) < 2:
            raise ValueError("skill has not generalized across start bins")
        if not skill["simulation_passed"] or not skill["shadow_passed"]:
            raise ValueError("skill must pass simulation and shadow validation")
        skill["promoted"] = True
        _atomic_json(self.path, self.value)


def load_profile(path: str | Path) -> tuple[list[TaskSpec], VerifierConfig, dict[str, Any]]:
    profile_path = Path(path).resolve()
    value = json.loads(profile_path.read_text())
    if value.get("schema") != "piper_robot.agentic_collection_profile/v1":
        raise ValueError("unsupported agentic collection profile schema")
    tasks = [TaskSpec(**item) for item in value["tasks"]]
    verifier = VerifierConfig(**value["verifier"])
    return tasks, verifier, value
