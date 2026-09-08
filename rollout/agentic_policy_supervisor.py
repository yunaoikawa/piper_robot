"""ACT-facing adapter for :mod:`rollout.agentic_collection`.

It infers semantic checkpoints from gripper transitions, but never inserts work
into the 30 Hz command path.  Expensive perception may enrich observations via
the ``semantic`` field; otherwise calibrated demonstrations and synchronized
visual change provide the conservative baseline verifier.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import glob
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
import cv2

from .agentic_collection import (
    AgenticEpisode,
    AuditTrail,
    Checkpoint,
    CheckpointVerifier,
    CollectionMode,
    EpisodeClass,
    PetriTaskScheduler,
    ReferenceEnvelope,
    SemanticObservation,
    TaskSpec,
    Verdict,
    load_profile,
    normalized_visual_difference,
)
from .sam_segmentation import SamSegmentationClient


class AgenticPolicySupervisor:
    """Supervise one policy controller without owning cameras or robot RPCs."""

    def __init__(
        self,
        profile_path: str | Path,
        *,
        mode: str = "shadow",
        condition: str = "unknown",
        armed: bool = False,
        output_dir: str | Path = "data/agentic_collection",
        semantic_provider: Callable[
            [Checkpoint, Mapping[str, Any], SemanticObservation], Mapping[str, Any]
        ] | None = None,
        now=time.time,
    ):
        tasks, verifier_config, profile = load_profile(profile_path)
        self.profile_path = Path(profile_path).resolve()
        self.profile = profile
        self.mode = CollectionMode(mode)
        self.condition = str(condition)
        self.armed = bool(armed)
        self.now = now
        self.semantic_provider = semantic_provider
        self.verifier = CheckpointVerifier(verifier_config)
        self.scheduler = PetriTaskScheduler(
            tasks, initial_station=str(profile["initial_station"])
        )
        self.references = {
            task.name: self._load_reference(task) for task in tasks
        }
        reference_graph = self._validate_reference_graph(tasks)
        sam_config = profile.get("sam") or {}
        self._sam = (
            SamSegmentationClient(
                str(sam_config["endpoint"]),
                timeout_ms=int(sam_config.get("timeout_ms", 5000)),
            )
            if sam_config.get("enabled", False)
            else None
        )
        self._sam_config = sam_config
        self._sam_frame_id = 0
        self._initial_target_uv = None
        self._initial_target_area_fraction = None
        self._last_sam_mask = None
        self._target_attached_confirmed: bool | None = None
        session_name = time.strftime("%Y%m%d_%H%M%S", time.localtime(now()))
        self.run_dir = Path(output_dir).resolve() / session_name
        self.audit = AuditTrail(
            self.run_dir,
            {
                "profile": str(self.profile_path),
                "mode": self.mode.value,
                "condition": self.condition,
                "armed": self.armed,
                "reference_graph": reference_graph,
            },
        )
        self.task = self.scheduler.next_task()
        self.episode: AgenticEpisode | None = None
        self.latest: SemanticObservation | None = None
        self._latest_raw: Mapping[str, Any] | None = None
        self._initial_images: dict[str, np.ndarray] = {}
        self._commanded_gripper_open = True
        self._terminal: tuple[EpisodeClass, str] | None = None
        self._held_uncertain = False
        self._resume_transition: Checkpoint | None = None
        self._resume_replan = False
        self._lock = threading.RLock()
        self._stable_release = 0

    def _validate_reference_graph(self, tasks: list[TaskSpec]) -> list[dict[str, Any]]:
        """Reject profiles whose recorded endpoints do not form their task graph."""
        report: list[dict[str, Any]] = []
        require_cycle = bool(self.profile.get("require_closed_cycle", False))
        for task in tasks:
            followers = [value for value in tasks if value.source == task.destination]
            if require_cycle and len(followers) != 1:
                raise ValueError(
                    f"closed-cycle task {task.name!r} has {len(followers)} followers"
                )
            current = self.references.get(task.name)
            comparable = [
                (value, self.references.get(value.name))
                for value in followers
                if self.references.get(value.name) is not None
            ]
            if current is None or not comparable:
                continue
            distances = [
                (
                    follower.name,
                    float(
                        np.linalg.norm(
                            np.asarray(current.release_position_xyz)
                            - np.asarray(reference.close_position_xyz)
                        )
                        / self.verifier.config.object_diameter_m
                    ),
                )
                for follower, reference in comparable
            ]
            follower_name, distance = min(distances, key=lambda item: item[1])
            report.append(
                {
                    "task": task.name,
                    "follower": follower_name,
                    "release_to_next_close_diameters": distance,
                }
            )
            if (
                require_cycle
                and distance
                > self.verifier.config.reference_position_tolerance_diameters
            ):
                raise ValueError(
                    f"reference graph discontinuity {task.name!r}->{follower_name!r}: "
                    f"{distance:.3f} object diameters"
                )
        return report

    def _load_reference(self, task: TaskSpec) -> ReferenceEnvelope | None:
        if not task.reference_glob:
            return None
        pattern = str(task.reference_glob).format(condition=self.condition)
        if not Path(pattern).is_absolute():
            pattern = str(self.profile_path.parents[2] / pattern)
        paths = sorted(glob.glob(pattern))
        if not paths:
            return None
        try:
            return ReferenceEnvelope.from_hdf5(paths, arm=task.arm)
        except ValueError:
            return None

    @property
    def instruction(self) -> str:
        return self.task.instruction

    @property
    def commands_enabled(self) -> bool:
        return self.mode is not CollectionMode.SHADOW and self.armed

    @staticmethod
    def _arm_position(observation: Mapping[str, Any], arm: str) -> tuple[float, float, float]:
        qpos = np.asarray(observation["qpos"], dtype=float)
        offset = 0 if arm == "left" else 10
        if qpos.shape != (20,):
            raise ValueError("agentic supervisor expects the 20D bimanual qpos")
        return tuple(float(value) for value in qpos[offset : offset + 3])

    @staticmethod
    def _arm_gripper(observation: Mapping[str, Any], arm: str) -> float:
        qpos = np.asarray(observation["qpos"], dtype=float)
        return float(qpos[9 if arm == "left" else 19])

    def _semantic_observation(self, raw: Mapping[str, Any]) -> SemanticObservation:
        semantic = dict(raw.get("semantic") or {})
        images = raw.get("images") or {}
        visual_changes: dict[str, float] = {}
        for name, current in images.items():
            if current is None:
                continue
            array = np.asarray(current)
            initial = self._initial_images.get(name)
            if initial is not None and initial.shape == array.shape:
                change = normalized_visual_difference(initial, array)
                visual_changes[name] = change["changed_pixel_fraction"]
        camera_timestamps = {
            str(name): float(value)
            for name, value in (raw.get("camera_timestamps") or {}).items()
            if value is not None
        }
        reference = self.references.get(self.task.name)
        position = self._arm_position(raw, self.task.arm)
        gripper = self._arm_gripper(raw, self.task.arm)
        diameter = self.verifier.config.object_diameter_m
        close_distance = (
            float(np.linalg.norm(np.asarray(position) - reference.close_position_xyz) / diameter)
            if reference is not None
            else float("inf")
        )
        release_distance = (
            float(np.linalg.norm(np.asarray(position) - reference.release_position_xyz) / diameter)
            if reference is not None
            else float("inf")
        )
        tolerance = self.verifier.config.reference_position_tolerance_diameters
        cameras_valid = sum(
            int(np.asarray(image).ndim == 3 and np.asarray(image).size > 0 and np.percentile(image, 95) > 8)
            for image in images.values()
            if image is not None
        )
        # Global image change is evidence, never a support predicate: arm
        # motion alone can change many pixels.  Support requires external
        # semantics or a fresh checkpoint SAM mask plus release geometry.
        inferred_supported = None
        return SemanticObservation(
            timestamp=float(raw.get("timestamp", self.now())),
            ee_position_xyz=position,
            gripper_open_fraction=gripper,
            camera_timestamps=camera_timestamps,
            visual_change_fraction=visual_changes,
            target_visible=semantic.get("target_visible"),
            target_station=semantic.get(
                "target_station",
                self.scheduler.station,
            ),
            target_position_xyz=semantic.get("target_position_xyz"),
            target_attached=semantic.get(
                "target_attached", self._target_attached_confirmed
            ),
            target_supported=semantic.get("target_supported", inferred_supported),
            action_stream_healthy=bool(raw.get("action_stream_healthy", True)),
            evidence={
                "semantic_source": (
                    "external" if raw.get("semantic") else "reference_geometry_and_visual_change"
                ),
                "valid_camera_count": cameras_valid,
                "reference_close_distance_diameters": close_distance,
                "reference_release_distance_diameters": release_distance,
                **dict(semantic.get("evidence") or {}),
            },
        )

    def begin_episode(self, raw: Mapping[str, Any]) -> None:
        with self._lock:
            if self.episode is not None:
                return
            self.task = self.scheduler.next_task()
            self._initial_images = {
                name: np.asarray(image).copy()
                for name, image in (raw.get("images") or {}).items()
                if image is not None
            }
            # Checkpoint perception reads the raw image/depth bundle. Publish
            # it before enriching the initial semantic observation.
            self._latest_raw = raw
            initial = self._semantic_observation(raw)
            initial = self._enrich_with_sam(Checkpoint.INITIAL, initial)
            initial = self._enrich_if_uncertain(Checkpoint.INITIAL, initial)
            self.episode = AgenticEpisode(
                self.task,
                self.verifier,
                self.references.get(self.task.name),
                initial,
            )
            self.latest = initial
            self._commanded_gripper_open = True
            self._target_attached_confirmed = None
            self._stable_release = 0
            self._terminal = None
            self._held_uncertain = False
            self._resume_transition = None
            self._resume_replan = False
            decision = self.episode.check(Checkpoint.INITIAL, initial, now=self.now())
            artifacts = self._save_checkpoint_artifacts(Checkpoint.INITIAL, initial)
            self.audit.event(
                "checkpoint", task=self.task.name, decision=decision.to_dict(),
                artifacts=artifacts,
            )
            if decision.verdict is Verdict.REJECT:
                self._terminal = (self.episode.classify(), "initial_checkpoint_veto")
            elif decision.verdict is Verdict.UNCERTAIN:
                self._held_uncertain = True

    def observe(self, raw: Mapping[str, Any]) -> None:
        with self._lock:
            self._latest_raw = raw
            if self.episode is None:
                return
            self.latest = self._semantic_observation(raw)
            if self.latest.target_attached is not None:
                self._target_attached_confirmed = self.latest.target_attached
            if self.episode.next_checkpoint is Checkpoint.POST_GRASP:
                diameter = self.verifier.config.object_diameter_m
                lift = (
                    self.latest.ee_position_xyz[2]
                    - self.episode.initial.ee_position_xyz[2]
                ) / diameter
                if lift >= self.verifier.config.minimum_lift_diameters:
                    self._check(Checkpoint.POST_GRASP)
            elif self.episode.next_checkpoint is Checkpoint.POST_RELEASE:
                if self.latest.gripper_open_fraction >= self.verifier.config.gripper_open_threshold:
                    self._stable_release += 1
                else:
                    self._stable_release = 0
                if self._stable_release >= self.verifier.config.stable_observations:
                    decision = self._check(Checkpoint.POST_RELEASE)
                    if decision.verdict is Verdict.ACCEPT:
                        self._terminal = (EpisodeClass.CLEAN_SUCCESS, "verified_success")

    def _check(self, checkpoint: Checkpoint):
        if self.episode is None or self.latest is None:
            raise RuntimeError("no active agentic episode")
        self.latest = self._enrich_with_sam(checkpoint, self.latest)
        self.latest = self._enrich_if_uncertain(checkpoint, self.latest)
        decision = self.episode.check(checkpoint, self.latest, now=self.now())
        artifacts = self._save_checkpoint_artifacts(checkpoint, self.latest)
        self.audit.event(
            "checkpoint", task=self.task.name, decision=decision.to_dict(),
            artifacts=artifacts,
        )
        if decision.verdict is Verdict.REJECT:
            self._terminal = (self.episode.classify(), f"{checkpoint.value}_veto")
        elif decision.verdict is Verdict.UNCERTAIN:
            self._held_uncertain = True
        return decision

    def _enrich_if_uncertain(
        self,
        checkpoint: Checkpoint,
        semantic: SemanticObservation,
    ) -> SemanticObservation:
        """Call an optional reasoner only for deterministic uncertainty.

        The provider runs while held at a checkpoint and may supply semantic
        facts. It never owns, sees, or returns robot transport commands.
        """
        if self.semantic_provider is None or self._latest_raw is None:
            return semantic
        provisional = self.verifier.evaluate(
            checkpoint,
            semantic,
            task=self.task,
            reference=self.references.get(self.task.name),
            initial=(self.episode.initial if self.episode is not None else semantic),
            now=self.now(),
        )
        if provisional.verdict is not Verdict.UNCERTAIN:
            return semantic
        try:
            supplied = dict(
                self.semantic_provider(checkpoint, self._latest_raw, semantic) or {}
            )
        except BaseException as error:
            return replace(
                semantic,
                evidence={
                    **dict(semantic.evidence),
                    "semantic_provider_error": f"{type(error).__name__}: {error}",
                },
            )
        allowed = {
            "target_visible",
            "target_station",
            "target_position_xyz",
            "target_attached",
            "target_supported",
        }
        updates = {key: supplied[key] for key in allowed if key in supplied}
        updates["evidence"] = {
            **dict(semantic.evidence),
            "semantic_provider": supplied.get("provider", "injected"),
            **dict(supplied.get("evidence") or {}),
        }
        if updates.get("target_attached") is not None:
            self._target_attached_confirmed = bool(updates["target_attached"])
        return replace(semantic, **updates)

    def _enrich_with_sam(
        self,
        checkpoint: Checkpoint,
        semantic: SemanticObservation,
    ) -> SemanticObservation:
        """Run SAM only at a stopped semantic checkpoint.

        SAM establishes object presence.  Station identity still comes from
        task-relative measured geometry, so a confident mask alone can never
        claim successful placement.
        """
        if self._sam is None or self._latest_raw is None:
            return semantic
        image = (self._latest_raw.get("images") or {}).get("cam_high")
        if image is None:
            return replace(
                semantic,
                target_visible=False,
                evidence={**dict(semantic.evidence), "sam_error": "missing_head_image"},
            )
        image = np.asarray(image)
        if str(self._sam_config.get("input_color", "rgb")).lower() == "rgb":
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        self._sam_frame_id += 1
        try:
            result = self._sam.segment(
                image_bgr,
                frame_id=self._sam_frame_id,
                timestamp=float(semantic.timestamp),
                prompt=str(self._sam_config.get("prompt", "petri dish")),
                confidence_threshold=float(
                    self._sam_config.get("confidence_threshold", 0.25)
                ),
            )
        except BaseException as error:
            return replace(
                semantic,
                target_visible=None,
                evidence={
                    **dict(semantic.evidence),
                    "sam_error": f"{type(error).__name__}: {error}",
                },
            )
        height, width = image_bgr.shape[:2]
        candidates = []
        for candidate in result.candidates:
            ys, xs = np.nonzero(candidate.mask)
            if not len(xs):
                continue
            uv = np.asarray([np.mean(xs) / width, np.mean(ys) / height], dtype=float)
            area = float(np.mean(candidate.mask))
            area_cost = (
                0.0
                if self._initial_target_area_fraction is None
                else abs(math.log(max(area, 1e-9) / self._initial_target_area_fraction))
            )
            candidates.append((area_cost - 0.25 * float(candidate.score), uv, area, candidate))
        if not candidates:
            return replace(
                semantic,
                target_visible=False,
                evidence={
                    **dict(semantic.evidence),
                    "sam_model": result.model,
                    "sam_candidate_count": 0,
                },
            )
        _, uv, area, candidate = min(candidates, key=lambda item: item[0])
        self._last_sam_mask = np.asarray(candidate.mask, dtype=np.uint8) * 255
        if checkpoint is Checkpoint.INITIAL:
            self._initial_target_uv = uv.copy()
            self._initial_target_area_fraction = area
        displacement = (
            None
            if self._initial_target_uv is None
            else float(np.linalg.norm(uv - self._initial_target_uv))
        )
        evidence = {
            **dict(semantic.evidence),
            "sam_model": result.model,
            "sam_inference_ms": result.inference_ms,
            "sam_candidate_count": len(candidates),
            "sam_score": float(candidate.score),
            "sam_center_uv": uv.tolist(),
            "sam_area_fraction": area,
            "sam_displacement_image_diagonal_fraction": displacement,
        }
        supported = semantic.target_supported
        station = semantic.target_station
        attached = semantic.target_attached
        if checkpoint is Checkpoint.POST_GRASP:
            minimum_displacement = float(
                self._sam_config.get(
                    "minimum_transport_displacement_diagonal_fraction", 0.01
                )
            )
            lift = (
                0.0
                if self.episode is None
                else (
                    semantic.ee_position_xyz[2]
                    - self.episode.initial.ee_position_xyz[2]
                )
                / self.verifier.config.object_diameter_m
            )
            evidence["sam_transport_displacement_threshold"] = minimum_displacement
            evidence["lift_diameters_at_sam_check"] = lift
            if displacement is not None and displacement >= minimum_displacement:
                attached = True
                self._target_attached_confirmed = True
        if checkpoint is Checkpoint.POST_RELEASE:
            reference = self.references.get(self.task.name)
            if reference is not None:
                distance = float(
                    np.linalg.norm(
                        np.asarray(semantic.ee_position_xyz)
                        - np.asarray(reference.release_position_xyz)
                    )
                    / self.verifier.config.object_diameter_m
                )
                if distance <= self.verifier.config.reference_position_tolerance_diameters:
                    supported = True
                    station = self.task.destination
                    if (
                        semantic.gripper_open_fraction
                        >= self.verifier.config.gripper_open_threshold
                    ):
                        attached = False
                        self._target_attached_confirmed = False
        return replace(
            semantic,
            target_visible=True,
            target_station=station,
            target_attached=attached,
            target_supported=supported,
            evidence=evidence,
        )

    def _save_checkpoint_artifacts(
        self,
        checkpoint: Checkpoint,
        semantic: SemanticObservation,
    ) -> dict[str, str]:
        if self._latest_raw is None:
            return {}
        episode_number = len(self.audit.manifest["episodes"]) + 1
        directory = self.run_dir / f"episode_{episode_number:04d}" / checkpoint.value
        directory.mkdir(parents=True, exist_ok=True)
        result: dict[str, str] = {}
        for name, image in (self._latest_raw.get("images") or {}).items():
            if image is None:
                continue
            path = directory / f"{name}.png"
            cv2.imwrite(str(path), np.asarray(image))
            result[name] = str(path)
        depth = self._latest_raw.get("depth")
        if depth is not None:
            path = directory / "head_depth.npz"
            np.savez_compressed(path, depth=np.asarray(depth))
            result["head_depth"] = str(path)
        if self._last_sam_mask is not None:
            path = directory / "sam_mask.png"
            cv2.imwrite(str(path), self._last_sam_mask)
            result["sam_mask"] = str(path)
        semantic_path = directory / "semantic.json"
        semantic_path.write_text(
            json.dumps(asdict(semantic), indent=2, ensure_ascii=False, default=str)
            + "\n"
        )
        result["semantic"] = str(semantic_path)
        return result

    @staticmethod
    def _command_gripper(action: Mapping[str, Any], arm: str) -> float | None:
        value = action.get(f"{arm}_gripper")
        return None if value is None else float(value)

    def before_action(self, action: Mapping[str, Any]) -> bool:
        with self._lock:
            if not self.commands_enabled or self.episode is None or self.latest is None:
                return False
            if self._held_uncertain:
                return False
            value = self._command_gripper(action, self.task.arm)
            if value is None:
                return True
            # Gate on the first departure from an endpoint. Waiting until the
            # opposite threshold would let a continuous ACT action partially
            # close on an unverified target or partially release too early.
            closing = (
                self._commanded_gripper_open
                and value < self.verifier.config.gripper_open_threshold
            )
            opening = (
                (not self._commanded_gripper_open)
                and value > self.verifier.config.gripper_closed_threshold
            )
            if closing:
                if self._resume_transition is Checkpoint.PRE_GRASP:
                    self._resume_transition = None
                    return True
                if self.episode.next_checkpoint is not Checkpoint.PRE_GRASP:
                    self._terminal = (EpisodeClass.INVALID, "unexpected_close_transition")
                    return False
                return self._check(Checkpoint.PRE_GRASP).verdict is Verdict.ACCEPT
            if opening:
                if self._resume_transition is Checkpoint.PRE_PLACE:
                    self._resume_transition = None
                    return True
                if self.episode.next_checkpoint is Checkpoint.POST_GRASP:
                    if self._check(Checkpoint.POST_GRASP).verdict is not Verdict.ACCEPT:
                        return False
                if self.episode.next_checkpoint is not Checkpoint.PRE_PLACE:
                    self._terminal = (EpisodeClass.INVALID, "unexpected_open_transition")
                    return False
                return self._check(Checkpoint.PRE_PLACE).verdict is Verdict.ACCEPT
            return True

    def after_action(self, action: Mapping[str, Any]) -> None:
        with self._lock:
            value = self._command_gripper(action, self.task.arm)
            if value is None:
                return
            if value <= self.verifier.config.gripper_closed_threshold:
                self._commanded_gripper_open = False
            elif value >= self.verifier.config.gripper_open_threshold:
                self._commanded_gripper_open = True

    def terminal_request(self) -> tuple[EpisodeClass, str] | None:
        with self._lock:
            return self._terminal

    @property
    def held_uncertain(self) -> bool:
        with self._lock:
            return self._held_uncertain

    def consume_resume_replan(self) -> bool:
        with self._lock:
            value = self._resume_replan
            self._resume_replan = False
            return value

    def request_hold(self, reason: str = "operator_hold") -> None:
        with self._lock:
            if self.episode is not None:
                self._held_uncertain = False
                self._terminal = (EpisodeClass.UNCERTAIN, str(reason))
                self.audit.event("operator_hold", task=self.task.name, reason=reason)

    def request_intervention(self, reason: str = "teleop_takeover") -> None:
        with self._lock:
            if self.episode is not None:
                self.episode.mark_intervention()
                self._held_uncertain = False
                self._terminal = (EpisodeClass.RECOVERY, str(reason))
                self.audit.event("teleop_takeover", task=self.task.name, reason=reason)

    def request_invalid(self, reason: str) -> None:
        """Fail closed after observation/state-machine integrity errors."""
        with self._lock:
            if self.episode is not None:
                self.episode.mark_invalid(str(reason))
            self._held_uncertain = False
            self._terminal = (EpisodeClass.INVALID, str(reason))
            self.audit.event("invalid", task=self.task.name, reason=str(reason))

    def override_uncertain_checkpoint(self, note: str) -> None:
        """Let a human continue an uncertain gate, never a rejected gate.

        The episode is marked as intervention/recovery and therefore remains
        excluded from the clean ACT dataset.
        """
        with self._lock:
            if self.episode is None or not self.episode.decisions:
                raise RuntimeError("no checkpoint is available to override")
            decision = self.episode.decisions[-1]
            if decision.verdict is not Verdict.UNCERTAIN:
                raise ValueError("only an uncertain checkpoint may be overridden")
            self.episode.decisions[-1] = replace(
                decision,
                verdict=Verdict.ACCEPT,
                reasons=("operator_override_uncertain", str(note)),
            )
            self.episode.mark_intervention()
            self._held_uncertain = False
            self._resume_transition = (
                decision.checkpoint
                if decision.checkpoint in {Checkpoint.PRE_GRASP, Checkpoint.PRE_PLACE}
                else None
            )
            self._resume_replan = True
            self._terminal = (
                (EpisodeClass.RECOVERY, "operator_override_completed")
                if decision.checkpoint is Checkpoint.POST_RELEASE
                else None
            )
            self.audit.event(
                "checkpoint_override",
                task=self.task.name,
                checkpoint=decision.checkpoint.value,
                note=str(note),
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "piper_robot.agentic_collection_status/v1",
                "mode": self.mode.value,
                "armed": self.armed,
                "condition": self.condition,
                "task": asdict(self.task),
                "station": self.scheduler.station,
                "checkpoint": (
                    None
                    if self.episode is None or self.episode.next_checkpoint is None
                    else self.episode.next_checkpoint.value
                ),
                "terminal": (
                    None
                    if self._terminal is None
                    else {
                        "classification": self._terminal[0].value,
                        "reason": self._terminal[1],
                    }
                ),
                "held_uncertain": self._held_uncertain,
                "last_decision": (
                    None
                    if self.episode is None or not self.episode.decisions
                    else self.episode.decisions[-1].to_dict()
                ),
                "run_dir": str(self.run_dir),
            }

    def mark_intervention(self) -> None:
        with self._lock:
            if self.episode is not None:
                self.episode.mark_intervention()
                self.audit.event("intervention", task=self.task.name)

    def finish(self, *, forced_class: EpisodeClass | None = None, reason: str = "") -> dict[str, Any]:
        with self._lock:
            if self.episode is None:
                return {}
            summary = self.episode.summary()
            if forced_class is not None:
                summary["classification"] = forced_class.value
            summary.update(
                {
                    "reason": reason,
                    "condition": self.condition,
                    "mode": self.mode.value,
                    "reference": (
                        asdict(self.references[self.task.name])
                        if self.references.get(self.task.name) is not None
                        else None
                    ),
                }
            )
            success = summary["classification"] == EpisodeClass.CLEAN_SUCCESS.value
            self.scheduler.record(self.task, success=success)
            self.audit.finish_episode(summary)
            self.audit.event("episode_finished", summary=summary)
            self.episode = None
            self.latest = None
            self._latest_raw = None
            self._initial_images = {}
            self._last_sam_mask = None
            self._target_attached_confirmed = None
            self._terminal = None
            self._held_uncertain = False
            self._resume_transition = None
            self._resume_replan = False
            self.task = self.scheduler.next_task()
            return summary

    def close(self) -> None:
        if self._sam is not None:
            self._sam.close()
