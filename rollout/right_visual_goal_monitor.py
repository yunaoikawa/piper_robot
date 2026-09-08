"""Asynchronous right-wrist SAM monitor for the accepted grasp image goal."""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from rollout.grasp_window import GraspWindowTemplate
from rollout.right_target_observer import RightTargetObserver
from rollout.sam_segmentation import SamSegmentationClient


class RightVisualGoalMonitor:
    """Keep SAM inference off the 30 Hz motion thread.

    The accepted image is used only as a final tool-relative grasp gate.  It
    does not provide joint positions, a Cartesian trajectory, or a gripper
    command.
    """

    def __init__(
        self,
        *,
        sam_endpoint: str,
        output_dir: str | Path,
        prompts,
        selection: dict,
        target_semantic_role: str,
        locked_instance_id: str,
        minimum_mask_circularity=None,
        timeout_ms: int = 20000,
    ):
        self.stop_event = threading.Event()
        self.runner = SimpleNamespace(
            stop_event=self.stop_event,
            sam=SamSegmentationClient(sam_endpoint, timeout_ms=timeout_ms),
            frame_id=100000,
        )
        self.observer = RightTargetObserver(
            self.runner,
            Path(output_dir),
            prompts=prompts,
            grasp_window_template=GraspWindowTemplate.from_dict(
                selection["template"]
            ),
            grasp_window_method=str(selection["selected_method"]),
            minimum_mask_circularity=minimum_mask_circularity,
        )
        self._lock = threading.Lock()
        self.target_semantic_role = str(target_semantic_role)
        self.locked_instance_id = str(locked_instance_id)
        if not self.target_semantic_role or not self.locked_instance_id:
            raise ValueError("right visual monitor requires a semantic identity lock")
        self._latest = None
        self._error = None
        self._thread = None

    @classmethod
    def from_files(
        cls,
        *,
        sam_endpoint: str,
        output_dir: str | Path,
        selection_path: str | Path,
        task_path: str | Path,
        locked_instance_id: str,
    ):
        selection = json.loads(Path(selection_path).read_text())
        task = json.loads(Path(task_path).read_text())
        target = task["target"]
        return cls(
            sam_endpoint=sam_endpoint,
            output_dir=output_dir,
            selection=selection,
            target_semantic_role=target["semantic_role"],
            locked_instance_id=locked_instance_id,
            # Runtime latency matters more than broad offline recall here.
            # Re-running three prompts on one stale frame can consume the
            # entire approach.  The task-specific primary prompt is the one
            # validated for this target; each completed request can therefore
            # advance immediately to a fresh wrist frame.
            prompts=(target["sam_prompt"],),
            minimum_mask_circularity=target.get(
                "minimum_mask_circularity"
            ),
        )

    def start(self) -> None:
        self.observer.start()
        self._thread = threading.Thread(
            target=self._run,
            name="right-visual-goal-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                geometry, _, image, camera_timestamp = self.observer.observe(
                    require_target=False
                )
                artifacts = self.observer.last_observation_artifacts or {}
                assessment = artifacts.get("visual_goal_assessment")
                record = {
                    "observed_at_monotonic_s": time.monotonic(),
                    "camera_timestamp_s": float(camera_timestamp),
                    "image": image,
                    "target_visible": geometry is not None,
                    "semantic_role": self.target_semantic_role,
                    "instance_id": self.locked_instance_id,
                    "area_fraction": (
                        None if geometry is None else geometry.area_fraction
                    ),
                    "assessment": assessment,
                }
                with self._lock:
                    self._latest = record
                    self._error = None
            except BaseException as error:
                if self.stop_event.is_set():
                    return
                with self._lock:
                    self._error = error
                time.sleep(0.05)

    def latest(self) -> dict | None:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def require_close_allowed(self, *, maximum_age_s: float = 3.0) -> dict:
        with self._lock:
            latest = None if self._latest is None else dict(self._latest)
            error = self._error
        if latest is None:
            if error is not None:
                raise RuntimeError(
                    f"right visual goal has no observation: {error}"
                ) from error
            raise RuntimeError("right visual goal has no observation")
        age = time.monotonic() - latest["observed_at_monotonic_s"]
        if age > float(maximum_age_s):
            raise RuntimeError(
                f"right visual goal is stale: age={age:.2f}s"
            )
        assessment = latest.get("assessment")
        if not latest.get("target_visible") or not isinstance(assessment, dict):
            raise RuntimeError(
                f"right SAM target is not visible; image={latest.get('image')}"
            )
        if assessment.get("allowed_to_close") is not True:
            raise RuntimeError(
                "right visual goal rejected closure: "
                f"{assessment.get('failure_reasons')}; "
                f"image={latest.get('image')}"
            )
        if (
            latest.get("semantic_role") != self.target_semantic_role
            or latest.get("instance_id") != self.locked_instance_id
        ):
            raise RuntimeError("right visual goal lost its semantic identity lock")
        return latest

    def stop(self) -> None:
        self.stop_event.set()
        stop_error = None
        try:
            self.observer.stop()
        except BaseException as error:
            stop_error = error
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.runner.sam.close()
        if stop_error is not None:
            raise stop_error
