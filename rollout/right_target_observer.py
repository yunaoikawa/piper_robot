"""Generic right-wrist SAM observer ranked in the demonstrated tool frame."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rollout.grasp_window import (
    GraspWindowAssessment,
    GraspWindowTemplate,
    assess_grasp_window,
)


@dataclass(frozen=True)
class TargetImageGeometry:
    center_px: np.ndarray
    area_fraction: float


def choose_tool_relative_target(
    candidates,
    image_bgr,
    template: GraspWindowTemplate,
    *,
    method,
    previous_center_px=None,
):
    """Choose a SAM candidate without requiring a tag or coloured marker."""

    image = np.asarray(image_bgr)
    diagonal = float(np.hypot(*image.shape[:2]))
    ranked = []
    for candidate in candidates:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.shape != image.shape[:2] or np.count_nonzero(mask) < 100:
            continue
        try:
            assessment, _ = assess_grasp_window(
                image,
                mask,
                template,
                method=method,
            )
        except ValueError:
            continue
        ys, xs = np.nonzero(mask)
        center = np.asarray([np.mean(xs), np.mean(ys)], dtype=float)
        continuity = 0.0
        if previous_center_px is not None:
            continuity = float(
                np.linalg.norm(center - np.asarray(previous_center_px))
                / diagonal
            )
        # Every term is dimensionless. Near the final grasp, matching the
        # demonstrated tool-relative geometry outweighs SAM's raw score.
        objective = (
            assessment.normalized_center_error
            + assessment.normalized_quantile_error
            + (1.0 - assessment.target_inside_fraction)
            + continuity
            - 0.05 * float(candidate.score)
        )
        ranked.append((objective, -float(candidate.score), candidate, assessment, center))
    if not ranked:
        return None
    objective, _, candidate, assessment, center = min(
        ranked, key=lambda value: (value[0], value[1])
    )
    geometry = TargetImageGeometry(
        center_px=center,
        area_fraction=float(
            np.count_nonzero(candidate.mask) / np.prod(image.shape[:2])
        ),
    )
    return candidate, geometry, assessment, float(objective)


class RightTargetObserver:
    """Right-wrist observer compatible with the existing camera manager."""

    def __init__(
        self,
        runner,
        output_dir,
        *,
        prompts,
        grasp_window_template,
        grasp_window_method,
    ):
        # Reuse only the proven camera lifecycle and freshness barrier. Target
        # selection and artifacts below are marker-independent.
        from src.run_staged_sam_pregrasp import RightLidObserver

        self._camera_observer = RightLidObserver(runner, Path(output_dir))
        self.runner = runner
        self.output_dir = self._camera_observer.output_dir
        self.prompts = tuple(dict.fromkeys(str(value) for value in prompts if value))
        if not self.prompts:
            raise ValueError("at least one target SAM prompt is required")
        self.template = grasp_window_template
        self.method = grasp_window_method
        self.previous_center = None
        self.sequence = 0
        self.last_observation_artifacts = None

    @property
    def camera(self):
        return self._camera_observer.camera

    def start(self):
        self._camera_observer.start()

    def stop(self):
        self._camera_observer.stop()

    def observe(self, *, require_target=True, require_lid=None):
        if require_lid is not None:
            require_target = bool(require_lid)
        rgb, timestamp = self._camera_observer._await_fresh_frame()
        image = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sequence = self.sequence
        self.sequence += 1
        raw_path = self.output_dir / f"{sequence:03d}_raw.png"
        request_path = self.output_dir / f"{sequence:03d}_sam_request_q90.jpg"
        if not cv2.imwrite(str(raw_path), image):
            raise RuntimeError(f"could not save right raw image {raw_path}")
        if not cv2.imwrite(
            str(request_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        ):
            raise RuntimeError(f"could not save right SAM request {request_path}")

        selected = None
        attempts = []
        selected_prompt = None
        selected_model = None
        selected_objective = None
        for prompt in self.prompts:
            result = self.runner.sam.segment(
                image,
                frame_id=self.runner.frame_id,
                timestamp=float(timestamp),
                prompt=prompt,
                confidence_threshold=0.05,
            )
            self.runner.frame_id += 1
            attempts.append((prompt, len(result.candidates)))
            candidate = choose_tool_relative_target(
                result.candidates,
                image,
                self.template,
                method=self.method,
                previous_center_px=self.previous_center,
            )
            if candidate is None:
                continue
            if (
                selected is None
                or float(candidate[3]) < float(selected_objective)
            ):
                selected = candidate
                selected_objective = float(candidate[3])
                selected_prompt = prompt
                selected_model = result.model
            if candidate[2].allowed_to_close:
                break

        overlay = image.copy()
        overlay_path = self.output_dir / f"{sequence:03d}.png"
        if selected is None:
            label = f"RIGHT target unavailable; attempts={attempts}"
            cv2.putText(
                overlay,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            if not cv2.imwrite(str(overlay_path), overlay):
                raise RuntimeError(
                    f"could not save right target overlay {overlay_path}"
                )
            self.last_observation_artifacts = {
                "schema": "sam_right_target_observation/v2",
                "sequence": sequence,
                "raw_image": str(raw_path),
                "sam_request_jpeg_q90": str(request_path),
                "overlay_image": str(overlay_path),
                "target_mask": None,
                "attempts": attempts,
                "target": None,
            }
            if require_target:
                raise RuntimeError(
                    "right SAM did not identify the generic target; "
                    f"attempts={attempts}, raw={raw_path}"
                )
            return None, None, str(overlay_path), float(timestamp)

        candidate, geometry, assessment, _ = selected
        self.previous_center = geometry.center_px.copy()
        mask = np.asarray(candidate.mask, dtype=bool)
        tint = np.full_like(overlay, (0, 190, 255))
        overlay[mask] = cv2.addWeighted(
            overlay[mask], 0.45, tint[mask], 0.55, 0
        )
        center = tuple(np.rint(geometry.center_px).astype(int))
        cv2.drawMarker(
            overlay, center, (0, 255, 0), cv2.MARKER_CROSS, 28, 2
        )
        label = (
            f"RIGHT target score={candidate.score:.3f} "
            f"tool_error={assessment.normalized_center_error:.2f}"
        )
        cv2.putText(
            overlay,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        if not cv2.imwrite(str(overlay_path), overlay):
            raise RuntimeError(
                f"could not save right target overlay {overlay_path}"
            )
        mask_path = self.output_dir / f"{sequence:03d}_target_mask.png"
        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"could not save target mask {mask_path}")
        self.last_observation_artifacts = {
            "schema": "sam_right_target_observation/v2",
            "sequence": sequence,
            "raw_image": str(raw_path),
            "sam_request_jpeg_q90": str(request_path),
            "overlay_image": str(overlay_path),
            "target_mask": str(mask_path),
            "attempts": attempts,
            "target": {
                "prompt": selected_prompt,
                "model": selected_model,
                "score": float(candidate.score),
                "box_xyxy": np.asarray(
                    candidate.box_xyxy, dtype=float
                ).tolist(),
            },
        }
        return geometry, candidate, str(overlay_path), float(timestamp)
