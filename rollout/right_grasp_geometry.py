"""Right wrist-image geometry for a lid grasp.

The right camera is rigidly attached to the gripper.  Therefore a successful
demo is used as a *gripper-relative* reference: the lid may start anywhere on
the bench, but it must eventually occupy the same jaw corridor and have a
similar apparent scale before the gripper may close.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RightGraspGeometry:
    area_px: int
    center_px: tuple[float, float]
    x05_px: float
    x95_px: float
    x99_px: float
    y05_px: float
    y95_px: float


@dataclass(frozen=True)
class RightGraspAssessment:
    allowed_to_close: bool
    area_ratio: float
    jaw_corridor_ratio: float
    insertion_error_px: float
    vertical_error_px: float
    scale_ready: bool
    corridor_ready: bool
    insertion_ready: bool
    vertical_ready: bool


def mask_geometry(mask: np.ndarray) -> RightGraspGeometry:
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise ValueError("right lid mask is too small")
    return RightGraspGeometry(
        area_px=int(len(xs)),
        center_px=(float(np.mean(xs)), float(np.mean(ys))),
        x05_px=float(np.percentile(xs, 5)),
        x95_px=float(np.percentile(xs, 95)),
        x99_px=float(np.percentile(xs, 99)),
        y05_px=float(np.percentile(ys, 5)),
        y95_px=float(np.percentile(ys, 95)),
    )


def assess_right_grasp(
    current_mask: np.ndarray,
    reference_mask: np.ndarray,
    *,
    minimum_area_ratio: float = 0.72,
    insertion_tolerance_px: float = 14.0,
    vertical_tolerance_px: float = 20.0,
    minimum_jaw_corridor_ratio: float = 0.70,
    jaw_corridor_normalized_xyxy: tuple[float, float, float, float] = (
        0.3125,
        0.625,
        0.5,
        0.792,
    ),
) -> RightGraspAssessment:
    """Compare a live lid mask with a successful, wrist-fixed reference.

    ``x99`` represents how deeply the visible lid reaches into the jaws.
    The fixed corridor is gripper-relative because the wrist camera is rigidly
    mounted.  Total visible area is diagnostic only: jaw occlusion can reduce
    it during a valid approach.
    """

    current = mask_geometry(current_mask)
    reference = mask_geometry(reference_mask)
    area_ratio = current.area_px / reference.area_px
    insertion_error = reference.x99_px - current.x99_px
    vertical_error = current.center_px[1] - reference.center_px[1]
    scale_ready = area_ratio >= minimum_area_ratio
    height, width = np.asarray(current_mask).shape[:2]
    x1, y1, x2, y2 = jaw_corridor_normalized_xyxy
    x1, x2 = int(round(x1 * width)), int(round(x2 * width))
    y1, y2 = int(round(y1 * height)), int(round(y2 * height))
    live_corridor = int(np.count_nonzero(current_mask[y1:y2, x1:x2]))
    demo_corridor = int(np.count_nonzero(reference_mask[y1:y2, x1:x2]))
    jaw_corridor_ratio = live_corridor / max(demo_corridor, 1)
    corridor_ready = jaw_corridor_ratio >= minimum_jaw_corridor_ratio
    insertion_ready = insertion_error <= insertion_tolerance_px
    vertical_ready = abs(vertical_error) <= vertical_tolerance_px
    return RightGraspAssessment(
        allowed_to_close=corridor_ready and insertion_ready and vertical_ready,
        area_ratio=float(area_ratio),
        jaw_corridor_ratio=float(jaw_corridor_ratio),
        insertion_error_px=float(insertion_error),
        vertical_error_px=float(vertical_error),
        scale_ready=bool(scale_ready),
        corridor_ready=bool(corridor_ready),
        insertion_ready=bool(insertion_ready),
        vertical_ready=bool(vertical_ready),
    )


def render_right_grasp_assessment(
    image_bgr: np.ndarray,
    current_mask: np.ndarray,
    reference_mask: np.ndarray,
    assessment: RightGraspAssessment,
) -> np.ndarray:
    """Render the live mask and the successful-demo jaw corridor."""

    out = np.asarray(image_bgr).copy()
    current_mask = np.asarray(current_mask, dtype=bool)
    reference = mask_geometry(reference_mask)
    tint = np.zeros_like(out)
    tint[:] = (0, 190, 255)
    out[current_mask] = cv2.addWeighted(
        out[current_mask], 0.45, tint[current_mask], 0.55, 0
    )
    cv2.line(
        out,
        (int(round(reference.x95_px)), int(round(reference.y05_px))),
        (int(round(reference.x95_px)), int(round(reference.y95_px))),
        (255, 0, 255),
        3,
    )
    color = (0, 220, 0) if assessment.allowed_to_close else (0, 0, 255)
    lines = (
        f"close={assessment.allowed_to_close}",
        f"area/demo={assessment.area_ratio:.2f}",
        f"jaw corridor/demo={assessment.jaw_corridor_ratio:.2f}",
        f"insertion_err={assessment.insertion_error_px:+.1f}px",
        f"vertical_err={assessment.vertical_error_px:+.1f}px",
    )
    for index, text in enumerate(lines):
        y = 26 + index * 24
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3
        )
        cv2.putText(
            out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1
        )
    return out
