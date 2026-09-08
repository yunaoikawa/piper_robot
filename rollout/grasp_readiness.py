"""Physical, normalized gates that must all pass before closing a gripper."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PrecloseThresholds:
    maximum_orientation_error_deg: float
    maximum_tip_clearance_m: float
    maximum_tip_penetration_m: float
    maximum_normalized_image_gap: float

    def __post_init__(self):
        values = (
            self.maximum_orientation_error_deg,
            self.maximum_tip_clearance_m,
            self.maximum_tip_penetration_m,
            self.maximum_normalized_image_gap,
        )
        if not np.all(np.isfinite(values)) or any(value < 0 for value in values):
            raise ValueError("preclose thresholds must be finite and nonnegative")

    @classmethod
    def from_dict(cls, value: dict) -> "PrecloseThresholds":
        return cls(
            maximum_orientation_error_deg=float(
                value["maximum_orientation_error_deg"]
            ),
            maximum_tip_clearance_m=float(value["maximum_tip_clearance_m"]),
            maximum_tip_penetration_m=float(
                value["maximum_tip_penetration_m"]
            ),
            maximum_normalized_image_gap=float(
                value["maximum_normalized_image_gap"]
            ),
        )


@dataclass(frozen=True)
class PrecloseAssessment:
    allowed: bool
    reasons: tuple[str, ...]
    target_visible: bool
    target_in_window: bool
    finger_pad_at_target: bool
    tool_horizontal: bool
    tool_tip_at_support: bool
    normalized_image_gap: float | None
    orientation_error_deg: float
    tool_support_clearance_m: float

    def to_dict(self) -> dict:
        return asdict(self)


def assess_preclose(
    *,
    target_visible: bool,
    target_in_window: bool,
    normalized_image_gap: float | None,
    orientation_error_deg: float,
    tool_support_clearance_m: float,
    thresholds: PrecloseThresholds,
) -> PrecloseAssessment:
    """Conjoin independent visual, orientation and physical-contact gates."""

    gap_ready = bool(
        normalized_image_gap is not None
        and np.isfinite(normalized_image_gap)
        and normalized_image_gap <= thresholds.maximum_normalized_image_gap
    )
    orientation_ready = bool(
        np.isfinite(orientation_error_deg)
        and orientation_error_deg
        <= thresholds.maximum_orientation_error_deg
    )
    support_ready = bool(
        np.isfinite(tool_support_clearance_m)
        and -thresholds.maximum_tip_penetration_m
        <= tool_support_clearance_m
        <= thresholds.maximum_tip_clearance_m
    )
    reasons = []
    if not target_visible:
        reasons.append("target_not_visible")
    if not target_in_window:
        reasons.append("target_not_in_demonstrated_tool_window")
    if not gap_ready:
        reasons.append("finger_pad_not_at_target")
    if not orientation_ready:
        reasons.append("tool_not_at_demonstrated_orientation")
    if not support_ready:
        reasons.append("tool_tip_not_at_support_plane")
    return PrecloseAssessment(
        allowed=not reasons,
        reasons=tuple(reasons),
        target_visible=bool(target_visible),
        target_in_window=bool(target_in_window),
        finger_pad_at_target=gap_ready,
        tool_horizontal=orientation_ready,
        tool_tip_at_support=support_ready,
        normalized_image_gap=(
            None
            if normalized_image_gap is None
            else float(normalized_image_gap)
        ),
        orientation_error_deg=float(orientation_error_deg),
        tool_support_clearance_m=float(tool_support_clearance_m),
    )
