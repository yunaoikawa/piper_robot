"""Minimal, fail-closed state machine for the first successful lid grasp.

Perception implementations stay outside this module.  The machine records the
only legal order: coarse transit, hover-only planar correction, one continuous
descent, one close, and a vertical lift.  In particular, a failed low visual
check can only cause a lift back to hover; it can never trigger low XY search.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class GraspStage(str, Enum):
    WAITING_FOR_TAP = "waiting_for_tap"
    TRANSIT = "transit"
    HOVER_ALIGN = "hover_align"
    DESCEND = "descend"
    PRECLOSE = "preclose"
    CLOSE = "close"
    LIFT = "lift"
    SUCCESS = "success"
    HOLD = "hold"


@dataclass(frozen=True)
class GraspAction:
    name: str
    reason: str


@dataclass(frozen=True)
class ClosureEvidence:
    measured_open_ratio: float
    successful_reference_ratios: tuple[float, ...] = (0.5888571429, 0.5734285714)
    empty_reference_ratio: float = 0.004857142857

    @property
    def nonempty(self) -> bool:
        measured = float(self.measured_open_ratio)
        success_distance = min(
            abs(measured - value) for value in self.successful_reference_ratios
        )
        empty_distance = abs(measured - self.empty_reference_ratio)
        # Nearest-reference classification is calibration-relative and does
        # not encode a task-specific absolute gripper threshold.
        return success_distance < empty_distance

    def to_dict(self) -> dict:
        value = asdict(self)
        value["nonempty"] = self.nonempty
        return value


class FastLidGraspMachine:
    def __init__(self):
        self.stage = GraspStage.WAITING_FOR_TAP
        self.history: list[dict] = []

    def _go(self, stage: GraspStage, action: str, reason: str) -> GraspAction:
        previous = self.stage
        self.stage = stage
        self.history.append(
            {"from": previous.value, "to": stage.value, "action": action, "reason": reason}
        )
        return GraspAction(action, reason)

    def tap_accepted(self) -> GraspAction:
        if self.stage != GraspStage.WAITING_FOR_TAP:
            raise ValueError("tap is only accepted before motion")
        return self._go(GraspStage.TRANSIT, "stream_transit", "operator fixed target identity")

    def transit_complete(self) -> GraspAction:
        if self.stage != GraspStage.TRANSIT:
            raise ValueError("transit completion is out of order")
        return self._go(GraspStage.HOVER_ALIGN, "inspect_right", "coarse hover reached")

    def hover_assessment(self, *, target_visible: bool, aligned: bool) -> GraspAction:
        if self.stage != GraspStage.HOVER_ALIGN:
            raise ValueError("hover assessment is out of order")
        if not target_visible:
            return self._go(GraspStage.HOLD, "hold", "selected target is not visible")
        if not aligned:
            return GraspAction("correct_xy_at_hover", "tool-relative marker/rim error")
        return self._go(GraspStage.DESCEND, "stream_descend", "hover alignment accepted")

    def descent_complete(self) -> GraspAction:
        if self.stage != GraspStage.DESCEND:
            raise ValueError("descent completion is out of order")
        return self._go(GraspStage.PRECLOSE, "inspect_preclose", "verified low pose reached")

    def preclose_assessment(self, *, rim_between_fingers: bool) -> GraspAction:
        if self.stage != GraspStage.PRECLOSE:
            raise ValueError("preclose assessment is out of order")
        if not rim_between_fingers:
            return self._go(
                GraspStage.HOVER_ALIGN,
                "lift_to_hover",
                "preclose geometry failed; planar correction is forbidden while low",
            )
        return self._go(GraspStage.CLOSE, "close_once", "rim geometry accepted")

    def closure_measured(self, evidence: ClosureEvidence) -> GraspAction:
        if self.stage != GraspStage.CLOSE:
            raise ValueError("closure evidence is out of order")
        if not evidence.nonempty:
            return self._go(GraspStage.HOLD, "hold", "closure matches empty baseline")
        return self._go(GraspStage.LIFT, "lift_vertical", "closure matches successful grasp")

    def lift_complete(self, *, still_nonempty: bool) -> GraspAction:
        if self.stage != GraspStage.LIFT:
            raise ValueError("lift completion is out of order")
        if not still_nonempty:
            return self._go(GraspStage.HOLD, "hold", "object slipped during lift")
        return self._go(GraspStage.SUCCESS, "hold", "nonempty closure survived vertical lift")
