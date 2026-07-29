"""Codex-independent state transitions for a generic target grasp."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GraspState(str, Enum):
    ACQUIRE_TARGET = "ACQUIRE_TARGET"
    COARSE_ALIGN = "COARSE_ALIGN"
    ORIENT_TOOL = "ORIENT_TOOL"
    FINE_ALIGN = "FINE_ALIGN"
    DESCEND_PROBE = "DESCEND_PROBE"
    RECOVER_TIP_CONTACT = "RECOVER_TIP_CONTACT"
    CLOSE_AND_MONITOR = "CLOSE_AND_MONITOR"
    VERIFY_LIFT = "VERIFY_LIFT"
    SUCCEEDED = "SUCCEEDED"
    HOLD_FAILED = "HOLD_FAILED"


@dataclass(frozen=True)
class GraspGates:
    target_visible: bool = False
    coarse_aligned: bool = False
    tool_horizontal: bool = False
    target_in_window: bool = False
    tip_at_support: bool = False
    tip_on_target: bool = False
    closure_captured: bool = False
    target_followed_lift: bool = False
    recoveries: int = 0
    maximum_recoveries: int = 4


@dataclass(frozen=True)
class StateDecision:
    state: GraspState
    action: str
    reason: str


def decide_state(state: GraspState, gates: GraspGates) -> StateDecision:
    if state == GraspState.ACQUIRE_TARGET:
        if not gates.target_visible:
            return StateDecision(state, "OBSERVE", "target_not_visible")
        return StateDecision(GraspState.COARSE_ALIGN, "PLAN_COARSE", "target_acquired")
    if state == GraspState.COARSE_ALIGN:
        if not gates.target_visible:
            return StateDecision(GraspState.ACQUIRE_TARGET, "HOLD", "target_lost")
        if not gates.coarse_aligned:
            return StateDecision(state, "MOVE_COARSE_CHUNK", "coarse_error")
        return StateDecision(GraspState.ORIENT_TOOL, "PLAN_ORIENTATION", "coarse_aligned")
    if state == GraspState.ORIENT_TOOL:
        if not gates.tool_horizontal:
            return StateDecision(state, "MOVE_ORIENTATION_CHUNK", "tool_not_horizontal")
        return StateDecision(GraspState.FINE_ALIGN, "OBSERVE_RIGHT", "tool_horizontal")
    if state == GraspState.FINE_ALIGN:
        if not gates.target_visible:
            return StateDecision(GraspState.ACQUIRE_TARGET, "HOLD", "target_lost")
        if not gates.target_in_window:
            return StateDecision(state, "MOVE_FINE_HORIZONTAL", "target_outside_window")
        return StateDecision(GraspState.DESCEND_PROBE, "DESCEND_2MM", "window_ready")
    if state == GraspState.DESCEND_PROBE:
        if gates.tip_on_target:
            if gates.recoveries >= gates.maximum_recoveries:
                return StateDecision(
                    GraspState.HOLD_FAILED, "OPEN_AND_HOLD", "recovery_limit"
                )
            return StateDecision(
                GraspState.RECOVER_TIP_CONTACT,
                "LIFT_AND_REALIGN",
                "tip_contacted_target",
            )
        if not gates.tip_at_support:
            return StateDecision(state, "DESCEND_2MM", "support_not_reached")
        if not gates.target_in_window:
            return StateDecision(GraspState.FINE_ALIGN, "LIFT_AND_REALIGN", "window_shift")
        return StateDecision(
            GraspState.CLOSE_AND_MONITOR, "CLOSE_GRIPPER", "all_preclose_gates"
        )
    if state == GraspState.RECOVER_TIP_CONTACT:
        return StateDecision(GraspState.FINE_ALIGN, "OBSERVE_RIGHT", "contact_released")
    if state == GraspState.CLOSE_AND_MONITOR:
        if not gates.closure_captured:
            return StateDecision(GraspState.HOLD_FAILED, "OPEN_AND_HOLD", "empty_close")
        return StateDecision(GraspState.VERIFY_LIFT, "LIFT_2MM", "closure_stable")
    if state == GraspState.VERIFY_LIFT:
        if not gates.target_followed_lift:
            return StateDecision(GraspState.HOLD_FAILED, "OPEN_AND_HOLD", "target_did_not_follow")
        return StateDecision(GraspState.SUCCEEDED, "HOLD", "grasp_verified")
    return StateDecision(state, "HOLD", "terminal")
