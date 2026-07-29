from rollout.autonomous_grasp_state_machine import (
    GraspGates,
    GraspState,
    decide_state,
)


def test_nominal_state_sequence_needs_no_external_reasoning():
    state = GraspState.ACQUIRE_TARGET
    gates = GraspGates(
        target_visible=True,
        coarse_aligned=True,
        tool_horizontal=True,
        target_in_window=True,
        tip_at_support=True,
        closure_captured=True,
        target_followed_lift=True,
    )
    expected = [
        GraspState.COARSE_ALIGN,
        GraspState.ORIENT_TOOL,
        GraspState.FINE_ALIGN,
        GraspState.DESCEND_PROBE,
        GraspState.CLOSE_AND_MONITOR,
        GraspState.VERIFY_LIFT,
        GraspState.SUCCEEDED,
    ]
    for next_state in expected:
        state = decide_state(state, gates).state
        assert state == next_state


def test_tip_on_target_branches_to_recovery():
    decision = decide_state(
        GraspState.DESCEND_PROBE,
        GraspGates(
            target_visible=True,
            target_in_window=True,
            tip_on_target=True,
        ),
    )
    assert decision.state == GraspState.RECOVER_TIP_CONTACT
    assert decision.action == "LIFT_AND_REALIGN"


def test_recovery_limit_fails_closed():
    decision = decide_state(
        GraspState.DESCEND_PROBE,
        GraspGates(tip_on_target=True, recoveries=4, maximum_recoveries=4),
    )
    assert decision.state == GraspState.HOLD_FAILED
