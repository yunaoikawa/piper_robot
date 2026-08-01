from rollout.fast_lid_grasp import ClosureEvidence, FastLidGraspMachine, GraspStage


def _at_preclose():
    machine = FastLidGraspMachine()
    machine.tap_accepted()
    machine.transit_complete()
    machine.hover_assessment(target_visible=True, aligned=True)
    machine.level_checkpoint(accepted=True)
    machine.descent_complete()
    return machine


def test_success_path_requires_one_close_then_vertical_lift():
    machine = _at_preclose()
    assert machine.preclose_assessment(
        rim_between_fingers=True, level_accepted=True
    ).name == "close_once_ramped"
    assert machine.closure_measured(ClosureEvidence(0.58)).name == "lift_vertical"
    assert machine.lift_complete(still_nonempty=True).name == "hold"
    assert machine.stage == GraspStage.SUCCESS


def test_failed_preclose_lifts_before_any_xy_correction():
    machine = _at_preclose()
    action = machine.preclose_assessment(
        rim_between_fingers=False, level_accepted=True
    )
    assert action.name == "freeze_aperture_and_lift_vertical"
    assert machine.stage == GraspStage.VERTICAL_RECOVERY
    assert "forbidden" in action.reason
    assert machine.recovery_lift_complete().name == "open_at_hover"
    assert machine.stage == GraspStage.HOLD


def test_empty_closure_never_lifts():
    machine = _at_preclose()
    machine.preclose_assessment(rim_between_fingers=True, level_accepted=True)
    action = machine.closure_measured(ClosureEvidence(0.005))
    assert action.name == "hold"
    assert machine.stage == GraspStage.HOLD


def test_level_is_checked_only_after_hover_alignment():
    machine = FastLidGraspMachine()
    machine.tap_accepted()
    machine.transit_complete()
    action = machine.hover_assessment(target_visible=True, aligned=True)
    assert action.name == "check_level_before_descend"
    assert machine.stage == GraspStage.LEVEL_CHECK
    action = machine.level_checkpoint(accepted=False)
    assert action.name == "hold"
    assert machine.stage == GraspStage.HOLD


def test_motion_during_descent_requires_vertical_recovery():
    machine = FastLidGraspMachine()
    machine.tap_accepted()
    machine.transit_complete()
    machine.hover_assessment(target_visible=True, aligned=True)
    machine.level_checkpoint(accepted=True)
    action = machine.motion_watchdog(
        triggered=True, at_least_one_camera_visible=True
    )
    assert action.name == "freeze_aperture_and_lift_vertical"
    assert machine.stage == GraspStage.VERTICAL_RECOVERY

def test_unseen_target_holds_without_search():
    machine = FastLidGraspMachine()
    machine.tap_accepted()
    machine.transit_complete()
    action = machine.hover_assessment(target_visible=False, aligned=False)
    assert action.name == "hold"
    assert machine.stage == GraspStage.HOLD
