from rollout.fast_lid_grasp import ClosureEvidence, FastLidGraspMachine, GraspStage


def _at_preclose():
    machine = FastLidGraspMachine()
    machine.tap_accepted()
    machine.transit_complete()
    machine.hover_assessment(target_visible=True, aligned=True)
    machine.descent_complete()
    return machine


def test_success_path_requires_one_close_then_vertical_lift():
    machine = _at_preclose()
    assert machine.preclose_assessment(rim_between_fingers=True).name == "close_once"
    assert machine.closure_measured(ClosureEvidence(0.58)).name == "lift_vertical"
    assert machine.lift_complete(still_nonempty=True).name == "hold"
    assert machine.stage == GraspStage.SUCCESS


def test_failed_preclose_lifts_before_any_xy_correction():
    machine = _at_preclose()
    action = machine.preclose_assessment(rim_between_fingers=False)
    assert action.name == "lift_to_hover"
    assert machine.stage == GraspStage.HOVER_ALIGN
    assert "forbidden" in action.reason


def test_empty_closure_never_lifts():
    machine = _at_preclose()
    machine.preclose_assessment(rim_between_fingers=True)
    action = machine.closure_measured(ClosureEvidence(0.005))
    assert action.name == "hold"
    assert machine.stage == GraspStage.HOLD

def test_unseen_target_holds_without_search():
    machine = FastLidGraspMachine()
    machine.tap_accepted()
    machine.transit_complete()
    action = machine.hover_assessment(target_visible=False, aligned=False)
    assert action.name == "hold"
    assert machine.stage == GraspStage.HOLD
