from rollout.right_active_visual_search import (
    RightActiveVisualSearch,
    SearchAction,
    VisualSearchObservation,
    observation_merit,
    select_unique_scene_target,
)


def _observation(
    *,
    role="target_lid",
    instance="lid-1",
    visible=True,
    center=0.30,
    quantile=0.20,
    inside=0.70,
    area=0.10,
    hover=True,
    **kwargs,
):
    return VisualSearchObservation(
        semantic_role=role,
        instance_id=instance,
        target_visible=visible,
        normalized_center_error=center,
        normalized_quantile_error=quantile,
        target_inside_fraction=inside,
        area_fraction=area,
        at_safe_hover=hover,
        **kwargs,
    )


def test_large_wrongly_aligned_mask_does_not_win_on_area():
    large_bad = _observation(center=0.7, quantile=0.6, inside=0.3, area=0.6)
    small_good = _observation(center=0.05, quantile=0.05, inside=0.95, area=0.1)
    assert observation_merit(small_good) < observation_merit(large_bad)


def test_identity_must_be_locked_from_wide_view():
    policy = RightActiveVisualSearch(target_semantic_role="target_lid")
    decision = policy.observe(_observation(), wide_view=False)
    assert decision.action == SearchAction.LIFT_AND_WIDE_REACQUIRE
    decision = policy.observe(_observation(), wide_view=True)
    assert decision.action == SearchAction.HOVER_XY_STEP
    assert decision.maximum_step_m == 0.030


def test_improving_step_is_kept_and_search_continues_fast():
    policy = RightActiveVisualSearch(target_semantic_role="target_lid")
    first = policy.observe(_observation(), wide_view=True)
    assert first.action == SearchAction.HOVER_XY_STEP
    improved = policy.observe(
        _observation(center=0.15, quantile=0.10, inside=0.85)
    )
    assert improved.action == SearchAction.HOVER_XY_STEP
    assert improved.maximum_step_m == 0.030


def test_lost_target_rolls_back_and_halves_step():
    policy = RightActiveVisualSearch(target_semantic_role="target_lid")
    policy.observe(_observation(), wide_view=True)
    lost = policy.observe(
        _observation(role=None, instance=None, visible=False)
    )
    assert lost.action == SearchAction.ROLLBACK_AND_RETRY
    assert lost.maximum_step_m == 0.015


def test_changed_instance_rolls_back():
    policy = RightActiveVisualSearch(target_semantic_role="target_lid")
    policy.observe(_observation(), wide_view=True)
    changed = policy.observe(_observation(instance="dish-1"))
    assert changed.action == SearchAction.ROLLBACK_AND_RETRY


def test_planar_search_is_forbidden_below_hover():
    policy = RightActiveVisualSearch(target_semantic_role="target_lid")
    policy.observe(_observation(), wide_view=True)
    policy.observe(_observation(center=0.1, quantile=0.1, inside=0.9))
    decision = policy.observe(
        _observation(center=0.2, quantile=0.2, inside=0.8, hover=False)
    )
    assert decision.action == SearchAction.LIFT_AND_WIDE_REACQUIRE


def test_close_and_postclose_verification_are_separate_gates():
    policy = RightActiveVisualSearch(target_semantic_role="target_lid")
    aligned = _observation(
        center=0.01,
        quantile=0.01,
        inside=0.99,
        hover=False,
        tool_horizontal=True,
        tip_at_support=True,
    )
    assert (
        policy.observe(aligned, wide_view=True, alignment_ready=True).action
        == SearchAction.DESCEND
    )
    assert (
        policy.observe(
            aligned,
            alignment_ready=True,
            closure_requested=True,
        ).action
        == SearchAction.CLOSE
    )
    obstructed = _observation(
        center=0.01,
        quantile=0.01,
        inside=0.99,
        hover=False,
        tool_horizontal=True,
        tip_at_support=True,
        stable_nonempty_closure=True,
    )
    assert (
        policy.observe(
            obstructed,
            alignment_ready=True,
            closure_completed=True,
        ).action
        == SearchAction.VERIFY_LIFT
    )
    followed = _observation(
        center=0.01,
        quantile=0.01,
        inside=0.99,
        hover=False,
        tool_horizontal=True,
        tip_at_support=True,
        stable_nonempty_closure=True,
        target_follows_lift=True,
    )
    assert (
        policy.observe(
            followed,
            alignment_ready=True,
            verification_lift_completed=True,
        ).action
        == SearchAction.SUCCESS
    )


def test_wrong_semantic_role_is_never_accepted_even_if_geometry_is_perfect():
    policy = RightActiveVisualSearch(target_semantic_role="target_lid")
    decision = policy.observe(
        _observation(
            role="dish_body",
            instance="dish-1",
            center=0.0,
            quantile=0.0,
            inside=1.0,
            area=0.9,
        ),
        wide_view=True,
        alignment_ready=True,
    )
    assert decision.action == SearchAction.LIFT_AND_WIDE_REACQUIRE


def test_scene_identity_resolves_lid_not_nearby_dish():
    scene = {
        "operator_confirmed": True,
        "objects": [
            {
                "instance_id": "petri_dish-current",
                "semantic_name": "petri_dish",
                "role": "dish_body",
            },
            {
                "instance_id": "petri_lid-current",
                "semantic_name": "petri_lid",
                "role": "target_lid",
            },
        ],
    }
    identity = select_unique_scene_target(scene, semantic_role="target_lid")
    assert identity.instance_id == "petri_lid-current"
    assert identity.semantic_name == "petri_lid"


def test_scene_identity_rejects_unconfirmed_or_ambiguous_scene():
    try:
        select_unique_scene_target(
            {"operator_confirmed": False, "objects": []},
            semantic_role="target_lid",
        )
    except ValueError as error:
        assert "operator-confirmed" in str(error)
    else:
        raise AssertionError("unconfirmed scene was accepted")
