from rollout.grasp_contact_verification import (
    assess_descent_probe,
    assess_stable_closure,
)


def test_descent_stall_is_not_contact_while_visual_gap_remains():
    result = assess_descent_probe(
        -0.0025,
        0.0,
        [0.0, -0.7],
        [0.25, -0.6],
        image_gap_closed=False,
    )
    assert not result.contact_candidate


def test_transient_nonzero_aperture_is_not_a_grasp():
    result = assess_stable_closure([0.94, 0.87, 0.81, 0.75, 0.68])
    assert not result.captured
    assert result.still_closing


def test_stable_obstructed_aperture_is_a_grasp_candidate():
    result = assess_stable_closure([0.31, 0.305, 0.306, 0.304, 0.305])
    assert result.captured
