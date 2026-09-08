from types import SimpleNamespace

from rollout.grasp_window import calibrate_grasp_window
from rollout.right_target_observer import choose_tool_relative_target
from tests.test_grasp_window import _scene


def test_tool_relative_candidate_beats_higher_raw_sam_score():
    image, reference_mask = _scene()
    template, _ = calibrate_grasp_window(image, reference_mask)
    _, shifted_mask = _scene(target_shift=(145, 0))
    correct = SimpleNamespace(
        mask=reference_mask,
        score=0.30,
        box_xyxy=[0, 0, 1, 1],
    )
    wrong = SimpleNamespace(
        mask=shifted_mask,
        score=0.99,
        box_xyxy=[0, 0, 1, 1],
    )
    selected = choose_tool_relative_target(
        [wrong, correct],
        image,
        template,
        method="HYBRID",
    )
    assert selected is not None
    assert selected[0] is correct


def test_candidate_selection_does_not_require_blue_marker():
    image, target_mask = _scene()
    template, _ = calibrate_grasp_window(image, target_mask)
    candidate = SimpleNamespace(
        mask=target_mask,
        score=0.5,
        box_xyxy=[0, 0, 1, 1],
    )
    selected = choose_tool_relative_target(
        [candidate],
        image,
        template,
        method="HYBRID",
    )
    assert selected is not None
    assert selected[2].allowed_to_close
