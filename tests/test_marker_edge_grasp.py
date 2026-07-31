from pathlib import Path

import cv2

from rollout.marker_edge_grasp import assess_marker_alignment, learn_marker_template


ROOT = Path(__file__).resolve().parents[1]
GOAL = ROOT / "src/configs/task_goals/pasteur_lid_grip_start_right_0082_q90.jpg"
PRECLOSE = ROOT / "data/runs/pasteur/urgent_lid_grasp_20260731_verified_preclose/right.png"
WRONG_LOW = ROOT / "data/runs/pasteur/right_active_search_20260731/correct_lid_lowest_safe_observe/right_raw.png"


def test_verified_real_preclose_matches_tool_relative_teacher_without_sam():
    reference = cv2.imread(str(GOAL))
    live = cv2.imread(str(PRECLOSE))
    template = learn_marker_template(reference, marker_hint_px=(206, 300))
    result = assess_marker_alignment(live, template)
    assert result.visible
    assert result.aligned
    assert result.error_scale < template.maximum_error_scale


def test_real_low_frame_without_target_marker_cannot_close():
    reference = cv2.imread(str(GOAL))
    live = cv2.imread(str(WRONG_LOW))
    template = learn_marker_template(reference, marker_hint_px=(206, 300))
    result = assess_marker_alignment(live, template)
    assert not result.aligned
    assert result.error_scale > template.maximum_error_scale
