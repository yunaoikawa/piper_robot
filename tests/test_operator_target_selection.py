import json

import pytest

from rollout.operator_target_selection import (
    validate_target_selection,
    write_target_selection,
)


def test_confirmed_selection_is_normalized_and_persisted(tmp_path):
    selection = validate_target_selection(
        {"u": 203, "v": 320, "confirmed": True},
        semantic_name="microscope_stage_central_aperture",
        image_path=tmp_path / "right.jpg",
        image_width_px=640,
        image_height_px=480,
        timestamp_s=12.5,
    )
    assert selection.pixel_uv == (203.0, 320.0)
    assert selection.normalized_uv == (203 / 640, 320 / 480)
    output = tmp_path / "selection.json"
    write_target_selection(output, selection)
    assert json.loads(output.read_text())["confirmed"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"u": 203, "v": 320, "confirmed": False},
        {"u": -1, "v": 320, "confirmed": True},
        {"u": 640, "v": 320, "confirmed": True},
        {"u": "bad", "v": 320, "confirmed": True},
    ],
)
def test_invalid_or_unconfirmed_selection_is_rejected(tmp_path, payload):
    with pytest.raises((ValueError, TypeError)):
        validate_target_selection(
            payload,
            semantic_name="target",
            image_path=tmp_path / "right.jpg",
            image_width_px=640,
            image_height_px=480,
        )
