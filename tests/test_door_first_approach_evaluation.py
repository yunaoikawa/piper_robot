"""Unit checks for the retrospective paper metric (no robot or raw data)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "docs/evaluate_door_first_approach.py"
spec = importlib.util.spec_from_file_location("door_first_approach_evaluation", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_position_and_area_are_separate():
    result = module.feature_errors([0.4, 0.5, -6.5], [0.1, 0.1, -6.0])
    assert result["uv_error"] == pytest.approx(0.5)
    assert result["absolute_log_area_error"] == pytest.approx(0.5)
    assert module.feature_errors([0.4, 0.5, -3], [0.1, 0.1, -6])["uv_error"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [[1, 2], [1, 2, np.nan], [1, np.inf, 3]])
def test_invalid_features_rejected(bad):
    with pytest.raises(ValueError):
        module.feature_errors(bad, [0.1, 0.1, -6.0])


def test_missing_and_selected_run_boundaries_are_explicit():
    rows = {row["stage"]: row for row in module.CONFIGURATIONS}
    assert set(rows) == {f"D{i}" for i in range(6)} | {"D4_pre_parser_fix"}
    for stage in ("D0", "D3", "D5"):
        assert "motion" not in rows[stage]
        assert rows[stage]["missing_reason"]
    for stage in ("D1", "D2", "D4_pre_parser_fix", "D4"):
        assert rows[stage]["boundary_evidence"]
        assert "visual_alignment" not in rows[stage]["motion"]
