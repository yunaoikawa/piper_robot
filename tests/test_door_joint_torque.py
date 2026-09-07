import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("seaborn")
PATH = Path(__file__).resolve().parents[1] / "docs/plot_door_joint_torque.py"
SPEC = importlib.util.spec_from_file_location("door_torque_plot", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_four_matched_stage_samples_are_not_called_pressure_or_peaks():
    report = json.loads(MODULE.REPORT.read_text())
    assert report["is_pressure"] is False
    assert len(report["trials"]) == 7
    assert all(r["selection"] == "last_torque_warning" and r["saved_vectors"] == 1 for r in report["trials"])
    assert all(r["stage"] == "while settling after trajectory" for r in report["trials"])
    rows = {r["trial"]: r for r in report["trials"]}
    fig = MODULE.plot_comparison(report)
    try:
        assert not fig.axes[0].lines  # No synthetic time curve.
        for c, trial in zip(fig.axes[0].collections, report["plotted_trials"]):
            np.testing.assert_allclose(c.get_offsets(), list(zip(range(1, 7), rows[trial]["torque_magnitude_nm"])))
        assert len(fig.axes) == 5
    finally:
        MODULE.plt.close(fig)


@pytest.mark.parametrize("values", [[0]*5, [0]*7, [-1]*6, [float("nan")]*6])
def test_invalid_magnitudes_are_not_plotted(values):
    with pytest.raises(ValueError):
        MODULE.validate_torque(values)
