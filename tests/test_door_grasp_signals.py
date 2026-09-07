"""Offline plot checks against tracked historical measurements; no robot access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("seaborn")
PATH = Path(__file__).resolve().parents[1] / "docs/plot_door_grasp_signals.py"
SPEC = importlib.util.spec_from_file_location("door_grasp_plot", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report():
    return json.loads(MODULE.REPORT.read_text())


def test_report_distinguishes_aperture_from_pressure_and_missing_data():
    data = report()
    assert data["is_pressure"] is False
    assert len(data["trials"]) == 7
    assert [t["trial"] for t in data["trials"] if t["post_pull_aperture"] is None] == ["T3", "T5"]
    assert len(data["contact_only_attempts_not_in_pull_plot"]) == 3
    for t in data["trials"]:
        assert len(t["close_samples"]) == 70
        assert len(t["post_proof_samples"]) == 12
        assert t["closed_aperture"] == pytest.approx(np.median(t["close_samples"][-10:]))
        assert t["post_proof_median"] == pytest.approx(np.median(t["post_proof_samples"]))
    assert "Open verified; grasp lost" == data["trials"][-1]["outcome_note"]


def test_summary_has_only_available_points_and_no_interpolation():
    data = report()
    fig = MODULE.plot_summary(data)
    try:
        for ax, trial in zip(fig.axes, data["trials"]):
            values = [trial["closed_aperture"], trial["post_proof_median"], trial["post_pull_aperture"]]
            expected = [[i, v] for i, v in enumerate(values) if v is not None]
            actual = np.concatenate([c.get_offsets() for c in ax.collections])
            np.testing.assert_allclose(actual, expected)
            assert len(ax.lines) == 1  # Only the reference bound, no synthetic trajectory.
    finally:
        MODULE.plt.close(fig)


def test_raw_trace_uses_indices_not_fabricated_timestamps():
    data = report()
    fig = MODULE.plot_samples(data)
    try:
        for i, trial in enumerate(data["trials"]):
            for j, key in enumerate(("close_samples", "post_proof_samples")):
                line = fig.axes[2 * i + j].lines[0]
                np.testing.assert_array_equal(line.get_xdata(), np.arange(len(trial[key])))
                np.testing.assert_array_equal(line.get_ydata(), trial[key])
    finally:
        MODULE.plt.close(fig)


@pytest.mark.parametrize("values", [[], [float("nan")], [-.1], [1.1], [[.5]]])
def test_bad_samples_are_rejected(values):
    with pytest.raises(ValueError):
        MODULE.checked_samples(values)
