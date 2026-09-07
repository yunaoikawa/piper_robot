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


def test_demo_comparison_includes_early_failures_and_later_nonmonotonic_change():
    data = report()["demo_comparison"]
    assert [r["trial"] for r in data["trials"]] == ["E1", "E2", "E3"] + [f"T{i}" for i in range(1, 8)]
    assert len(data["demo_gripper_audit"]) == 12
    assert all(r["unique_gripper_values"] == [0., 1.] for r in data["demo_gripper_audit"])
    for row in data["trials"]:
        position, angle = MODULE.pose_error(row["contact_pose_wxyz_xyz"], data["reference_contact_pose_wxyz_xyz"])
        assert position == pytest.approx(row["position_difference_mm"])
        assert angle == pytest.approx(row["orientation_difference_deg"])
    assert data["trials"][3]["position_difference_mm"] < 2
    assert data["trials"][-1]["position_difference_mm"] > 19


def test_quaternion_sign_is_not_an_orientation_error():
    ref = [1, 0, 0, 0, 0, 0, 0]
    assert MODULE.pose_error([-1, 0, 0, 0, .001, 0, 0], ref) == pytest.approx((1, 0))


def test_demo_plot_keeps_all_actual_coordinates():
    data = report()
    fig = MODULE.plot_demo_comparison(data)
    try:
        for ax, key in zip(fig.axes, ["position_difference_mm", "orientation_difference_deg"]):
            expected = [[i, r[key]] for i, r in enumerate(data["demo_comparison"]["trials"])]
            np.testing.assert_allclose(ax.collections[0].get_offsets(), expected)
            assert len(ax.lines) == 1  # Zero reference only.
    finally:
        MODULE.plt.close(fig)


def test_three_patterns_use_measured_cases_and_separate_endpoint_evidence():
    data = report()
    a, b, c = data["three_patterns"]["cases"]
    assert [r["trial"] for r in (a, b, c)] == ["E2", "T1", "T7"]
    assert a["proof"] is None and a["post_pull"] is None
    assert a["samples"][-1] < .02
    assert b["proof"] > .02 and c["proof"] > .02
    assert b["post_pull"] == c["post_pull"]
    assert b["door_state"] == "closed" and c["door_state"] == "open"
    assert c["rotation_clockwise_deg"] == 90
    fig = MODULE.plot_three_patterns(data)
    try:
        for i, case in enumerate((a, b, c)):
            np.testing.assert_array_equal(fig.axes[i*3].lines[0].get_ydata(), case["samples"])
            assert len(fig.axes[i*3+2].images) == 1
            if i:
                assert len(fig.axes[i*3+1].collections) == 2
                assert len(fig.axes[i*3+1].lines) == 1  # Bound only, no fictitious slip curve.
    finally:
        MODULE.plt.close(fig)


def test_three_pattern_image_hashes_match_tracked_thumbnails():
    import hashlib
    for case in report()["three_patterns"]["cases"]:
        actual = hashlib.sha256((MODULE.ROOT / case["display_image"]).read_bytes()).hexdigest()
        assert actual == case["display_image_sha256"]


def test_task_success_is_distinct_from_continuous_grasp():
    data = report()
    success = data["successful_opening"]
    assert success["task_success"] is True
    assert success["continuous_grasp_success"] is False
    assert (success["initial_state"], success["final_state"]) == ("closed", "open")
    fig = MODULE.plot_successful_opening(data)
    try:
        assert len(fig.axes) == 4
        assert len(fig.axes[2].images) == len(fig.axes[3].images) == 1
        assert len(fig.axes[1].lines) == 1  # Reference only; no fabricated retained-grasp trace.
    finally:
        MODULE.plt.close(fig)


def test_overlay_has_two_failures_two_successes_and_measured_open_starts():
    data = report()
    cases = data["outcome_overlay"]["cases"]
    assert [c["trial"] for c in cases] == ["T1", "T2", "T6", "T7"]
    assert [c["door_state"] for c in cases] == ["closed", "closed", "open", "open"]
    assert all(c["aperture"][0] == 1.0 for c in cases)
    assert len(set(c["aperture"][-1] for c in cases)) == 1
    fig = MODULE.plot_outcome_overlay(data)
    try:
        ax = fig.axes[0]
        for collection, case in zip(ax.collections, cases):
            np.testing.assert_allclose(collection.get_offsets(), list(zip([0, 1, 2, 4], case["aperture"])))
        for line in ax.lines[:4]:
            np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2])
        assert len(fig.axes) == 5
    finally:
        MODULE.plt.close(fig)


def test_recovered_stopping_values_are_not_treated_as_full_traces():
    audit = json.loads((MODULE.ASSETS / "door_pull_trace_audit.json").read_text())
    assert audit["runtime_modified"] is False
    recovered = audit["recovered_stop_readings"]
    assert len(recovered) == 5
    for label in ["retry3", "retry5"]:
        sample = next(r for r in recovered if label in r["source"])
        assert sample["aperture_rounded_4dp"] == .0034
    assert audit["private_event_log_audit"]["tool_output_records_scanned"] == 642
