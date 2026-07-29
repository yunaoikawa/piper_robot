"""Temporal contact and grasp checks that cannot be inferred from one frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DescentProbeAssessment:
    contact_candidate: bool
    progress_ratio: float
    torque_change_nm: float
    image_gap_closed: bool


@dataclass(frozen=True)
class ClosureAssessment:
    captured: bool
    final_open_ratio: float
    stable_range: float
    still_closing: bool


def assess_descent_probe(
    requested_z_m: float,
    actual_z_m: float,
    torque_before_nm,
    torque_peak_nm,
    *,
    image_gap_closed: bool,
    maximum_progress_ratio: float = 0.30,
    minimum_torque_change_nm: float = 0.10,
) -> DescentProbeAssessment:
    """Require robot resistance *and* visual fingertip contact.

    Low gain and static friction can also cause poor Cartesian progress, while
    the wrist body may contact before a fingertip.  Therefore robot feedback
    alone is deliberately insufficient.
    """

    requested = float(requested_z_m)
    if requested >= 0:
        raise ValueError("a descent probe must request negative z")
    progress = float(actual_z_m) / requested
    before = np.asarray(torque_before_nm, dtype=float)
    peak = np.asarray(torque_peak_nm, dtype=float)
    if before.shape != peak.shape or not np.all(np.isfinite(before)) or not np.all(
        np.isfinite(peak)
    ):
        raise ValueError("torque samples must be finite and have equal shape")
    torque_change = float(np.max(np.abs(peak - before)))
    candidate = (
        progress <= maximum_progress_ratio
        and torque_change >= minimum_torque_change_nm
        and bool(image_gap_closed)
    )
    return DescentProbeAssessment(
        contact_candidate=bool(candidate),
        progress_ratio=progress,
        torque_change_nm=torque_change,
        image_gap_closed=bool(image_gap_closed),
    )


def assess_stable_closure(
    samples,
    *,
    empty_close_ratio: float = 0.02,
    minimum_capture_ratio: float = 0.03,
    stable_window_samples: int = 5,
    maximum_stable_range: float = 0.01,
) -> ClosureAssessment:
    """Reject a transient non-zero aperture while the gripper is still moving."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or len(values) < stable_window_samples:
        raise ValueError("not enough gripper samples")
    tail = values[-stable_window_samples:]
    stable_range = float(np.ptp(tail))
    final = float(tail[-1])
    still_closing = bool(
        stable_range > maximum_stable_range
        or np.mean(np.diff(tail)) < -maximum_stable_range / stable_window_samples
    )
    captured = (
        not still_closing
        and final > max(float(empty_close_ratio), float(minimum_capture_ratio))
    )
    return ClosureAssessment(
        captured=bool(captured),
        final_open_ratio=final,
        stable_range=stable_range,
        still_closing=still_closing,
    )
