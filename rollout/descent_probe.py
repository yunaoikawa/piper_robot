"""Hardware-independent interpretation of one bounded support-plane probe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DescentProbeAssessment:
    requested_distance_m: float
    measured_progress_m: float
    progress_ratio: float
    maximum_torque_change_nm: float
    early_contact: bool


def assess_descent_probe(
    *,
    requested_distance_m: float,
    measured_delta_xyz_m,
    descent_direction_xyz,
    torque_before_nm,
    torque_after_nm,
    support_clearance_m: float,
    maximum_support_clearance_m: float,
    minimum_progress_ratio_at_contact: float,
    minimum_torque_change_nm: float,
) -> DescentProbeAssessment:
    requested = float(requested_distance_m)
    direction = np.asarray(descent_direction_xyz, dtype=float)
    measured = np.asarray(measured_delta_xyz_m, dtype=float)
    before = np.asarray(torque_before_nm, dtype=float)
    after = np.asarray(torque_after_nm, dtype=float)
    if (
        requested <= 0
        or direction.shape != (3,)
        or measured.shape != (3,)
        or before.shape != (6,)
        or after.shape != (6,)
        or not np.all(np.isfinite((direction, measured)))
        or not np.all(np.isfinite((before, after)))
    ):
        raise ValueError("invalid descent probe measurement")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-9:
        raise ValueError("descent direction has zero length")
    direction /= direction_norm
    progress = max(0.0, float(measured @ direction))
    ratio = progress / requested
    torque_change = float(np.max(np.abs(after - before)))
    early_contact = bool(
        support_clearance_m > maximum_support_clearance_m
        and ratio < float(minimum_progress_ratio_at_contact)
        and torque_change >= float(minimum_torque_change_nm)
    )
    return DescentProbeAssessment(
        requested_distance_m=requested,
        measured_progress_m=progress,
        progress_ratio=ratio,
        maximum_torque_change_nm=torque_change,
        early_contact=early_contact,
    )
