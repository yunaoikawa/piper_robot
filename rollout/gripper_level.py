"""Jaw-level geometry and checkpoint gates for the physical right gripper.

The production EE convention is not the semantic NYU mesh convention.  For
the physical right arm, the vector between the two open fingertips is EE
local-Z and the finger approach direction is EE local-X.  At canonical home
both are horizontal and EE local-Y points up.  Keep that bridge explicit so a
semantic local-Z "up" assertion cannot silently authorize tilted hardware.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import mink
import numpy as np
from scipy.spatial.transform import Rotation


PHYSICAL_RIGHT_TIP_BASELINE_EE = np.asarray([0.0, 0.0, 1.0])
PHYSICAL_RIGHT_APPROACH_EE = np.asarray([1.0, 0.0, 0.0])
PHYSICAL_RIGHT_UP_EE = np.asarray([0.0, 1.0, 0.0])


def _unit(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(result))
    if not np.all(np.isfinite(result)) or norm < 1e-9:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return result / norm


@dataclass(frozen=True)
class JawLevelReference:
    support_up_robot: tuple[float, float, float] = (0.0, 0.0, 1.0)
    tip_baseline_ee: tuple[float, float, float] = (0.0, 0.0, 1.0)
    approach_axis_ee: tuple[float, float, float] = (1.0, 0.0, 0.0)
    open_tip_span_m: float = 0.135
    maximum_checkpoint_tilt_deg: float = 3.0
    maximum_planned_tilt_deg: float = 2.0
    maximum_tip_height_difference_m: float = 0.0017
    source: str = "canonical_physical_right_home_plus_verified_preclose"

    def __post_init__(self):
        _unit(self.support_up_robot, "support up")
        baseline = _unit(self.tip_baseline_ee, "tip baseline")
        approach = _unit(self.approach_axis_ee, "approach axis")
        if abs(float(baseline @ approach)) > 1e-6:
            raise ValueError("tip baseline and approach axis must be orthogonal")
        if not math.isfinite(self.open_tip_span_m) or self.open_tip_span_m <= 0:
            raise ValueError("open fingertip span must be positive")
        if min(
            self.maximum_checkpoint_tilt_deg,
            self.maximum_planned_tilt_deg,
            self.maximum_tip_height_difference_m,
        ) <= 0:
            raise ValueError("jaw-level limits must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class JawLevelAssessment:
    accepted: bool
    combined_tilt_deg: float
    tip_baseline_tilt_deg: float
    approach_tilt_deg: float
    tip_height_difference_m: float
    maximum_tilt_deg: float
    maximum_tip_height_difference_m: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def assess_jaw_level(
    pose_wxyz_xyz: Sequence[float],
    reference: JawLevelReference,
    *,
    planned: bool = False,
) -> JawLevelAssessment:
    pose = np.asarray(pose_wxyz_xyz, dtype=float).reshape(7)
    if not np.all(np.isfinite(pose)):
        raise ValueError("EE pose must be finite wxyz+xyz")
    rotation = mink.SE3(pose).as_matrix()[:3, :3]
    up = _unit(reference.support_up_robot, "support up")
    baseline = rotation @ _unit(reference.tip_baseline_ee, "tip baseline")
    approach = rotation @ _unit(reference.approach_axis_ee, "approach axis")
    ee_up = rotation @ PHYSICAL_RIGHT_UP_EE
    tip_sine = float(np.clip(abs(baseline @ up), 0.0, 1.0))
    approach_sine = float(np.clip(abs(approach @ up), 0.0, 1.0))
    tip_tilt = math.degrees(math.asin(tip_sine))
    approach_tilt = math.degrees(math.asin(approach_sine))
    combined = math.degrees(
        math.acos(float(np.clip(ee_up @ up, -1.0, 1.0)))
    )
    height_difference = reference.open_tip_span_m * tip_sine
    maximum_tilt = (
        reference.maximum_planned_tilt_deg
        if planned
        else reference.maximum_checkpoint_tilt_deg
    )
    reasons = []
    if combined > maximum_tilt:
        reasons.append("jaw_plane_not_parallel_to_support")
    if height_difference > reference.maximum_tip_height_difference_m:
        reasons.append("left_right_fingertip_height_mismatch")
    return JawLevelAssessment(
        accepted=not reasons,
        combined_tilt_deg=combined,
        tip_baseline_tilt_deg=tip_tilt,
        approach_tilt_deg=approach_tilt,
        tip_height_difference_m=height_difference,
        maximum_tilt_deg=maximum_tilt,
        maximum_tip_height_difference_m=(
            reference.maximum_tip_height_difference_m
        ),
        reasons=tuple(reasons),
    )


def leveled_pose(
    pose_wxyz_xyz: Sequence[float],
    reference: JawLevelReference,
) -> np.ndarray:
    """Remove roll/pitch while preserving the current horizontal approach yaw."""

    pose = np.asarray(pose_wxyz_xyz, dtype=float).reshape(7)
    rotation = mink.SE3(pose).as_matrix()[:3, :3]
    up = _unit(reference.support_up_robot, "support up")
    approach = rotation @ _unit(reference.approach_axis_ee, "approach axis")
    approach = approach - float(approach @ up) * up
    approach = _unit(approach, "projected approach")
    # Columns are physical EE local X (approach), local Y (up), and local Z
    # (the left/right fingertip baseline).  x cross y = z.
    baseline = _unit(np.cross(approach, up), "horizontal tip baseline")
    desired = np.column_stack((approach, up, baseline))
    xyzw = Rotation.from_matrix(desired).as_quat()
    result = pose.copy()
    result[:4] = xyzw[[3, 0, 1, 2]]
    return result


class RightJawLevelCheckpoint:
    """Read one measured EE pose at a named checkpoint; never poll at 30 Hz."""

    def __init__(self, rpc, reference: JawLevelReference):
        self.rpc = rpc
        self.reference = reference
        self.records: list[dict] = []

    def require(self, checkpoint: str) -> JawLevelAssessment:
        pose = np.asarray(
            self.rpc.get_right_ee_pose().parameters(), dtype=float
        )
        assessment = assess_jaw_level(pose, self.reference)
        self.records.append(
            {"checkpoint": str(checkpoint), **assessment.to_dict()}
        )
        if not assessment.accepted:
            raise RuntimeError(
                f"right jaw level rejected at {checkpoint}: "
                f"{assessment.reasons}, tilt={assessment.combined_tilt_deg:.2f}deg, "
                f"tip_delta={assessment.tip_height_difference_m * 1000:.2f}mm"
            )
        return assessment
