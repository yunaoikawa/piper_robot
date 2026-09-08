"""Environment-independent planning and gates for contact placement.

The module separates semantic scene understanding from deterministic motion:

* RGB-D/SAM or an operator tap estimates a goal in the robot frame.
* Measured robot state and exact CAD select the kinematic branch.
* A level free-space route stops at a hover; it never assumes contact.
* Short descent probes continue until pressure or support-plane evidence proves
  contact.  Stalled motion alone requests a rebranch instead of claiming that
  the object was placed.
* Every transition requires fresh, provenance-labelled camera frames.

Nothing in this module names a particular laboratory, object, pixel, or arm.
Site-specific calibration belongs in JSON and semantic-scene artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from rollout.gripper_level import JawLevelReference, leveled_pose


SCHEMA = "piper_robot.contact_placement/v1"

# Physical names are the public API.  The production MJCF inherited crossed
# branch names, while the semantic scene uses ordinary physical names.
PRODUCTION_BRANCH = {"right": "left_arm", "left": "right_arm"}
SEMANTIC_BRANCH = {"right": "right", "left": "left"}


def _unit(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.all(np.isfinite(vector)) or norm < 1e-9:
        raise ValueError(f"{name} must be a finite non-zero 3-vector")
    return vector / norm


def _pose(value: Sequence[float], name: str = "pose") -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(7)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain seven finite values")
    quaternion_norm = float(np.linalg.norm(result[:4]))
    if quaternion_norm < 1e-9:
        raise ValueError(f"{name} quaternion is zero")
    result = result.copy()
    result[:4] /= quaternion_norm
    if result[0] < 0.0:
        result[:4] *= -1.0
    return result


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class ArmIdentity:
    """Explicit bridge between physical hardware and both model conventions."""

    physical_arm: str
    observer_arm: str
    production_branch: str
    semantic_branch: str

    @classmethod
    def for_physical_arm(cls, physical_arm: str) -> "ArmIdentity":
        arm = str(physical_arm).lower()
        if arm not in {"left", "right"}:
            raise ValueError("physical arm must be left or right")
        observer = "left" if arm == "right" else "right"
        return cls(
            physical_arm=arm,
            observer_arm=observer,
            production_branch=PRODUCTION_BRANCH[arm],
            semantic_branch=SEMANTIC_BRANCH[arm],
        )


@dataclass(frozen=True)
class FrameEvidence:
    camera: str
    frame_id: str
    captured_at_s: float
    image_path: str | None = None
    width_px: int | None = None
    height_px: int | None = None

    def __post_init__(self):
        if not self.camera or not self.frame_id:
            raise ValueError("camera and frame_id are required")
        if not math.isfinite(self.captured_at_s):
            raise ValueError("frame timestamp must be finite")
        if (self.width_px is None) != (self.height_px is None):
            raise ValueError("frame width and height must be provided together")
        if self.width_px is not None and (
            self.width_px <= 0 or self.height_px is None or self.height_px <= 0
        ):
            raise ValueError("frame dimensions must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FrameGateResult:
    accepted: bool
    reasons: tuple[str, ...]
    maximum_age_s: float
    timestamp_skew_s: float


def validate_fresh_frames(
    frames: Iterable[FrameEvidence],
    *,
    required_cameras: Iterable[str],
    now_s: float,
    maximum_age_s: float,
    maximum_skew_s: float,
    prior_frame_ids: Mapping[str, str] | None = None,
) -> FrameGateResult:
    """Require one new, mutually synchronized frame from every named camera."""

    if maximum_age_s <= 0.0 or maximum_skew_s < 0.0:
        raise ValueError("invalid frame freshness limits")
    by_camera: dict[str, FrameEvidence] = {}
    duplicate = False
    for frame in frames:
        duplicate |= frame.camera in by_camera
        by_camera[frame.camera] = frame
    required = tuple(dict.fromkeys(str(name) for name in required_cameras))
    reasons: list[str] = []
    if duplicate:
        reasons.append("duplicate_camera_frame")
    for camera in required:
        frame = by_camera.get(camera)
        if frame is None:
            reasons.append(f"missing_frame:{camera}")
            continue
        age = float(now_s - frame.captured_at_s)
        if age < -maximum_skew_s:
            reasons.append(f"future_frame:{camera}")
        elif age > maximum_age_s:
            reasons.append(f"stale_frame:{camera}")
        if prior_frame_ids and prior_frame_ids.get(camera) == frame.frame_id:
            reasons.append(f"reused_frame:{camera}")
    timestamps = [by_camera[name].captured_at_s for name in required if name in by_camera]
    skew = float(max(timestamps) - min(timestamps)) if timestamps else float("inf")
    if len(timestamps) > 1 and skew > maximum_skew_s:
        reasons.append("camera_timestamp_skew")
    return FrameGateResult(
        accepted=not reasons,
        reasons=tuple(reasons),
        maximum_age_s=float(maximum_age_s),
        timestamp_skew_s=skew,
    )


@dataclass(frozen=True)
class GoalEstimate:
    semantic_name: str
    position_robot_m: tuple[float, float, float]
    support_normal_robot: tuple[float, float, float]
    characteristic_scale_m: float
    source: str
    scene_revision: str

    def __post_init__(self):
        position = np.asarray(self.position_robot_m, dtype=float)
        _unit(self.support_normal_robot, "support normal")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("goal position must contain three finite values")
        if not math.isfinite(self.characteristic_scale_m) or self.characteristic_scale_m <= 0:
            raise ValueError("goal characteristic scale must be positive")
        if not self.semantic_name or not self.source or not self.scene_revision:
            raise ValueError("goal semantic name, source, and scene revision are required")

    def to_dict(self) -> dict:
        return asdict(self)


def deproject_normalized_goal(
    *,
    normalized_uv: Sequence[float],
    depth_m: np.ndarray,
    intrinsics_fx_fy_cx_cy: Sequence[float],
    camera_to_robot_4x4: Sequence[Sequence[float]],
    semantic_name: str,
    support_normal_robot: Sequence[float],
    characteristic_scale_m: float,
    scene_revision: str,
    sampling_radius_fraction: float = 0.01,
) -> GoalEstimate:
    """Lift a confirmed normalized image point into the robot frame.

    The depth neighborhood scales with the image dimensions.  This intentionally
    avoids retaining the pixel coordinates or resolution of one laboratory.
    """

    uv = np.asarray(normalized_uv, dtype=float).reshape(2)
    depth = np.asarray(depth_m, dtype=float)
    intrinsics = np.asarray(intrinsics_fx_fy_cx_cy, dtype=float).reshape(4)
    transform = np.asarray(camera_to_robot_4x4, dtype=float).reshape(4, 4)
    if depth.ndim != 2 or min(depth.shape) < 2:
        raise ValueError("depth image must be a non-trivial 2-D array")
    if not np.all(np.isfinite(uv)) or np.any(uv < 0.0) or np.any(uv >= 1.0):
        raise ValueError("normalized target must lie in [0, 1)")
    if not np.all(np.isfinite(intrinsics)) or np.any(intrinsics[:2] <= 0.0):
        raise ValueError("camera intrinsics are invalid")
    if not np.all(np.isfinite(transform)) or not np.allclose(
        transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6
    ):
        raise ValueError("camera-to-robot transform is invalid")
    if not 0.0 < sampling_radius_fraction <= 0.1:
        raise ValueError("depth sampling radius fraction is invalid")
    height, width = depth.shape
    pixel = uv * np.asarray([width, height], dtype=float)
    center_x = int(np.clip(round(pixel[0]), 0, width - 1))
    center_y = int(np.clip(round(pixel[1]), 0, height - 1))
    radius = max(1, int(math.ceil(min(width, height) * sampling_radius_fraction)))
    neighborhood = depth[
        max(0, center_y - radius) : min(height, center_y + radius + 1),
        max(0, center_x - radius) : min(width, center_x + radius + 1),
    ]
    valid = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0.0)]
    minimum_samples = max(3, int(math.ceil(neighborhood.size * 0.15)))
    if valid.size < minimum_samples:
        raise ValueError("confirmed target has insufficient valid local depth")
    z = float(np.median(valid))
    fx, fy, cx, cy = intrinsics
    point_camera = np.asarray(
        [(pixel[0] - cx) * z / fx, (pixel[1] - cy) * z / fy, z, 1.0]
    )
    point_robot = transform @ point_camera
    return GoalEstimate(
        semantic_name=str(semantic_name),
        position_robot_m=tuple(float(value) for value in point_robot[:3]),
        support_normal_robot=tuple(float(value) for value in _unit(support_normal_robot, "support normal")),
        characteristic_scale_m=float(characteristic_scale_m),
        source="confirmed_normalized_rgbd_target",
        scene_revision=str(scene_revision),
    )


@dataclass(frozen=True)
class ContactPlacementConfig:
    physical_arm: str
    required_cameras: tuple[str, ...]
    maximum_frame_age_s: float = 0.75
    maximum_camera_skew_s: float = 0.15
    maximum_normalized_goal_error: float = 0.20
    free_space_step_fraction: float = 0.25
    probe_step_fraction: float = 0.04
    minimum_free_space_step_m: float = 0.005
    maximum_free_space_step_m: float = 0.030
    minimum_probe_step_m: float = 0.001
    maximum_probe_step_m: float = 0.005
    route_clearance_fraction: float = 0.75
    final_hover_fraction: float = 0.20
    maximum_support_clearance_fraction: float = 0.08
    minimum_progress_ratio: float = 0.35
    minimum_torque_change_nm: float = 0.10
    required_contact_candidates: int = 2
    maximum_probe_count: int = 20

    def __post_init__(self):
        ArmIdentity.for_physical_arm(self.physical_arm)
        if not self.required_cameras or len(set(self.required_cameras)) != len(self.required_cameras):
            raise ValueError("required cameras must be a non-empty unique list")
        positive = (
            self.maximum_frame_age_s,
            self.maximum_normalized_goal_error,
            self.free_space_step_fraction,
            self.probe_step_fraction,
            self.minimum_free_space_step_m,
            self.maximum_free_space_step_m,
            self.minimum_probe_step_m,
            self.maximum_probe_step_m,
            self.route_clearance_fraction,
            self.final_hover_fraction,
            self.maximum_support_clearance_fraction,
            self.minimum_torque_change_nm,
        )
        if not np.all(np.isfinite(positive)) or min(positive) <= 0.0:
            raise ValueError("placement configuration values must be finite and positive")
        if not 0.0 <= self.maximum_camera_skew_s <= self.maximum_frame_age_s:
            raise ValueError("camera skew limit must not exceed frame age limit")
        if not 0.0 < self.minimum_progress_ratio < 1.0:
            raise ValueError("minimum progress ratio must lie in (0, 1)")
        if self.minimum_free_space_step_m > self.maximum_free_space_step_m:
            raise ValueError("free-space step bounds are reversed")
        if self.minimum_probe_step_m > self.maximum_probe_step_m:
            raise ValueError("probe step bounds are reversed")
        if self.required_contact_candidates < 1 or self.maximum_probe_count < 1:
            raise ValueError("contact/probe counts must be positive")

    @classmethod
    def from_dict(cls, value: Mapping) -> "ContactPlacementConfig":
        return cls(
            physical_arm=str(value["physical_arm"]),
            required_cameras=tuple(str(item) for item in value["required_cameras"]),
            **{
                key: value[key]
                for key in cls.__dataclass_fields__
                if key not in {"physical_arm", "required_cameras"} and key in value
            },
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def free_space_step_m(self, characteristic_scale_m: float) -> float:
        return float(
            np.clip(
                characteristic_scale_m * self.free_space_step_fraction,
                self.minimum_free_space_step_m,
                self.maximum_free_space_step_m,
            )
        )

    def probe_step_m(self, characteristic_scale_m: float, clearance_m: float | None = None) -> float:
        step = float(
            np.clip(
                characteristic_scale_m * self.probe_step_fraction,
                self.minimum_probe_step_m,
                self.maximum_probe_step_m,
            )
        )
        if clearance_m is not None and math.isfinite(clearance_m) and clearance_m > 0.0:
            step = min(step, max(self.minimum_probe_step_m, 0.5 * clearance_m))
        return step


@dataclass(frozen=True)
class TransferPlan:
    schema: str
    physical_arm: str
    production_branch: str
    semantic_branch: str
    scene_revision: str
    goal_semantic_name: str
    characteristic_scale_m: float
    free_space_step_m: float
    route_clearance_m: float
    final_hover_clearance_m: float
    waypoints_wxyz_xyz: tuple[tuple[float, ...], ...]
    poses_wxyz_xyz: tuple[tuple[float, ...], ...]
    q_physical_rad: tuple[tuple[float, ...], ...] = ()
    collision_audit: Mapping = field(default_factory=dict)

    @property
    def motion_ready(self) -> bool:
        return bool(self.q_physical_rad and self.collision_audit.get("accepted") is True)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["motion_ready"] = self.motion_ready
        return result


def _sample_pose_segment(first: np.ndarray, second: np.ndarray, maximum_step_m: float) -> np.ndarray:
    distance = float(np.linalg.norm(second[4:] - first[4:]))
    first_rotation = Rotation.from_quat(first[[1, 2, 3, 0]])
    second_rotation = Rotation.from_quat(second[[1, 2, 3, 0]])
    angular = float((first_rotation.inv() * second_rotation).magnitude())
    count = max(
        2,
        int(math.ceil(distance / maximum_step_m)) + 1,
        int(math.ceil(angular / math.radians(3.0))) + 1,
    )
    alpha = np.linspace(0.0, 1.0, count)
    positions = first[4:] + alpha[:, None] * (second[4:] - first[4:])
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_quat(np.asarray([first, second])[:, [1, 2, 3, 0]]),
    )(alpha).as_quat()
    return np.c_[rotations[:, [3, 0, 1, 2]], positions]


def build_level_transfer_plan(
    *,
    start_pose_wxyz_xyz: Sequence[float],
    goal: GoalEstimate,
    config: ContactPlacementConfig,
    level_reference: JawLevelReference,
) -> TransferPlan:
    """Plan a task-independent level route ending above the support contact."""

    start = _pose(start_pose_wxyz_xyz, "measured start pose")
    level_start = leveled_pose(start, level_reference)
    normal = _unit(goal.support_normal_robot, "support normal")
    goal_xyz = np.asarray(goal.position_robot_m, dtype=float)
    scale = float(max(goal.characteristic_scale_m, level_reference.open_tip_span_m))
    step = config.free_space_step_m(scale)
    route_clearance = max(
        config.minimum_free_space_step_m,
        config.route_clearance_fraction * goal.characteristic_scale_m,
    )
    final_hover = max(
        config.minimum_probe_step_m * 2.0,
        config.final_hover_fraction * goal.characteristic_scale_m,
    )
    start_height = float(start[4:] @ normal)
    goal_height = float(goal_xyz @ normal)
    cruise_height = max(start_height, goal_height + final_hover) + route_clearance
    start_cruise = start[4:] + normal * max(0.0, cruise_height - start_height)
    goal_cruise = goal_xyz + normal * (cruise_height - goal_height)
    goal_hover = goal_xyz + normal * final_hover
    raw = [
        start,
        level_start,
        np.r_[level_start[:4], start_cruise],
        np.r_[level_start[:4], goal_cruise],
        np.r_[level_start[:4], goal_hover],
    ]
    waypoints: list[np.ndarray] = []
    for pose in raw:
        if not waypoints or (
            np.linalg.norm(pose[4:] - waypoints[-1][4:]) > 1e-9
            or abs(float(np.dot(pose[:4], waypoints[-1][:4]))) < 1.0 - 1e-9
        ):
            waypoints.append(pose)
    samples = [waypoints[0].reshape(1, 7)]
    for first, second in zip(waypoints, waypoints[1:]):
        samples.append(_sample_pose_segment(first, second, step)[1:])
    identity = ArmIdentity.for_physical_arm(config.physical_arm)
    return TransferPlan(
        schema=SCHEMA,
        physical_arm=identity.physical_arm,
        production_branch=identity.production_branch,
        semantic_branch=identity.semantic_branch,
        scene_revision=goal.scene_revision,
        goal_semantic_name=goal.semantic_name,
        characteristic_scale_m=goal.characteristic_scale_m,
        free_space_step_m=step,
        route_clearance_m=float(route_clearance),
        final_hover_clearance_m=float(final_hover),
        waypoints_wxyz_xyz=tuple(tuple(float(v) for v in pose) for pose in waypoints),
        poses_wxyz_xyz=tuple(tuple(float(v) for v in pose) for pose in np.vstack(samples)),
    )


class Stage(str, Enum):
    OBSERVE = "observe"
    APPROACH = "approach"
    ALIGN = "align"
    DESCEND = "descend"
    CONTACT = "contact"
    RELEASE = "release"
    VERIFY = "verify"
    RETRACT = "retract"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class Action(str, Enum):
    CAPTURE = "capture"
    EXECUTE_APPROACH = "execute_approach"
    REALIGN = "realign"
    PROBE_DESCENT = "probe_descent"
    REBRANCH = "rebranch"
    HOLD_CONTACT = "hold_contact"
    OPEN_GRIPPER = "open_gripper"
    VERIFY_RELEASE = "verify_release"
    RETRACT = "retract"
    COMPLETE = "complete"
    HOLD = "hold"


@dataclass(frozen=True)
class PipelineState:
    stage: Stage = Stage.OBSERVE
    revision: int = 0
    probe_count: int = 0
    consecutive_contact_candidates: int = 0
    last_frame_ids: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping) -> "PipelineState":
        return cls(
            stage=Stage(value.get("stage", Stage.OBSERVE.value)),
            revision=int(value.get("revision", 0)),
            probe_count=int(value.get("probe_count", 0)),
            consecutive_contact_candidates=int(
                value.get("consecutive_contact_candidates", 0)
            ),
            last_frame_ids=dict(value.get("last_frame_ids", {})),
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["stage"] = self.stage.value
        return result


@dataclass(frozen=True)
class RuntimeObservation:
    physical_arm: str
    scene_revision: str
    frames: tuple[FrameEvidence, ...]
    normalized_goal_error: float | None = None
    support_clearance_m: float | None = None
    requested_descent_m: float | None = None
    measured_descent_m: float | None = None
    maximum_torque_change_nm: float | None = None
    pressure_latched: bool = False
    object_over_support: bool | None = None
    object_on_support: bool | None = None

    @classmethod
    def from_dict(cls, value: Mapping) -> "RuntimeObservation":
        return cls(
            physical_arm=str(value["physical_arm"]),
            scene_revision=str(value["scene_revision"]),
            frames=tuple(FrameEvidence(**item) for item in value.get("frames", ())),
            normalized_goal_error=value.get("normalized_goal_error"),
            support_clearance_m=value.get("support_clearance_m"),
            requested_descent_m=value.get("requested_descent_m"),
            measured_descent_m=value.get("measured_descent_m"),
            maximum_torque_change_nm=value.get("maximum_torque_change_nm"),
            pressure_latched=bool(value.get("pressure_latched", False)),
            object_over_support=value.get("object_over_support"),
            object_on_support=value.get("object_on_support"),
        )


@dataclass(frozen=True)
class Transition:
    action: Action
    state: PipelineState
    allowed: bool
    reasons: tuple[str, ...]
    distance_m: float | None = None

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "distance_m": self.distance_m,
            "state": self.state.to_dict(),
        }


class ContactPlacementPolicy:
    """Codex-free checkpoint state machine for one contact placement."""

    def __init__(self, config: ContactPlacementConfig, goal: GoalEstimate):
        self.config = config
        self.goal = goal

    def _hold(
        self,
        state: PipelineState,
        reasons: Iterable[str],
        *,
        terminal: bool = False,
    ) -> Transition:
        held = replace(
            state,
            stage=Stage.BLOCKED if terminal else state.stage,
            revision=state.revision + 1,
        )
        return Transition(Action.HOLD, held, False, tuple(reasons))

    def advance(
        self,
        state: PipelineState,
        observation: RuntimeObservation,
        *,
        now_s: float | None = None,
    ) -> Transition:
        now = float(time.time() if now_s is None else now_s)
        if observation.physical_arm != self.config.physical_arm:
            return self._hold(state, ("physical_arm_mismatch",), terminal=True)
        if observation.scene_revision != self.goal.scene_revision:
            return self._hold(state, ("scene_revision_changed",), terminal=True)
        frame_gate = validate_fresh_frames(
            observation.frames,
            required_cameras=self.config.required_cameras,
            now_s=now,
            maximum_age_s=self.config.maximum_frame_age_s,
            maximum_skew_s=self.config.maximum_camera_skew_s,
            prior_frame_ids=state.last_frame_ids,
        )
        if not frame_gate.accepted:
            return self._hold(state, frame_gate.reasons)
        frame_ids = {frame.camera: frame.frame_id for frame in observation.frames}
        base = replace(
            state,
            revision=state.revision + 1,
            last_frame_ids=frame_ids,
        )
        if state.stage == Stage.OBSERVE:
            return Transition(
                Action.EXECUTE_APPROACH,
                replace(base, stage=Stage.APPROACH),
                True,
                ("fresh_scene_and_arm_identity_confirmed",),
            )
        if state.stage in {Stage.APPROACH, Stage.ALIGN}:
            error = observation.normalized_goal_error
            if error is None or not math.isfinite(error):
                return self._hold(base, ("goal_alignment_missing",))
            if error > self.config.maximum_normalized_goal_error:
                return Transition(
                    Action.REALIGN,
                    replace(base, stage=Stage.ALIGN),
                    True,
                    ("goal_outside_scale_normalized_tolerance",),
                )
            distance = self.config.probe_step_m(
                self.goal.characteristic_scale_m,
                observation.support_clearance_m,
            )
            return Transition(
                Action.PROBE_DESCENT,
                replace(base, stage=Stage.DESCEND),
                True,
                ("fresh_alignment_accepted",),
                distance,
            )
        if state.stage == Stage.DESCEND:
            if observation.pressure_latched:
                return Transition(
                    Action.HOLD_CONTACT,
                    replace(base, stage=Stage.CONTACT),
                    True,
                    ("pressure_guard_latched_measured_hold",),
                )
            requested = observation.requested_descent_m
            measured = observation.measured_descent_m
            torque = observation.maximum_torque_change_nm
            if requested is None or measured is None or requested <= 0.0:
                return self._hold(base, ("descent_measurement_missing",))
            if not np.all(np.isfinite([requested, measured])):
                return self._hold(base, ("descent_measurement_invalid",))
            ratio = max(0.0, float(measured)) / float(requested)
            clearance_limit = (
                self.config.maximum_support_clearance_fraction
                * self.goal.characteristic_scale_m
            )
            support_agrees = bool(
                observation.support_clearance_m is not None
                and math.isfinite(observation.support_clearance_m)
                and observation.support_clearance_m <= clearance_limit
            )
            torque_agrees = bool(
                torque is not None
                and math.isfinite(torque)
                and torque >= self.config.minimum_torque_change_nm
            )
            stalled = ratio < self.config.minimum_progress_ratio
            if stalled and not (support_agrees or torque_agrees):
                return Transition(
                    Action.REBRANCH,
                    replace(
                        base,
                        stage=Stage.ALIGN,
                        consecutive_contact_candidates=0,
                    ),
                    True,
                    ("stalled_without_contact_evidence",),
                )
            candidates = (
                state.consecutive_contact_candidates + 1
                if stalled and (support_agrees or torque_agrees)
                else 0
            )
            probes = state.probe_count + 1
            if candidates >= self.config.required_contact_candidates:
                return Transition(
                    Action.HOLD_CONTACT,
                    replace(
                        base,
                        stage=Stage.CONTACT,
                        probe_count=probes,
                        consecutive_contact_candidates=candidates,
                    ),
                    True,
                    ("repeated_stall_with_contact_evidence",),
                )
            if probes >= self.config.maximum_probe_count:
                return self._hold(
                    base, ("maximum_descent_probes_reached",), terminal=True
                )
            distance = self.config.probe_step_m(
                self.goal.characteristic_scale_m,
                observation.support_clearance_m,
            )
            return Transition(
                Action.PROBE_DESCENT,
                replace(
                    base,
                    stage=Stage.DESCEND,
                    probe_count=probes,
                    consecutive_contact_candidates=candidates,
                ),
                True,
                ("contact_not_yet_proven",),
                distance,
            )
        if state.stage == Stage.CONTACT:
            if observation.object_over_support is not True:
                return self._hold(base, ("object_over_support_not_verified",))
            return Transition(
                Action.OPEN_GRIPPER,
                replace(base, stage=Stage.RELEASE),
                True,
                ("contact_and_support_overlap_verified",),
            )
        if state.stage in {Stage.RELEASE, Stage.VERIFY}:
            if observation.object_on_support is not True:
                return Transition(
                    Action.VERIFY_RELEASE,
                    replace(base, stage=Stage.VERIFY),
                    False,
                    ("release_not_visually_verified",),
                )
            return Transition(
                Action.RETRACT,
                replace(base, stage=Stage.RETRACT),
                True,
                ("object_remains_on_support",),
            )
        if state.stage == Stage.RETRACT:
            return Transition(
                Action.COMPLETE,
                replace(base, stage=Stage.COMPLETE),
                True,
                ("vertical_retract_observed",),
            )
        if state.stage == Stage.COMPLETE:
            return Transition(Action.COMPLETE, base, True, ("already_complete",))
        return self._hold(
            base, ("pipeline_blocked_requires_new_run",), terminal=True
        )


MOBILE_HTML = """<!doctype html><html lang=ja><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1,viewport-fit=cover'>
<title>Contact placement</title><style>
body{margin:0;background:#101114;color:#f5f5f5;font-family:-apple-system,sans-serif}
main{max-width:960px;margin:auto;padding:12px}.card{background:#202228;border-radius:14px;padding:12px;margin:8px 0}
#images{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:8px}
figure{margin:0;background:#202228;padding:8px;border-radius:12px}img{width:100%;height:auto;background:#000;border-radius:8px}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px}@media(max-width:650px){main{padding:7px}}
</style><main><h2>Contact placement — latest evidence</h2><div class=card id=status>loading…</div>
<div id=images></div><div class=card><pre id=details></pre></div></main><script>
let rev=-1;async function refresh(){try{let r=await fetch('current.json?t='+Date.now(),{cache:'no-store'}),s=await r.json();
status.textContent=`${s.semantic_name} / ${s.physical_arm} arm / ${s.stage} → ${s.action}`;
details.textContent=JSON.stringify(s,null,2);if(s.revision!==rev){rev=s.revision;images.innerHTML='';
for(let [name,f] of Object.entries(s.frames)){let q=document.createElement('figure');q.innerHTML=`<b>${name}</b><img src='${f.published_path}?r=${rev}'><small>${f.frame_id}</small>`;images.appendChild(q)}}}
catch(e){status.textContent='waiting for fresh evidence: '+e}}refresh();setInterval(refresh,1000)</script></html>"""


class MobileEvidencePublisher:
    """Atomically publish fresh checkpoint images for a persistent phone URL."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.image_directory = self.directory / "images"
        self.image_directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "index.html").write_text(MOBILE_HTML, encoding="utf-8")
        current_path = self.directory / "current.json"
        current = json.loads(current_path.read_text()) if current_path.is_file() else {}
        self.revision = int(current.get("revision", 0))
        self.last_frame_ids = {
            camera: str(record["frame_id"])
            for camera, record in current.get("frames", {}).items()
            if isinstance(record, dict) and record.get("frame_id")
        }

    def publish(
        self,
        *,
        semantic_name: str,
        physical_arm: str,
        stage: Stage | str,
        action: Action | str,
        frames: Iterable[FrameEvidence],
        required_cameras: Iterable[str],
        maximum_age_s: float,
        maximum_skew_s: float,
        now_s: float | None = None,
        metrics: Mapping | None = None,
    ) -> dict:
        now = float(time.time() if now_s is None else now_s)
        values = tuple(frames)
        gate = validate_fresh_frames(
            values,
            required_cameras=required_cameras,
            now_s=now,
            maximum_age_s=maximum_age_s,
            maximum_skew_s=maximum_skew_s,
            prior_frame_ids=self.last_frame_ids,
        )
        if not gate.accepted:
            raise RuntimeError("mobile evidence rejected: " + ", ".join(gate.reasons))
        self.revision += 1
        published = {}
        for frame in values:
            if frame.image_path is None:
                raise ValueError(f"{frame.camera} frame has no image path")
            source = Path(frame.image_path).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            suffix = source.suffix.lower() or ".jpg"
            filename = f"{self.revision:04d}_{frame.camera}{suffix}"
            destination = self.image_directory / filename
            temporary = destination.with_name(f".{destination.name}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            published[frame.camera] = {
                **frame.to_dict(),
                "published_path": f"images/{filename}",
                "sha256": digest,
            }
        manifest = {
            "schema": SCHEMA,
            "revision": self.revision,
            "published_at_s": now,
            "semantic_name": str(semantic_name),
            "physical_arm": ArmIdentity.for_physical_arm(physical_arm).physical_arm,
            "stage": stage.value if isinstance(stage, Stage) else str(stage),
            "action": action.value if isinstance(action, Action) else str(action),
            "timestamp_skew_s": gate.timestamp_skew_s,
            "frames": published,
            "metrics": dict(metrics or {}),
        }
        _atomic_json(self.directory / "current.json", manifest)
        self.last_frame_ids = {frame.camera: frame.frame_id for frame in values}
        return manifest
