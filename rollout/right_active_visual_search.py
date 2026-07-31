"""Pure policy for fast, identity-preserving right-wrist target search.

MuJoCo supplies the first collision-checked hover.  This policy is used only
when the live wrist observation does not yet match the accepted grasp window.
It deliberately separates semantic reacquisition from local image servoing:

* a wide-view observation locks the requested semantic instance;
* planar search happens only at a collision-checked hover;
* target area is a weak visibility term, never the identity or goal;
* a worse/lost observation rolls back and halves the trust-region step;
* descent and closure are separate, explicit gates.

The module contains no RPC or camera code so the same state machine can be
tested offline and embedded in autonomous runners without Codex.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping


class SearchAction(str, Enum):
    ACCEPT_OBSERVATION = "ACCEPT_OBSERVATION"
    HOVER_XY_STEP = "HOVER_XY_STEP"
    ROLLBACK_AND_RETRY = "ROLLBACK_AND_RETRY"
    LIFT_AND_WIDE_REACQUIRE = "LIFT_AND_WIDE_REACQUIRE"
    DESCEND = "DESCEND"
    CLOSE = "CLOSE"
    VERIFY_LIFT = "VERIFY_LIFT"
    SUCCESS = "SUCCESS"
    ABORT = "ABORT"


@dataclass(frozen=True)
class VisualSearchObservation:
    """Resolution-independent observation in the tool-relative image frame."""

    semantic_role: str | None
    instance_id: str | None
    target_visible: bool
    normalized_center_error: float | None
    normalized_quantile_error: float | None
    target_inside_fraction: float | None
    area_fraction: float | None
    at_safe_hover: bool
    tool_horizontal: bool = False
    tip_at_support: bool = False
    stable_nonempty_closure: bool = False
    target_follows_lift: bool = False


@dataclass(frozen=True)
class VisualSearchDecision:
    action: SearchAction
    reason: str
    maximum_step_m: float
    merit: float | None
    locked_instance_id: str | None
    attempts: int


@dataclass(frozen=True)
class SceneTargetIdentity:
    semantic_role: str
    instance_id: str
    semantic_name: str


def select_unique_scene_target(
    scene: Mapping,
    *,
    semantic_role: str,
    require_operator_confirmed: bool = True,
) -> SceneTargetIdentity:
    """Resolve one semantic instance before any wrist-area optimization."""

    if require_operator_confirmed and scene.get("operator_confirmed") is not True:
        raise ValueError("semantic scene is not operator-confirmed")
    matches = [
        item
        for item in scene.get("objects", ())
        if isinstance(item, Mapping) and item.get("role") == semantic_role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {semantic_role!r} scene object, "
            f"found {len(matches)}"
        )
    item = matches[0]
    instance_id = item.get("instance_id")
    semantic_name = item.get("semantic_name")
    if not instance_id or not semantic_name:
        raise ValueError("semantic target lacks instance_id or semantic_name")
    return SceneTargetIdentity(
        semantic_role=str(semantic_role),
        instance_id=str(instance_id),
        semantic_name=str(semantic_name),
    )


def observation_merit(
    observation: VisualSearchObservation,
    *,
    area_weight: float = 0.03,
) -> float:
    """Return a goal error where area is only a small visibility tie-breaker."""

    if not observation.target_visible:
        return math.inf
    values = (
        observation.normalized_center_error,
        observation.normalized_quantile_error,
        observation.target_inside_fraction,
        observation.area_fraction,
    )
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return math.inf
    inside = min(1.0, max(0.0, float(observation.target_inside_fraction)))
    area = min(1.0, max(0.0, float(observation.area_fraction)))
    return (
        float(observation.normalized_center_error)
        + float(observation.normalized_quantile_error)
        + (1.0 - inside)
        - float(area_weight) * area
    )


class RightActiveVisualSearch:
    """Trust-region policy for a MuJoCo-seeded right-wrist search."""

    def __init__(
        self,
        *,
        target_semantic_role: str,
        initial_maximum_step_m: float = 0.030,
        minimum_step_m: float = 0.002,
        maximum_attempts: int = 8,
        improvement_epsilon: float = 0.01,
    ):
        if not target_semantic_role:
            raise ValueError("target_semantic_role is required")
        if not 0 < minimum_step_m <= initial_maximum_step_m:
            raise ValueError("invalid visual-search trust region")
        if maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")
        self.target_semantic_role = str(target_semantic_role)
        self.initial_maximum_step_m = float(initial_maximum_step_m)
        self.minimum_step_m = float(minimum_step_m)
        self.maximum_attempts = int(maximum_attempts)
        self.improvement_epsilon = float(improvement_epsilon)
        self.locked_instance_id: str | None = None
        self.best_merit = math.inf
        self.maximum_step_m = self.initial_maximum_step_m
        self.attempts = 0
        self._step_outstanding = False

    def _decision(
        self,
        action: SearchAction,
        reason: str,
        merit: float | None,
    ) -> VisualSearchDecision:
        return VisualSearchDecision(
            action=action,
            reason=reason,
            maximum_step_m=self.maximum_step_m,
            merit=None if merit is None or not math.isfinite(merit) else merit,
            locked_instance_id=self.locked_instance_id,
            attempts=self.attempts,
        )

    def observe(
        self,
        observation: VisualSearchObservation,
        *,
        wide_view: bool = False,
        alignment_ready: bool = False,
        closure_requested: bool = False,
        closure_completed: bool = False,
        verification_lift_completed: bool = False,
    ) -> VisualSearchDecision:
        """Advance the policy after one fresh semantic visual observation."""

        if self.attempts >= self.maximum_attempts:
            return self._decision(
                SearchAction.ABORT,
                "maximum visual-search attempts reached",
                None,
            )

        identity_matches_role = (
            observation.target_visible
            and observation.semantic_role == self.target_semantic_role
            and observation.instance_id is not None
        )
        if not identity_matches_role:
            if self._step_outstanding and observation.at_safe_hover:
                self.attempts += 1
                self.maximum_step_m = max(
                    self.minimum_step_m, 0.5 * self.maximum_step_m
                )
                self._step_outstanding = False
                return self._decision(
                    SearchAction.ROLLBACK_AND_RETRY,
                    "locked target disappeared or changed after XY step",
                    None,
                )
            return self._decision(
                SearchAction.LIFT_AND_WIDE_REACQUIRE,
                "target semantic identity is not established",
                None,
            )

        if self.locked_instance_id is None:
            if not wide_view:
                return self._decision(
                    SearchAction.LIFT_AND_WIDE_REACQUIRE,
                    "instance identity must first be locked in a wide view",
                    None,
                )
            self.locked_instance_id = observation.instance_id
        elif observation.instance_id != self.locked_instance_id:
            if self._step_outstanding and observation.at_safe_hover:
                self.attempts += 1
                self.maximum_step_m = max(
                    self.minimum_step_m, 0.5 * self.maximum_step_m
                )
                self._step_outstanding = False
                return self._decision(
                    SearchAction.ROLLBACK_AND_RETRY,
                    "semantic instance changed after XY step",
                    None,
                )
            return self._decision(
                SearchAction.LIFT_AND_WIDE_REACQUIRE,
                "observation is not the locked target instance",
                None,
            )

        merit = observation_merit(observation)
        if not math.isfinite(merit):
            return self._decision(
                SearchAction.LIFT_AND_WIDE_REACQUIRE,
                "target geometry is incomplete",
                None,
            )

        # Never interpret a changed contact-height image as permission for an
        # XY correction.  A low tool first returns to the collision-checked
        # hover, even if the previous hover correction was outstanding.
        if not observation.at_safe_hover and not alignment_ready:
            self._step_outstanding = False
            return self._decision(
                SearchAction.LIFT_AND_WIDE_REACQUIRE,
                "planar image search is forbidden below safe hover",
                merit,
            )

        if self._step_outstanding:
            self.attempts += 1
            if merit >= self.best_merit - self.improvement_epsilon:
                self.maximum_step_m = max(
                    self.minimum_step_m, 0.5 * self.maximum_step_m
                )
                self._step_outstanding = False
                return self._decision(
                    SearchAction.ROLLBACK_AND_RETRY,
                    "visual merit did not improve; undo step and retry smaller",
                    merit,
                )
            self._step_outstanding = False

        self.best_merit = min(self.best_merit, merit)
        if verification_lift_completed:
            if (
                observation.stable_nonempty_closure
                and observation.target_follows_lift
            ):
                return self._decision(
                    SearchAction.SUCCESS,
                    "nonempty closure retained the locked target during lift",
                    merit,
                )
            return self._decision(
                SearchAction.ABORT,
                "locked target did not follow the verification lift",
                merit,
            )

        if closure_completed:
            if observation.stable_nonempty_closure:
                return self._decision(
                    SearchAction.VERIFY_LIFT,
                    "nonempty closure observed; lift straight up to verify",
                    merit,
                )
            return self._decision(
                SearchAction.ABORT,
                "gripper closed empty or closure was unstable",
                merit,
            )

        if closure_requested:
            preclose_gates = (
                alignment_ready
                and observation.tool_horizontal
                and observation.tip_at_support
            )
            if preclose_gates:
                return self._decision(
                    SearchAction.CLOSE,
                    "identity, geometry, orientation and support gates agree",
                    merit,
                )
            return self._decision(
                SearchAction.ABORT,
                "closure requested before all preclose gates passed",
                merit,
            )

        if alignment_ready:
            if not observation.at_safe_hover:
                return self._decision(
                    SearchAction.DESCEND,
                    "locked target is aligned; continue controlled descent",
                    merit,
                )
            return self._decision(
                SearchAction.DESCEND,
                "locked target is aligned at hover",
                merit,
            )

        self._step_outstanding = True
        return self._decision(
            SearchAction.HOVER_XY_STEP,
            "take direct tool-relative correction at safe hover",
            merit,
        )
