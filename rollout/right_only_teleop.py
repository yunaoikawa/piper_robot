"""Quest teleoperation controller with no left-arm command path.

This module is intentionally narrower than the bimanual collector.  It never
initializes the robot and only calls the two right-arm RPC methods needed for
relative teleoperation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import mink
import numpy as np

from rollout.autonomous_mpc import AbsoluteCartesianTargetSender
from rollout.safety import SafetyLayer


class RightTeleopEvent(str, Enum):
    NONE = "none"
    ENGAGED = "engaged"
    DISENGAGED = "disengaged"
    STALE = "stale"
    REJECTED = "rejected"
    COMMANDED = "commanded"


@dataclass
class RightOnlyTeleop:
    """Convert Quest right-controller motion into right EE targets."""

    rpc: object
    timeout_s: float = 0.75
    safety: SafetyLayer | None = None

    def __post_init__(self):
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.safety is None:
            self.safety = SafetyLayer()
        self.engaged = False
        self._engage_armed = False
        self._controller_anchor = None
        self._ee_anchor = None
        self._target_sender = AbsoluteCartesianTargetSender(self.rpc)
        self._H = mink.SE3.from_rotation(
            mink.SO3.from_matrix(
                np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)
            )
        )

    def disengage(self, event=RightTeleopEvent.DISENGAGED):
        was_engaged = self.engaged
        self.engaged = False
        self._controller_anchor = None
        self._ee_anchor = None
        self.safety.reset("right")
        return event if was_engaged else RightTeleopEvent.NONE

    def step(self, controller_state, now: float | None = None) -> RightTeleopEvent:
        """Process one controller state and issue at most one right-arm target."""
        now = time.time() if now is None else float(now)
        if now - float(controller_state.created_timestamp) > self.timeout_s:
            # Reconnection must include an observed A-release before a new
            # engagement.  A stale held button can therefore never move an arm.
            self._engage_armed = False
            return self.disengage(RightTeleopEvent.STALE)

        # Stop wins if A and B are somehow reported together.
        if controller_state.right_b:
            event = self.disengage()
            if not controller_state.right_a:
                self._engage_armed = True
            return event

        if not controller_state.right_a:
            self._engage_armed = True

        just_engaged = False
        if controller_state.right_a and not self.engaged and self._engage_armed:
            self._engage_armed = False
            self._controller_anchor = controller_state.right_SE3
            self._ee_anchor = self.rpc.get_right_ee_pose()
            self.safety.reset("right")
            self.engaged = True
            just_engaged = True

        if not self.engaged:
            return RightTeleopEvent.NONE

        delta = self._controller_anchor.inverse().multiply(controller_state.right_SE3)
        robot_delta = self._H.inverse() @ delta @ self._H
        position = self._ee_anchor.translation() + robot_delta.translation()
        rotation = robot_delta.rotation() @ self._ee_anchor.rotation()
        position = self.safety.check("right", position)
        if position is None:
            return RightTeleopEvent.REJECTED

        gripper = 1.0 if controller_state.right_index_trigger < 0.5 else 0.0
        self._target_sender.send(
            mink.SE3(np.concatenate([rotation.wxyz, position])),
            gripper_target=gripper,
        )
        return RightTeleopEvent.ENGAGED if just_engaged else RightTeleopEvent.COMMANDED
