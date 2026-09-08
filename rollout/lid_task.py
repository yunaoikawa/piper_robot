"""Petri-lid perception adapter for the generic visual servo controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .apriltag_retarget import TagProfile, detect_tags
from .lid_vision import VisionProfile, detect_blue_marker, inspect_lid
from .visual_servo import FineObservation, ObjectEstimate, StampedObservation


class LidTaskAdapter:
    """Use fixed workspace tags + a blue cross; no lid AprilTag is required."""

    def __init__(
        self,
        tag_profile: TagProfile,
        wrist_profile: VisionProfile,
        *,
        object_z_m: float = 0.0,
        empty_close_ratio: float = 0.01,
    ):
        self.tag_profile = tag_profile
        self.wrist_profile = wrist_profile
        self.object_z_m = float(object_z_m)
        self.empty_close_ratio = float(empty_close_ratio)
        self.last_position_m = None
        self.last_tracker = None

    @classmethod
    def load(
        cls,
        tag_profile_path: str | Path,
        wrist_profile_path: str | Path,
        **kwargs,
    ) -> "LidTaskAdapter":
        return cls(
            TagProfile.load(tag_profile_path),
            VisionProfile.load(wrist_profile_path),
            **kwargs,
        )

    def detect_object(self, observation: StampedObservation) -> ObjectEstimate | None:
        image = observation.images.get("head")
        if image is None:
            return None
        detections = detect_tags(image, self.tag_profile.family)
        fixed_ids = {
            int(tag_id)
            for tag_id in (
                self.tag_profile.fixed_plane_corners
                or self.tag_profile.fixed_robot_xy
            )
        }
        visible_fixed = [tag for tag in detections if tag.tag_id in fixed_ids]
        if len(visible_fixed) < 3:
            return None
        try:
            transform = self.tag_profile.fit_image_transform(visible_fixed)
            position, tracker = self.tag_profile.locate_lid(
                image, detections, transform
            )
        except (ValueError, RuntimeError):
            return None
        position = np.asarray(position, dtype=float)
        position[2] = self.object_z_m
        if (
            self.last_position_m is not None
            and np.linalg.norm(position[:2] - self.last_position_m[:2]) > 0.030
        ):
            return None
        self.last_position_m = position.copy()
        self.last_tracker = tracker
        confidence = min(1.0, 0.70 + 0.08 * (len(visible_fixed) - 3))
        return ObjectEstimate(
            position_m=position,
            timestamp=observation.timestamp,
            confidence=confidence,
            yaw_rad=None,
            source="head_fixed_tags+blue_cross",
            diagnostics={
                "fixed_tag_ids": [tag.tag_id for tag in visible_fixed],
                "blue_cross_px": (
                    None if tracker is None else np.asarray(tracker["center"]).tolist()
                ),
            },
        )

    def fine_observation(
        self, observation: StampedObservation
    ) -> FineObservation | None:
        image = observation.images.get("right")
        if image is None:
            return None
        result = inspect_lid(image, self.wrist_profile)
        marker = result.get("marker")
        if marker is None:
            return None
        # A verified transparent edge raises confidence, but the blue fiducial
        # remains usable during partial gripper occlusion.
        confidence = 0.95 if result["ok"] else 0.70
        return FineObservation(
            feature=np.asarray(marker, dtype=float),
            confidence=confidence,
            diagnostics={
                "edge_ok": bool(result["ok"]),
                "edge_reason": result.get("reason"),
                "edge_points": int(result.get("edge_points", 0)),
            },
        )

    def check_success(
        self, observation: StampedObservation
    ) -> tuple[bool, Mapping[str, Any]]:
        image = observation.images.get("right")
        marker_visible = False
        if image is not None:
            marker, _ = detect_blue_marker(image, self.wrist_profile)
            marker_visible = marker is not None
        blocked_close = observation.gripper_ratio > self.empty_close_ratio
        diagnostics = {
            "gripper_ratio": float(observation.gripper_ratio),
            "empty_close_ratio": self.empty_close_ratio,
            "blue_marker_visible": marker_visible,
        }
        return bool(blocked_close and marker_visible), diagnostics
