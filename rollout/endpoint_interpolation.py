"""Two-endpoint calibration for a one-dimensional lid placement range."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EndpointSample:
    feature_px: np.ndarray
    pregrasp_pose: np.ndarray
    feature_status: str = "confirmed"

    @classmethod
    def from_dict(cls, value):
        return cls(
            feature_px=np.asarray(value["feature_px"], dtype=float).reshape(2),
            pregrasp_pose=np.asarray(value["pregrasp_pose_wxyz_xyz"], dtype=float).reshape(7),
            feature_status=value.get("feature_status", "confirmed"),
        )

    def to_dict(self):
        value = {
            "feature_px": self.feature_px.tolist(),
            "pregrasp_pose_wxyz_xyz": self.pregrasp_pose.tolist(),
        }
        if self.feature_status != "confirmed":
            value["feature_status"] = self.feature_status
        return value


@dataclass(frozen=True)
class InterpolationResult:
    fraction: float
    unclamped_fraction: float
    cross_track_error_px: float
    target_pose: np.ndarray


@dataclass(frozen=True)
class EndpointCalibration:
    left: EndpointSample
    right: EndpointSample
    observer_pose: np.ndarray | None
    contact_drop_m: float = 0.010
    max_cross_track_px: float = 25.0
    observer_camera: str = "left"

    @classmethod
    def load(cls, path: str | Path):
        value = json.loads(Path(path).read_text())
        return cls(
            left=EndpointSample.from_dict(value["endpoints"]["left"]),
            right=EndpointSample.from_dict(value["endpoints"]["right"]),
            observer_pose=(
                np.asarray(value["observer_pose_wxyz_xyz"], dtype=float).reshape(7)
                if "observer_pose_wxyz_xyz" in value
                else None
            ),
            contact_drop_m=float(value.get("contact_drop_m", 0.010)),
            max_cross_track_px=float(value.get("max_cross_track_px", 25.0)),
            observer_camera=value.get("observer_camera", "left"),
        )

    def save(self, path: str | Path):
        output = {
            "version": 1,
            "feature_source": "blue_cross",
            "observer_camera": self.observer_camera,
            "contact_drop_m": self.contact_drop_m,
            "max_cross_track_px": self.max_cross_track_px,
            "endpoints": {
                "left": self.left.to_dict(),
                "right": self.right.to_dict(),
            },
        }
        if self.observer_pose is not None:
            output["observer_pose_wxyz_xyz"] = self.observer_pose.tolist()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")

    def interpolate(self, feature_px, *, reject_outside: bool = False) -> InterpolationResult:
        unconfirmed = [
            side
            for side, sample in (("left", self.left), ("right", self.right))
            if sample.feature_status != "confirmed"
        ]
        if unconfirmed:
            raise ValueError(
                "endpoint image feature needs recalibration: " + ", ".join(unconfirmed)
            )
        feature = np.asarray(feature_px, dtype=float).reshape(2)
        start = self.left.feature_px
        axis = self.right.feature_px - start
        length_sq = float(axis @ axis)
        if length_sq < 100.0:
            raise ValueError("endpoint image features are too close")
        raw_fraction = float((feature - start) @ axis / length_sq)
        projection = start + raw_fraction * axis
        cross_track = float(np.linalg.norm(feature - projection))
        if cross_track > self.max_cross_track_px:
            raise ValueError(
                f"lid feature is {cross_track:.1f}px away from calibrated endpoint line"
            )
        if reject_outside and not 0.0 <= raw_fraction <= 1.0:
            raise ValueError(f"lid feature is outside calibrated range: t={raw_fraction:.3f}")
        fraction = float(np.clip(raw_fraction, 0.0, 1.0))
        # Orientation and pregrasp Z are intentionally fixed to the left
        # endpoint. Only robot-frame XY is interpolated.
        target = self.left.pregrasp_pose.copy()
        target[4:6] = (
            self.left.pregrasp_pose[4:6]
            + fraction
            * (self.right.pregrasp_pose[4:6] - self.left.pregrasp_pose[4:6])
        )
        return InterpolationResult(
            fraction=fraction,
            unclamped_fraction=raw_fraction,
            cross_track_error_px=cross_track,
            target_pose=target,
        )
