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
    image_shape_hw: tuple[int, int] | None = None
    max_cross_track_fraction: float = 0.035

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
            image_shape_hw=(
                tuple(int(v) for v in value["image_shape_hw"])
                if value.get("image_shape_hw") is not None
                else None
            ),
            max_cross_track_fraction=float(
                value.get("max_cross_track_fraction", 0.035)
            ),
        )

    def save(self, path: str | Path):
        output = {
            "version": 2,
            "feature_source": "blue_cross",
            "observer_camera": self.observer_camera,
            "contact_drop_m": self.contact_drop_m,
            "max_cross_track_px": self.max_cross_track_px,
            "max_cross_track_fraction": self.max_cross_track_fraction,
            "endpoints": {
                "left": self.left.to_dict(),
                "right": self.right.to_dict(),
            },
        }
        if self.observer_pose is not None:
            output["observer_pose_wxyz_xyz"] = self.observer_pose.tolist()
        if self.image_shape_hw is not None:
            output["image_shape_hw"] = list(self.image_shape_hw)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")

    def interpolate(
        self,
        feature_px,
        *,
        image_shape_hw: tuple[int, int] | None = None,
        reject_outside: bool = False,
    ) -> InterpolationResult:
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
        shape = image_shape_hw or self.image_shape_hw
        if shape is not None:
            height, width = (int(shape[0]), int(shape[1]))
            if height <= 0 or width <= 0:
                raise ValueError("image shape must be positive")
            scale = np.array([width, height], dtype=float)
            feature_work = feature / scale
            start = self.left.feature_px / scale
            end = self.right.feature_px / scale
            minimum_axis_sq = (10.0 / float(np.hypot(width, height))) ** 2
            cross_limit = self.max_cross_track_fraction
            cross_units = "image diagonal fraction"
        else:
            feature_work = feature
            start = self.left.feature_px
            end = self.right.feature_px
            minimum_axis_sq = 100.0
            cross_limit = self.max_cross_track_px
            cross_units = "px"
        axis = end - start
        length_sq = float(axis @ axis)
        if length_sq < minimum_axis_sq:
            raise ValueError("endpoint image features are too close")
        raw_fraction = float((feature_work - start) @ axis / length_sq)
        projection = start + raw_fraction * axis
        cross_track_work = float(np.linalg.norm(feature_work - projection))
        cross_track = (
            cross_track_work * float(np.hypot(*shape))
            if shape is not None
            else cross_track_work
        )
        if cross_track_work > cross_limit:
            raise ValueError(
                "lid feature is too far from calibrated endpoint line: "
                f"{cross_track_work:.4f} {cross_units}"
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
