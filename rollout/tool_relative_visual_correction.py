"""Convert a wrist-image target error into a metric gripper-plane correction.

The grasp-window frame follows the visible finger pad.  For the reviewed
right-wrist mounting, increasing image ``u`` points opposite semantic-model
local X and increasing image ``v`` points along model-local Y.  This sign is
validated by the observed dish-to-lid vector and their independent scene
coordinates.  The target's known physical
diameter supplies scale, so the result is independent of image resolution and
contains no absolute pixel target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rollout.grasp_window import (
    GraspWindowTemplate,
    ToolImageFrame,
    target_points_in_tool_frame,
)


@dataclass(frozen=True)
class VisualPlaneCorrection:
    raw_error_uv: tuple[float, float]
    metres_per_tool_unit: float
    metric_scale_source: str
    raw_model_local_xy_m: tuple[float, float]
    bounded_model_local_xy_m: tuple[float, float]
    raw_norm_m: float
    bounded_norm_m: float

    def to_dict(self) -> dict:
        return {
            "raw_error_uv": list(self.raw_error_uv),
            "metres_per_tool_unit": self.metres_per_tool_unit,
            "metric_scale_source": self.metric_scale_source,
            "raw_model_local_xy_m": list(self.raw_model_local_xy_m),
            "bounded_model_local_xy_m": list(
                self.bounded_model_local_xy_m
            ),
            "raw_norm_m": self.raw_norm_m,
            "bounded_norm_m": self.bounded_norm_m,
        }


def estimate_model_plane_correction(
    target_mask,
    tool_frame: ToolImageFrame,
    template: GraspWindowTemplate,
    *,
    target_diameter_m: float,
    maximum_step_m: float,
) -> VisualPlaneCorrection:
    """Estimate one direct correction toward the accepted visual goal.

    The accepted goal's 5--95 percentile span is the stable image-space
    observation of the known round target diameter.  Using it for metric
    scale avoids overestimating corrections when the live transparent target
    is cropped by the image border or occluded by a finger.  The live mask is
    used only for the current tool-relative centre error.
    """

    diameter = float(target_diameter_m)
    maximum = float(maximum_step_m)
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError("target_diameter_m must be positive")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum_step_m must be positive")
    points = target_points_in_tool_frame(target_mask, tool_frame)
    reference_quantiles = np.asarray(
        template.reference_quantiles_uv, dtype=float
    )
    if reference_quantiles.shape != (4,) or not np.all(
        np.isfinite(reference_quantiles)
    ):
        raise ValueError("template reference quantiles are invalid")
    extent = float(
        max(
            reference_quantiles[1] - reference_quantiles[0],
            reference_quantiles[3] - reference_quantiles[2],
        )
    )
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("target tool-frame extent is invalid")
    metres_per_tool = diameter / extent
    center = np.mean(points, axis=0)
    reference = np.asarray(template.reference_center_uv, dtype=float)
    error_uv = center - reference

    # Image-tool +u is semantic-model -X and image-tool +v is model +Y.
    # Moving the camera/tool by [-du, +dv] reduces the static target's
    # relative error.
    raw_local = metres_per_tool * np.asarray(
        [-error_uv[0], error_uv[1]],
        dtype=float,
    )
    raw_norm = float(np.linalg.norm(raw_local))
    bounded = raw_local.copy()
    if raw_norm > maximum:
        bounded *= maximum / raw_norm
    return VisualPlaneCorrection(
        raw_error_uv=(float(error_uv[0]), float(error_uv[1])),
        metres_per_tool_unit=float(metres_per_tool),
        metric_scale_source="accepted_goal_reference_quantiles",
        raw_model_local_xy_m=(
            float(raw_local[0]),
            float(raw_local[1]),
        ),
        bounded_model_local_xy_m=(
            float(bounded[0]),
            float(bounded[1]),
        ),
        raw_norm_m=raw_norm,
        bounded_norm_m=float(np.linalg.norm(bounded)),
    )
