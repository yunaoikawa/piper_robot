"""Target-specific perception behind a common grasp-pipeline contract.

The motion pipeline owns homing, scene planning, camera lifetime, trajectory
streaming, replanning, descent, and closure.  An adapter owns only semantic
identity and object-relative grasp geometry.  This prevents a successful
pixel goal for one object (for example a marked Petri lid) from leaking into
another task (for example a cylindrical bottle cap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from rollout.grasp_window import ToolImageFrame


@dataclass(frozen=True)
class TargetObservation:
    center_px: tuple[float, float]
    center_uv: tuple[float, float]
    component_pixels: int
    equivalent_diameter_tool_units: float
    mask: np.ndarray
    diagnostics: dict


class TargetAdapter(Protocol):
    name: str
    geometry: str

    def observe(
        self, image_bgr: np.ndarray, tool_frame: ToolImageFrame
    ) -> TargetObservation:
        """Return one semantically identified target in tool coordinates."""


def stable_observation(
    camera,
    adapter: TargetAdapter,
    tool_frame: ToolImageFrame,
    *,
    frame_count: int = 3,
) -> tuple[TargetObservation, np.ndarray, float]:
    """Observe through one persistent camera and reject temporal jumps.

    Thresholds are normalized by the target's equivalent diameter rather than
    image pixels, so the same gate works across camera resolutions.
    """

    records = []
    for _ in range(int(frame_count)):
        image, timestamp = camera.frame()
        observation = adapter.observe(image, tool_frame)
        records.append((image, float(timestamp), observation))
    centers = np.asarray([item[2].center_px for item in records], dtype=float)
    median_center = np.median(centers, axis=0)
    representative = min(
        records,
        key=lambda item: float(
            np.linalg.norm(np.asarray(item[2].center_px) - median_center)
        ),
    )
    image, timestamp, observed = representative
    center_uv = tool_frame.image_to_tool([median_center])[0]
    pixels = int(np.median([item[2].component_pixels for item in records]))
    equivalent_diameter_px = 2.0 * np.sqrt(max(pixels, 1) / np.pi)
    spread_px = float(
        np.max(np.linalg.norm(centers - median_center, axis=1))
    )
    spread_scale = spread_px / max(equivalent_diameter_px, 1.0)
    diagnostics = dict(observed.diagnostics)
    diagnostics.update(
        {
            "temporal_frame_count": int(frame_count),
            "temporal_center_spread_target_diameters": spread_scale,
            "median_center_px": median_center.tolist(),
        }
    )
    stable = TargetObservation(
        center_px=(float(median_center[0]), float(median_center[1])),
        center_uv=(float(center_uv[0]), float(center_uv[1])),
        component_pixels=pixels,
        equivalent_diameter_tool_units=float(
            equivalent_diameter_px / tool_frame.scale_px
        ),
        mask=np.asarray(observed.mask, dtype=bool),
        diagnostics=diagnostics,
    )
    return stable, image, timestamp


def render_target_and_contact_goal(
    image_bgr: np.ndarray,
    observation: TargetObservation,
    tool_frame: ToolImageFrame,
    contact_center_uv,
) -> np.ndarray:
    out = np.asarray(image_bgr).copy()
    mask = np.asarray(observation.mask, dtype=bool)
    out[mask] = (
        0.45 * out[mask] + 0.55 * np.asarray([0, 255, 0])
    ).astype(np.uint8)
    center = tuple(np.rint(observation.center_px).astype(int))
    goal = tuple(
        np.rint(tool_frame.tool_to_image([contact_center_uv])[0]).astype(int)
    )
    cv2.circle(out, center, 7, (0, 0, 255), -1)
    cv2.circle(out, goal, 10, (0, 255, 255), 3)
    cv2.line(out, center, goal, (0, 255, 255), 2, cv2.LINE_AA)
    return out
