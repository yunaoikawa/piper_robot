"""Validated, motion-free operator target selection for camera images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


@dataclass(frozen=True)
class OperatorTargetSelection:
    schema: str
    semantic_name: str
    image_path: str
    image_width_px: int
    image_height_px: int
    pixel_uv: tuple[float, float]
    normalized_uv: tuple[float, float]
    confirmed: bool
    confirmed_at_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_target_selection(
    payload: dict,
    *,
    semantic_name: str,
    image_path: str | Path,
    image_width_px: int,
    image_height_px: int,
    timestamp_s: float | None = None,
) -> OperatorTargetSelection:
    """Convert one UI POST into an immutable, normalized selection."""

    if payload.get("confirmed") is not True:
        raise ValueError("target selection must be explicitly confirmed")
    uv = np.asarray([payload.get("u"), payload.get("v")], dtype=float)
    if uv.shape != (2,) or not np.all(np.isfinite(uv)):
        raise ValueError("target selection requires finite u,v pixels")
    width = int(image_width_px)
    height = int(image_height_px)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not (0.0 <= uv[0] < width and 0.0 <= uv[1] < height):
        raise ValueError("target selection lies outside the image")
    normalized = uv / np.asarray([width, height], dtype=float)
    return OperatorTargetSelection(
        schema="piper_robot.operator_target_selection/v1",
        semantic_name=str(semantic_name),
        image_path=str(Path(image_path).resolve()),
        image_width_px=width,
        image_height_px=height,
        pixel_uv=(float(uv[0]), float(uv[1])),
        normalized_uv=(float(normalized[0]), float(normalized[1])),
        confirmed=True,
        confirmed_at_s=float(time.time() if timestamp_s is None else timestamp_s),
    )


def write_target_selection(path: str | Path, selection: OperatorTargetSelection) -> None:
    """Atomically persist a confirmed target for a polling orchestrator."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps(selection.to_dict(), indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
