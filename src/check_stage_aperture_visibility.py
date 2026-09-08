#!/usr/bin/env python3
"""Project and appearance-confirm one user-selected stage aperture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rollout.apriltag_retarget import detect_tags
from rollout.stage_aperture_visibility import (
    assess_aperture_visibility,
    project_tag_point,
    render_aperture_visibility,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(args.image)
    tag_id = int(config["anchor_tag_id"])
    tags = [tag for tag in detect_tags(image) if tag.tag_id == tag_id]
    if len(tags) != 1:
        raise RuntimeError(f"expected anchor tag {tag_id}; found {[tag.tag_id for tag in detect_tags(image)]}")
    tag = tags[0]
    uv, _, rms = project_tag_point(
        tag,
        np.asarray(config["camera_matrix"], dtype=float),
        float(config["anchor_tag_size_m"]),
        config["target_point_tag_xyz_m"],
    )
    result = assess_aperture_visibility(
        image,
        projected_uv=uv,
        anchor_perimeter_px=tag.perimeter,
        anchor_rms_px=rms,
        **config.get("visibility", {}),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), render_aperture_visibility(image, result)):
        raise RuntimeError(f"failed to write {args.output}")
    report = {
        "schema": "piper_robot.stage_aperture_visibility/v1",
        **result.__dict__,
        "visible": result.visible,
        "image": str(args.image.resolve()),
        "output": str(args.output.resolve()),
        "anchor_tag_id": tag_id,
        "anchor_perimeter_px": tag.perimeter,
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if result.visible else 2


if __name__ == "__main__":
    raise SystemExit(main())
