#!/usr/bin/env python3
"""Convert a confirmed phone tap into a reusable tag-frame target point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rollout.apriltag_retarget import detect_tags  # noqa: E402
from rollout.stage_aperture_visibility import refine_tag_point_from_pixel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    selection = json.loads(args.selection.read_text())
    if selection.get("confirmed") is not True:
        raise RuntimeError("operator target selection is not confirmed")
    image_path = Path(selection["image_path"])
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    if [image.shape[1], image.shape[0]] != [
        int(selection["image_width_px"]), int(selection["image_height_px"])
    ]:
        raise RuntimeError("selection image dimensions changed")
    tag_id = int(config["anchor_tag_id"])
    tags = [tag for tag in detect_tags(image) if tag.tag_id == tag_id]
    if len(tags) != 1:
        raise RuntimeError(f"anchor tag {tag_id} is not uniquely visible")
    point, depth, rms = refine_tag_point_from_pixel(
        tags[0],
        np.asarray(config["camera_matrix"], dtype=float),
        float(config["anchor_tag_size_m"]),
        config["target_point_tag_xyz_m"],
        selection["pixel_uv"],
    )
    updated = dict(config)
    updated["target_point_tag_xyz_m"] = [float(value) for value in point]
    updated["operator_selection"] = {
        "selection": str(args.selection.resolve()),
        "image": str(image_path.resolve()),
        "pixel_uv": [float(value) for value in selection["pixel_uv"]],
        "retained_camera_depth_m": depth,
        "anchor_rms_px": rms,
        "refined_at_s": time.time(),
        "motion_authorized": False,
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(updated, indent=2) + "\n")
    print(json.dumps(updated["operator_selection"], indent=2))
    print("target_point_tag_xyz_m", updated["target_point_tag_xyz_m"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
