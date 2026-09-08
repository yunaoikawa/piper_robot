#!/usr/bin/env python3
"""Apply a tool-relative right-wrist residual to one semantic scene object.

The input correction is produced from a live SAM mask and the accepted
tool-relative visual goal.  This utility updates only the selected object's
planar translation; its height, geometry, and every other scene object remain
unchanged.  It is therefore suitable for the replan loop:

    simulate -> approach open -> observe -> update object -> simulate again
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np


def apply_scene_correction(
    scene: dict,
    correction: dict,
    *,
    semantic_name: str,
    maximum_correction_m: float,
    correction_source: str | None = None,
) -> dict:
    if correction.get("schema") != "piper_robot.right_visual_correction/v1":
        raise ValueError("unexpected right visual correction schema")
    delta = np.asarray(correction.get("world_delta_model_m"), dtype=float)
    if delta.shape != (3,) or not np.all(np.isfinite(delta)):
        raise ValueError("world_delta_model_m must contain three finite values")
    if abs(float(delta[2])) > 1e-9:
        raise ValueError("right visual scene correction must be planar")
    norm = float(np.linalg.norm(delta))
    maximum = float(maximum_correction_m)
    if not np.isfinite(maximum) or maximum <= 0.0 or norm > maximum:
        raise ValueError(
            f"scene correction norm {norm:.6f}m exceeds {maximum:.6f}m"
        )
    objects = [
        item
        for item in scene.get("objects", [])
        if item.get("semantic_name") == semantic_name
    ]
    if len(objects) != 1:
        raise ValueError(
            f"expected one {semantic_name!r} object, found {len(objects)}"
        )

    result = copy.deepcopy(scene)
    target = next(
        item
        for item in result["objects"]
        if item.get("semantic_name") == semantic_name
    )
    pose = np.asarray(target.get("pose_scene"), dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("target pose_scene must be a finite 4x4 transform")
    before = pose[:3, 3].copy()
    pose[:2, 3] += delta[:2]
    target["pose_scene"] = pose.tolist()
    target["status"] = "right_wrist_goal_residual_corrected"
    target.setdefault("perception", {})["runtime_absolute_pixels_used"] = False

    source = result.setdefault("source", {})
    history = source.setdefault("right_visual_refinement_history", [])
    history.append(
        {
            "created_at_s": time.time(),
            "correction_artifact": correction_source,
            "semantic_name": semantic_name,
            "before_xyz_m": before.tolist(),
            "world_delta_model_m": delta.tolist(),
            "after_xyz_m": pose[:3, 3].tolist(),
            "metric_scale_source": correction.get("correction", {}).get(
                "metric_scale_source"
            ),
            "successful_pose_or_trajectory_used": False,
            "runtime_absolute_pixels_used": False,
        }
    )
    result["operator_confirmed"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--correction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-name", default="petri_lid")
    parser.add_argument("--maximum-correction-m", type=float, default=0.05)
    args = parser.parse_args(argv)

    scene_path = Path(args.scene).resolve()
    correction_path = Path(args.correction).resolve()
    scene = json.loads(scene_path.read_text())
    correction = json.loads(correction_path.read_text())
    result = apply_scene_correction(
        scene,
        correction,
        semantic_name=args.semantic_name,
        maximum_correction_m=args.maximum_correction_m,
        correction_source=str(correction_path),
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "semantic_name": args.semantic_name,
                "world_delta_model_m": correction["world_delta_model_m"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
