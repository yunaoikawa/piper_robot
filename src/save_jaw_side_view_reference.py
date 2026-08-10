#!/usr/bin/env python3
"""Save a tapped, scale-independent jaw side-view reference without motion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.rpc import RPCClient
from rollout.gripper_level import JawLevelReference, assess_jaw_level
from rollout.wrist_observer_tracking import blue_components, describe_blue_component


def _selected_component(image, point_xy):
    x, y = map(float, point_xy)
    components = blue_components(image)
    containing = []
    for component in components:
        left, top, width, height = component.bbox_xywh
        if left <= x <= left + width and top <= y <= top + height:
            containing.append(component)
    if containing:
        return min(
            containing,
            key=lambda value: np.linalg.norm(np.asarray(value.centroid_xy) - (x, y)),
        )
    if not components:
        raise ValueError("no blue component was found at the tapped side view")
    return min(
        components,
        key=lambda value: np.linalg.norm(np.asarray(value.centroid_xy) - (x, y)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="operator_confirmed_side_view")
    parser.add_argument(
        "--operator-label",
        choices=("approximately_level", "strictly_level"),
        default="approximately_level",
    )
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=8081)
    args = parser.parse_args()

    image_path = Path(args.image)
    encoded = image_path.read_bytes()
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {image_path}")
    component = _selected_component(image, (args.target_x, args.target_y))
    descriptor = describe_blue_component(image, component)

    rpc = RPCClient(args.rpc_host, args.rpc_port)
    rpc.init()
    right_pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    left_pose = np.asarray(rpc.get_left_ee_pose().parameters(), dtype=float)
    geometry = assess_jaw_level(right_pose, JawLevelReference())
    result = {
        "schema": "piper_robot.jaw_side_view_reference/v1",
        "physical_arm": "right",
        "observer_arm": "left",
        "source": args.source,
        "operator_label": args.operator_label,
        "strict_geometry_accepted": bool(geometry.accepted),
        "use_as_strict_level_reference": bool(
            args.operator_label == "strictly_level" and geometry.accepted
        ),
        "image_sha256": hashlib.sha256(encoded).hexdigest(),
        "shape_reference": asdict(descriptor),
        "observer_pose_wxyz_xyz": left_pose.tolist(),
        "observed_right_pose_wxyz_xyz": right_pose.tolist(),
        "geometry_assessment": geometry.to_dict(),
        "runtime_policy": {
            "shape_reference_checks_view_repeatability_only": True,
            "position_and_absolute_pixel_area_are_diagnostic_only": True,
            "strict_horizontal_gate": "rollout.gripper_level.assess_jaw_level",
            "reacquire_view_with": "measured_local_image_jacobian",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
