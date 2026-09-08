#!/usr/bin/env python3
"""Save a tapped, scale-independent jaw side-view reference without motion."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
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


DEFAULT_LEVEL_CONFIG = ROOT / "src/configs/pasteur_fast_lid_grasp_level.json"


def _level_reference(path: str | Path) -> JawLevelReference:
    value = json.loads(Path(path).read_text())
    names = {
        "support_up_robot",
        "tip_baseline_ee",
        "approach_axis_ee",
        "open_tip_span_m",
        "maximum_checkpoint_tilt_deg",
        "maximum_planned_tilt_deg",
        "maximum_tip_height_difference_m",
    }
    kwargs = {name: value[name] for name in names if name in value}
    kwargs["source"] = str(value.get("schema", path))
    return JawLevelReference(**kwargs)


def _pose_difference(first, second) -> tuple[float, float]:
    first = np.asarray(first, dtype=float).reshape(7)
    second = np.asarray(second, dtype=float).reshape(7)
    first_q = first[:4] / np.linalg.norm(first[:4])
    second_q = second[:4] / np.linalg.norm(second[:4])
    cosine = float(np.clip(abs(first_q @ second_q), -1.0, 1.0))
    angle_deg = math.degrees(2.0 * math.acos(cosine))
    translation_m = float(np.linalg.norm(first[4:] - second[4:]))
    return translation_m, angle_deg


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
    parser.add_argument("--level-config", default=str(DEFAULT_LEVEL_CONFIG))
    parser.add_argument(
        "--physical-level-consensus",
        help="JSON calibration whose top-level level_consensus is accepted",
    )
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
    geometry = assess_jaw_level(right_pose, _level_reference(args.level_config))
    physical_consensus = None
    if args.physical_level_consensus:
        calibration = json.loads(Path(args.physical_level_consensus).read_text())
        physical_consensus = calibration.get("level_consensus")
        if not isinstance(physical_consensus, dict):
            raise ValueError("physical level calibration lacks level_consensus")
        calibrated_pose = calibration.get(
            "calibrated_pose_wxyz_xyz",
            calibration.get("calibrated_pose_wxyz_xyz_audit_only"),
        )
        if calibrated_pose is None:
            raise ValueError("physical level calibration lacks its measured pose")
        translation_error_m, rotation_error_deg = _pose_difference(
            right_pose, calibrated_pose
        )
        if translation_error_m > 0.002 or rotation_error_deg > 0.75:
            raise ValueError(
                "physical level consensus belongs to a different right-arm pose"
            )
    else:
        translation_error_m = None
        rotation_error_deg = None
    if args.operator_label == "strictly_level" and not (
        physical_consensus and physical_consensus.get("accepted") is True
    ):
        raise ValueError(
            "strictly_level requires an accepted independent RGB-D consensus"
        )
    result = {
        "schema": "piper_robot.jaw_side_view_reference/v1",
        "physical_arm": "right",
        "observer_arm": "left",
        "source": args.source,
        "operator_label": args.operator_label,
        "strict_geometry_accepted": bool(geometry.accepted),
        "strict_physical_level_accepted": bool(
            physical_consensus and physical_consensus.get("accepted") is True
        ),
        "use_as_strict_level_reference": bool(
            args.operator_label == "strictly_level"
            and physical_consensus
            and physical_consensus.get("accepted") is True
        ),
        "image_sha256": hashlib.sha256(encoded).hexdigest(),
        "shape_reference": asdict(descriptor),
        "observer_pose_wxyz_xyz": left_pose.tolist(),
        "observed_right_pose_wxyz_xyz": right_pose.tolist(),
        "geometry_assessment": geometry.to_dict(),
        "physical_level_consensus": physical_consensus,
        "physical_level_pose_match": {
            "translation_error_m": translation_error_m,
            "rotation_error_deg": rotation_error_deg,
            "maximum_translation_error_m": 0.002,
            "maximum_rotation_error_deg": 0.75,
        },
        "runtime_policy": {
            "shape_reference_checks_view_repeatability_only": True,
            "position_and_absolute_pixel_area_are_diagnostic_only": True,
            "strict_horizontal_gate": (
                "three independent RGB-D bursts plus calibrated attachment geometry"
            ),
            "single_rgbd_burst_can_authorize_level": False,
            "stale_probe_reuse_allowed": False,
            "reacquire_view_with": "measured_local_image_jacobian",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
