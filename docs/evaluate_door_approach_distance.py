#!/usr/bin/env python3
"""Offline, explicitly conditional EE-distance audit of Door approach records.

The accepted RGB-D-to-robot bridge is absent. Never label this fixed-door
reference comparison as a fully registered handle distance or motion target.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs"))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_door_first_approach import read_json, source
from rollout.articulated_appliance import load_endpoint, _registration, _tag_map, _warp_depth
from rollout.teleop_trajectory_stream import ProductionRightFK

REFERENCE = "data/runs/pasteur/incubator_door_20260808_retry6_yaw_aligned_contact/after"
ALTERNATE = "data/runs/pasteur/incubator_auto_open_20260808_demo_retry2/05_aligned-contact/motion/after"
IMAGE_REPORT = "docs/assets/code_as_learning_machine/door_first_approach_report.json"
MODEL = "robot/cone-e-description/robot-welded-base-and-lift.mjcf"
CALIBRATION = "src/configs/pasteur_head_robot_calibration.json"
PROFILE = "src/configs/pasteur_incubator_door_demo.json"


def pose_difference(actual, goal):
    actual, goal = np.asarray(actual, float), np.asarray(goal, float)
    if actual.shape != (7,) or goal.shape != (7,) or not np.isfinite([actual, goal]).all():
        raise ValueError("Expected finite [qw,qx,qy,qz,x,y,z] poses")
    rotations = [Rotation.from_quat(p[[1, 2, 3, 0]]) for p in (actual, goal)]
    delta = goal[4:] - actual[4:]
    return {"delta_xyz_mm": (delta * 1000).tolist(),
            "distance_mm": float(np.linalg.norm(delta) * 1000),
            "orientation_difference_deg": float(np.degrees((rotations[1].inv() * rotations[0]).magnitude()))}


def checked_observation(directory, fk):
    path = directory + "/observation.json"
    observation = read_json(path)
    pose = observation["right_ee_wxyz_xyz"]
    check = pose_difference(fk.pose(observation["right_q_rad"]).parameters(), pose)
    if check["distance_mm"] > 0.001 or check["orientation_difference_deg"] > 0.0001:
        raise ValueError(f"Archived joint state does not reproduce EE pose: {directory}")
    return observation, {"observation": source(path), "fk_reconstruction_error": check}


def head_consistency(directory, reference, settings):
    """Image/depth consistency only. A homography is NOT an SE(3) bridge."""
    live = load_endpoint(ROOT / directory / "head.png", ROOT / directory / "head_depth.npy")
    # Do not use the appliance anchor (13) to fit the registration that tests it.
    settings = {**settings, "fixed_tag_ids": [3, 12], "minimum_fixed_tags": 2}
    h, info = _registration(live.image_bgr, reference.image_bgr, settings)
    tags = _tag_map(reference.image_bgr, settings)
    live_tags = _tag_map(live.image_bgr, settings)
    if 13 not in tags or 13 not in live_tags:
        raise ValueError("Held-out appliance tag 13 is missing")
    projected = cv2.perspectiveTransform(live_tags[13].corners.astype(np.float32)[None], h)[0]
    corner_error = np.linalg.norm(projected - tags[13].corners, axis=1)
    depth_h, depth_w = reference.depth_m.shape
    image_h, image_w = reference.image_bgr.shape[:2]
    corners = tags[13].corners
    # Interior-only diagnostic avoids the marker's depth-discontinuity boundary.
    interior = corners.mean(0) + 0.65 * (corners - corners.mean(0))
    polygon = np.rint(interior * [depth_w / image_w, depth_h / image_h]).astype(np.int32)
    mask = np.zeros((depth_h, depth_w), np.uint8)
    cv2.fillConvexPoly(mask, polygon, 1)
    warped = _warp_depth(live, reference, h)
    valid = (mask > 0) & np.isfinite(warped) & np.isfinite(reference.depth_m)
    valid &= (warped > 0) & (reference.depth_m > 0)
    if not valid.any():
        raise ValueError("No valid interior depth pixels at held-out appliance tag")
    return {"status": "diagnostic_not_metric_registration", **info,
            "held_out_tag_id": 13,
            "held_out_corner_median_error_px": float(np.median(corner_error)),
            "held_out_depth_patch_pixel_count": int(valid.sum()),
            "held_out_depth_median_absolute_difference_mm": float(np.median(abs(warped[valid] - reference.depth_m[valid])) * 1000),
            "translation_correction_applied": False,
            "sources": [source(directory + "/head.png"), source(directory + "/head_depth.npy")]}


def evaluate():
    cv2.setRNGSeed(0)
    evidence = read_json(IMAGE_REPORT)
    fk = ProductionRightFK(MODEL)
    goal, goal_audit = checked_observation(REFERENCE, fk)
    alternate, alternate_audit = checked_observation(ALTERNATE, fk)
    reference = load_endpoint(ROOT / REFERENCE / "head.png", ROOT / REFERENCE / "head_depth.npy")
    calibration = read_json(CALIBRATION)
    if calibration.get("accepted"):
        raise ValueError("Calibration status changed: re-audit before using the fixed-door fallback")
    rows = []
    for spec in evidence["configurations"]:
        if spec["status"] != "measured":
            continue
        directory = str(Path(spec["motion"]).parent / "after")
        actual, audit = checked_observation(directory, fk)
        motion = read_json(spec["motion"])
        try:
            head = head_consistency(directory, reference, read_json(PROFILE)["state_detection"])
        except (ValueError, RuntimeError) as exc:
            head = {"status": "unavailable", "reason": str(exc)}
        rows.append({"stage": f"D{len(rows)+1}", "historical_stage": spec["stage"],
                     "status": "conditional_fixed_door_estimate",
                     "actual_ee_pose_wxyz_xyz": actual["right_ee_wxyz_xyz"],
                     "stage_before_ee_pose_wxyz_xyz": motion["before"]["right_ee_wxyz_xyz"],
                     "settle_status": ({key: motion["settle"].get(key) for key in ("accepted", "motion_eligible")}
                                       if "settle" in motion else None),
                     **pose_difference(actual["right_ee_wxyz_xyz"], goal["right_ee_wxyz_xyz"]),
                     "alternate_contact_distance_mm": pose_difference(actual["right_ee_wxyz_xyz"], alternate["right_ee_wxyz_xyz"])["distance_mm"],
                     "image_position_error": spec["uv_error"], "robot_state_audit": audit,
                     "head_consistency": head})
    before_fix = next(row for row in rows if row["historical_stage"] == "D4_pre_parser_fix")
    after_fix = next(row for row in rows if row["historical_stage"] == "D4")
    return {
        "schema": "door_approach_distance_audit/v1",
        "metric": "1000 * norm(recorded_actual_EE_xyz - recorded_success_contact_EE_xyz)",
        "unit": "mm",
        "fully_registered_3d_distance_available": False,
        "registration_blocker": "Head-to-robot calibration is unaccepted and its transform is null. No demonstrated appliance-frame enrollment was attached to these runs.",
        "reference": {"definition": "Earlier same-session yaw-aligned open-jaw contact followed by proof retention and an observed open endpoint (historical D3); not the later autonomous result being scored.",
                      "pose_wxyz_xyz": goal["right_ee_wxyz_xyz"], **goal_audit,
                      "sources": [source(REFERENCE + "/head.png"), source(REFERENCE + "/head_depth.npy"),
                                  source("data/runs/pasteur/incubator_door_20260808_retry6_yaw_aligned_proof/proof_state.json")]},
        "reference_sensitivity": {"definition": "Later autonomous contact; sensitivity check only, not a confidence interval or a separate approach sample.",
                                  "reference_separation": pose_difference(goal["right_ee_wxyz_xyz"], alternate["right_ee_wxyz_xyz"]),
                                  **alternate_audit},
        "limitations": [
            "Assumes the closed door and physical robot base stayed fixed in the same robot coordinate frame.",
            "Measures the EE-frame origin, not the jaw-contact point; orientation is reported separately, not mixed into millimetres.",
            "Reference is a demonstrated workable contact, not the unique optimal contact or shortest collision-free travel distance.",
            "Preclose deliberately leaves a contact-approach gap; a nonzero distance is expected even for a good approach.",
            "Head registration uses fixed tags 3 and 12 and holds out appliance tag 13. Its 2-D/depth diagnostics do not establish full 3-D stationarity or submillimetre accuracy.",
            "Joint/FK consistency verifies bookkeeping only, not independent physical metrology.",
            "Do not use these retrospectively estimated distances as robot commands or replace the image-only figure silently.",
        ],
        "configurations": rows,
        "continuation_audit": {
            "pre_parser_fix_stage": "D3", "post_parser_fix_stage": "D4",
            "identical_recorded_after_pose": bool(np.array_equal(before_fix["actual_ee_pose_wxyz_xyz"], after_fix["actual_ee_pose_wxyz_xyz"])),
            "retry_starts_at_previous_after_pose": bool(np.array_equal(before_fix["actual_ee_pose_wxyz_xyz"], after_fix["stage_before_ee_pose_wxyz_xyz"])),
            "interpretation": "D4 continues from D3's stopped pose; do not count them as independent reset approach trials. Both stages were motion-eligible but not accepted by the stricter settling tolerance."},
        "sources": [source(IMAGE_REPORT), source(MODEL), source(CALIBRATION), source(PROFILE),
                    source("rollout/teleop_trajectory_stream.py"), source("rollout/articulated_appliance.py")]}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, allow_nan=False))
