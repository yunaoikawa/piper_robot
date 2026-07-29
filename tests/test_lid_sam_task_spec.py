#!/usr/bin/env python3
"""Validate the generic SAM target task and its demonstrated grasp goal."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from rollout.grasp_window import (
    GraspWindowTemplate,
    assess_grasp_window,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "src" / "configs"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains_key(value, forbidden):
    if isinstance(value, dict):
        return any(
            key in forbidden or _contains_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


task = load_json(CONFIG_ROOT / "pasteur_lid_sam_task.json")
runtime = load_json(CONFIG_ROOT / "pasteur_autonomous_lid_grasp.json")
selection = load_json(CONFIG_ROOT / "pasteur_grasp_window_selection.json")

assert task["schema"] == "piper_robot.sam_target_task/v2"
assert runtime["schema"] == "piper_robot.autonomous_sam_target_grasp_config/v2"
assert task["target"]["runtime_apriltag_required"] is False

goal = task["demonstration_reference"]
assert goal["selection"] == "start of longest gripper-closed run"
assert goal["frame_index"] == 82

relative_image = Path(goal["right_image_path"])
assert not relative_image.is_absolute()
image_path = (REPO_ROOT / relative_image).resolve()
image_path.relative_to(REPO_ROOT.resolve())
assert image_path.is_file()
assert sha256_file(image_path) == goal["right_image_sha256"]
assert (
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_image.as_posix()],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode
    == 0
)

image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
assert image is not None
assert list(image.shape[:2]) == goal["right_image_shape_hw"]
ellipse = goal["target_annotation"]["ellipse"]
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.ellipse(
    mask,
    tuple(ellipse["center_px"]),
    tuple(ellipse["axes_px"]),
    ellipse["angle_deg"],
    0,
    360,
    255,
    -1,
)
assessment, _ = assess_grasp_window(
    image,
    mask > 0,
    GraspWindowTemplate.from_dict(selection["template"]),
    method=selection["selected_method"],
)
assert assessment.allowed_to_close

assert task["grasp_window"]["runtime_absolute_pixels_forbidden"] is True
assert task["grasp_window"]["selected_method"] == selection["selected_method"]
assert selection["selected_method"] in task["grasp_window"]["candidate_methods"]
assert not _contains_key(
    (task, runtime),
    {
        "goal_px",
        "right_goal_px",
        "right_tolerance_px",
        "maximum_gripper_lid_m",
    },
)
gates = task["preclose_gates"]
assert gates["maximum_orientation_error_deg"] > 0
assert gates["maximum_tip_clearance_m"] > 0
assert gates["maximum_tip_penetration_m"] >= 0
assert gates["maximum_normalized_image_gap"] > 0
assert task["closure"]["verification_lift_m"] > 0
assert len(goal["grasp_orientation_wxyz"]) == 4
assert np.isclose(np.linalg.norm(goal["grasp_orientation_wxyz"]), 1.0, atol=1e-3)

print("Generic SAM target task specification checks passed")
