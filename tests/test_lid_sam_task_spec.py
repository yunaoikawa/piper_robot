#!/usr/bin/env python3
"""Validate the SAM lid task against its demonstrated grasp goal."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "src" / "configs"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


task = load_json(CONFIG_ROOT / "pasteur_lid_sam_task.json")
template = load_json(CONFIG_ROOT / "pasteur_lid_grasp_template.json")
vision = load_json(CONFIG_ROOT / "pasteur_lid_vision.json")

assert task["schema"] == "piper_robot.sam_lid_task/v1"
assert task["object"]["runtime_apriltag_required"] is False

goal = task["demonstration_goal"]
assert goal["selection"] == "start of longest gripper-closed run"
assert goal["frame_index"] == vision["goal_frame"] == 82

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

ellipse = goal["user_confirmed_lid_ellipse"]
vision_ellipse = vision["right_goal"]["lid_ellipse"]
assert ellipse["center_px"] == vision_ellipse["center"]
assert ellipse["axes_px"] == vision_ellipse["axes"]
assert ellipse["angle_deg"] == vision_ellipse["angle_deg"]

fine = task["right_fine_alignment"]
assert np.allclose(fine["goal_px"], template["fine_feature_goal_px"])
assert fine["require_lid_visible"] is True
assert task["descent"]["separate_operator_approval_required"] is True
assert task["descent"]["require_right_lid_visible"] is True
assert task["descent"]["require_torque_monitor"] is True
assert task["grasp"]["separate_operator_approval_required"] is True
assert task["grasp"]["empty_close_ratio"] == template["empty_close_ratio"]
assert len(task["grasp"]["success_requires"]) == 2

print("SAM lid task specification checks passed")
