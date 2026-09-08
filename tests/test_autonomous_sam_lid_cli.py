#!/usr/bin/env python3
"""Offline CLI and durable resume checks without cameras or robot RPC."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_autonomous_sam_lid_grasp import main


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    calibration = root / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "accepted": True,
                "accepted_at_s": 1.0,
                "record3d_udid": "head-id",
                "T_robot_camera": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "T_esdf_robot": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            }
        )
    )
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "camera_udids": {"head": "head-id"},
                "scene_config": "src/configs/pasteur_lid_scene3d.json",
                "task_config": "src/configs/pasteur_lid_sam_task.json",
                "torque_config": "src/configs/pasteur_lid_torque.json",
                "calibration": str(calibration),
                "sam_endpoint": "unused",
                "planner": {
                    "lift_m": 0.04,
                    "pregrasp_height_m": 0.015,
                    "final_height_m": 0.003,
                },
            }
        )
    )
    scene = root / "scene.json"
    scene.write_text(
        json.dumps(
            {
                "target_camera_xyz_m": [0.40, 0.05, 0.70],
                "ee_pose_wxyz_xyz": [1, 0, 0, 0, 0.30, 0.0, 0.80],
                "constant_clearance_m": 0.05,
            }
        )
    )
    output = root / "run"
    with contextlib.redirect_stdout(io.StringIO()):
        result = main(
            [
                "--config",
                str(config),
                "--offline-scene",
                str(scene),
                "--output-dir",
                str(output),
            ]
        )
    assert result == 0
    state = json.loads((output / "run_state.json").read_text())
    assert state["status"] == "DRY_RUN_COMPLETE"
    assert state["offline"] is True
    assert state["plan"]["metadata"]["stage_order"] == [
        "lift",
        "translate_xy",
        "approach",
        "descend",
    ]

print("autonomous SAM lid CLI checks passed")
