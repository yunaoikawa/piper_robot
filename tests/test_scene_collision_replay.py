from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.autonomous_mpc import ESDFGrid
from rollout.daily_scene import DailyScene
from src.replay_scene_collision import replay


q = [
    0.0268955231,
    1.8055955172,
    -0.6696653962,
    0.0435110591,
    -1.1529295444,
    2.3609592915,
]
grid = ESDFGrid(
    np.full((100, 100, 100), 0.20, dtype=np.float32),
    (-1.0, -1.0, -1.0),
    0.02,
    body_radius_m=0.0,
)
scene = DailyScene(
    scene_id="test-bench",
    revision=3,
    local_date="2026-01-01",
    status="confirmed",
    captured_at_s=0.0,
    calibration_id="test",
    camera_ids={},
)
report = replay(
    "robot/cone-e-description/robot-welded-base-and-lift.mjcf",
    [q, q],
    grid,
    scene,
)
assert report["sample_count"] == 2
assert report["scene_revision"] == 3
assert report["first_predicted_collision"] is None
assert report["minimum_clearance_m"] > 0.10

print("scene collision replay checks passed")
