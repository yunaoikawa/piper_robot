#!/usr/bin/env python3
"""Replay recorded Piper q-waypoints against measured and semantic obstacles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.autonomous_mpc import (
    AnalyticObstacleSet,
    ESDFGrid,
    MuJoCoIKValidator,
)
from rollout.daily_scene import DailySceneStore


def load_q_waypoints(path):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload
    plan = payload.get("plan", payload)
    metadata = plan.get("metadata", {})
    return metadata["mujoco_q_waypoints"]


def load_esdf(path):
    with np.load(path) as data:
        transform = (
            data["T_camera_robot"]
            if "T_camera_robot" in data
            else data.get("T_esdf_robot", np.eye(4))
        )
    return ESDFGrid.from_npz(path, T_esdf_robot=transform, body_radius_m=0.0)


def replay(model, q_waypoints, esdf, daily_scene, *, collision_margin_m=0.0):
    if not q_waypoints:
        raise ValueError("recorded trajectory has no q-waypoints")
    validator = MuJoCoIKValidator(model, q_waypoints[0])
    obstacles = AnalyticObstacleSet(
        [item.__dict__ for item in daily_scene.objects]
    )
    samples = []
    first_collision = None
    for index, q in enumerate(q_waypoints):
        validator.set_right_q(q)
        measured = validator.esdf_clearance(esdf)
        proxy = obstacles.clearance(validator.robot_proxy_samples())
        minimum = min(
            np.inf if measured is None else measured,
            proxy,
        )
        row = {
            "index": index,
            "measured_esdf_clearance_m": measured,
            "analytic_proxy_clearance_m": (
                None if not np.isfinite(proxy) else proxy
            ),
            "minimum_clearance_m": (
                None if not np.isfinite(minimum) else float(minimum)
            ),
        }
        samples.append(row)
        if first_collision is None and minimum <= float(collision_margin_m):
            first_collision = row
    return {
        "schema": "piper_robot.scene_collision_replay/v1",
        "scene_id": daily_scene.scene_id,
        "scene_revision": daily_scene.revision,
        "sample_count": len(samples),
        "collision_margin_m": float(collision_margin_m),
        "first_predicted_collision": first_collision,
        "minimum_clearance_m": min(
            row["minimum_clearance_m"]
            for row in samples
            if row["minimum_clearance_m"] is not None
        ),
        "samples": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--q-waypoints", required=True)
    parser.add_argument("--esdf", required=True)
    parser.add_argument("--daily-scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collision-margin-m", type=float, default=0.0)
    args = parser.parse_args(argv)
    scene = DailySceneStore(args.daily_scene).load()
    if scene is None:
        raise SystemExit("daily scene does not exist")
    report = replay(
        args.model,
        load_q_waypoints(args.q_waypoints),
        load_esdf(args.esdf),
        scene,
        collision_margin_m=args.collision_margin_m,
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["first_predicted_collision"], indent=2))


if __name__ == "__main__":
    main()
