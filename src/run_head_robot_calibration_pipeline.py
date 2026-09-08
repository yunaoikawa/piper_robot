#!/usr/bin/env python3
"""Observation-only fixed-head calibration and scene-registration pipeline.

This entrypoint consumes an already completed five-pose capture.  It never
imports robot RPC code and never sends hardware commands.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.semantic_scene_pipeline import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "piper_robot.head_robot_calibration_pipeline/v1"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _commands(args, output: Path) -> list[tuple[str, list[str]]]:
    calibration = output / "camera_calibration.json"
    registration = output / "robot_scene_registration.json"
    semantic_output = output / "semantic"
    calibrate = [
        sys.executable,
        str(REPO_ROOT / "src" / "calibrate_head_robot_from_cad.py"),
        "--capture",
        str(Path(args.capture).resolve()),
        "--profile",
        str(Path(args.profile).resolve()),
        "--output",
        str(calibration),
        "--sam-endpoint",
        args.sam_endpoint,
        "--minimum-views",
        "5",
    ]
    for mask in args.robot_mask:
        calibrate.extend(["--mask", mask])
    register = [
        sys.executable,
        str(REPO_ROOT / "src" / "register_fixed_head_scene.py"),
        "--fixed-capture",
        str(Path(args.capture).resolve()),
        "--camera-calibration",
        str(calibration),
        "--multiview-report",
        str(Path(args.multiview_report).resolve()),
        "--output",
        str(registration),
    ]
    semantic = [
        sys.executable,
        str(REPO_ROOT / "src" / "run_semantic_scene_pipeline.py"),
        "--multiview-report",
        str(Path(args.multiview_report).resolve()),
        "--profile",
        str(Path(args.profile).resolve()),
        "--output-dir",
        str(semantic_output),
        "--scene-registration-report",
        str(registration),
    ]
    if args.daily_scene:
        semantic.extend(["--daily-scene", str(Path(args.daily_scene).resolve())])
    if args.resume_confirmed:
        semantic.append("--resume-confirmed")
    if args.require_collision_ready:
        semantic.append("--require-collision-ready")
    return [
        ("calibrate_head_robot_from_cad", calibrate),
        ("register_fixed_head_scene", register),
        ("build_robot_frame_semantic_scene", semantic),
    ]


def _write_motion_config(args, output: Path) -> Path:
    template_path = Path(args.motion_config_template).resolve()
    payload = json.loads(template_path.read_text())
    payload["calibration"] = str(
        (output / "camera_calibration.json").resolve()
    )
    daily = dict(payload.get("daily_scene", {}))
    daily["state_path"] = str(
        (output / "semantic" / "lid_daily_scene.json").resolve()
    )
    payload["daily_scene"] = daily
    payload["scene_registration"] = str(
        (output / "robot_scene_registration.json").resolve()
    )
    payload["semantic_scene"] = str(
        (output / "semantic" / "scene" / "scene.json").resolve()
    )
    path = output / "pregrasp_motion_config.json"
    _write_json(path, payload)
    return path


def run(args) -> dict:
    output = Path(args.output_dir).resolve()
    commands = _commands(args, output)
    if args.dry_run:
        return {
            "schema": SCHEMA,
            "status": "dry_run",
            "commands_sent": False,
            "commands": [
                {"stage": stage, "command": command}
                for stage, command in commands
            ],
        }
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    stages = []
    environment = dict(os.environ)
    environment.setdefault("MUJOCO_GL", "egl")
    environment["PYTHONPATH"] = str(REPO_ROOT)
    for stage, command in commands:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path = logs / f"{stage}.stdout.log"
        stderr_path = logs / f"{stage}.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        record = {
            "stage": stage,
            "command": command,
            "returncode": completed.returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        stages.append(record)
        if completed.returncode != 0:
            report = {
                "schema": SCHEMA,
                "status": "failed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "commands_sent": False,
                "stages": stages,
                "failure": {
                    "stage": stage,
                    "returncode": completed.returncode,
                    "stderr_tail": completed.stderr[-4000:],
                },
            }
            _write_json(output / "pipeline_report.json", report)
            raise RuntimeError(
                f"{stage} failed with exit code {completed.returncode}; "
                f"see {output / 'pipeline_report.json'}"
            )
    registration_path = output / "robot_scene_registration.json"
    semantic_report = output / "semantic" / "pipeline_report.json"
    motion_config = _write_motion_config(args, output)
    dry_run_command = [
        sys.executable,
        str(REPO_ROOT / "src" / "run_autonomous_sam_lid_grasp.py"),
        "--config",
        str(motion_config),
        "--output-dir",
        str((output / "pregrasp_dry_run").resolve()),
        "--dry-run",
    ]
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands_sent": False,
        "inputs": {
            "capture_manifest": {
                "path": str(Path(args.capture).resolve() / "manifest.json"),
                "sha256": sha256_file(
                    Path(args.capture).resolve() / "manifest.json"
                ),
            },
            "multiview_report": {
                "path": str(Path(args.multiview_report).resolve()),
                "sha256": sha256_file(args.multiview_report),
            },
            "profile": {
                "path": str(Path(args.profile).resolve()),
                "sha256": sha256_file(args.profile),
            },
        },
        "stages": stages,
        "artifacts": {
            "camera_calibration": str(
                (output / "camera_calibration.json").resolve()
            ),
            "robot_scene_registration": str(registration_path.resolve()),
            "semantic_pipeline_report": str(semantic_report.resolve()),
            "phone_view": str((output / "semantic" / "index.html").resolve()),
            "pregrasp_motion_config": str(motion_config.resolve()),
        },
        "next_gate": {
            "action": "confirm daily semantic scene, rerun collision-ready gate, then run live pregrasp dry-run",
            "dry_run_command": dry_run_command,
            "physical_motion_authorized": False,
        },
    }
    _write_json(output / "pipeline_report.json", report)
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--multiview-report", required=True)
    parser.add_argument(
        "--profile",
        default="src/configs/pasteur_semantic_scene.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument(
        "--motion-config-template",
        default="src/configs/pasteur_autonomous_lid_grasp.json",
    )
    parser.add_argument(
        "--robot-mask",
        action="append",
        default=[],
        help="accepted VIEW:robot=/absolute/mask.png for offline replay",
    )
    parser.add_argument("--daily-scene")
    parser.add_argument("--resume-confirmed", action="store_true")
    parser.add_argument("--require-collision-ready", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
