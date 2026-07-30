#!/usr/bin/env python3
"""Run SAM-first multiview reconstruction through validated MuJoCo output.

This is the Codex-free production entrypoint.  It orchestrates existing
deterministic stages, captures their logs, validates the resulting contracts,
and writes a compact pipeline report.  It never imports robot RPC and never
sends robot commands.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.semantic_scene_pipeline import load_profile, sha256_file


SCHEMA = "piper_robot.semantic_scene_pipeline_run/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_stage(
    name: str,
    command: list[str],
    *,
    logs_dir: Path,
    environment: dict[str, str] | None = None,
) -> dict:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{name}.stdout.log"
    stderr_path = logs_dir / f"{name}.stderr.log"
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    record = {
        "name": name,
        "command": command,
        "started_at_utc": datetime.fromtimestamp(
            started, timezone.utc
        ).isoformat(),
        "duration_s": float(time.time() - started),
        "returncode": int(result.returncode),
        "stdout_log": str(stdout_path.resolve()),
        "stderr_log": str(stderr_path.resolve()),
    }
    if result.returncode != 0:
        record["stderr_tail"] = stderr_path.read_text(encoding="utf-8")[-4000:]
    return record


def validate_completed_scene(scene: dict, profile_path: str | Path) -> dict:
    profile, catalog = load_profile(profile_path)
    reasons = []
    if scene.get("schema") != "piper_robot.multiview_completed_scene/v1":
        reasons.append("completed_scene_schema_invalid")
    if not scene.get("mujoco_compile", {}).get("ok", False):
        reasons.append("mujoco_compile_failed")
    objects = {
        item["semantic_name"]: item for item in scene.get("objects", ())
    }
    missing = sorted(set(profile.get("objects", ())) - set(objects))
    if missing:
        reasons.append(f"configured_objects_missing:{','.join(missing)}")
    required_nyu = set(
        profile.get("robot_end_effector", {}).get(
            "required_visual_geoms", ()
        )
    )
    missing_nyu = set(
        scene.get("mujoco_compile", {}).get(
            "missing_required_nyu_geoms", ()
        )
    )
    if required_nyu and missing_nyu:
        reasons.append("required_nyu_gripper_geoms_missing")
    if scene.get("mujoco_compile", {}).get("forbidden_stock_bodies"):
        reasons.append("forbidden_stock_gripper_links_present")

    optimization = {}
    for name, item in objects.items():
        definition = catalog.get(name)
        report = item.get("semantic_volume_fit")
        eligible = bool(
            definition is not None
            and definition.completion == "template"
            and definition.primitive == "box"
            and profile.get("semantic_volume_fit", {}).get("enabled", False)
        )
        if not eligible:
            continue
        if not report or not report.get("attempted"):
            reasons.append(f"semantic_volume_fit_not_attempted:{name}")
            continue
        if not report.get("accepted"):
            reasons.append(f"semantic_volume_fit_not_accepted:{name}")
        optimization[name] = {
            "accepted": bool(report.get("accepted")),
            "method": report.get("method"),
            "objective_before": report.get("initial", {}).get("objective"),
            "objective_after": report.get("optimized", {}).get("objective"),
            "improvement_fraction": report.get("improvement_fraction"),
            "known_free_intrusion_before": report.get("initial", {}).get(
                "known_free_intrusion_fraction"
            ),
            "known_free_intrusion_after": report.get("optimized", {}).get(
                "known_free_intrusion_fraction"
            ),
            "yaw_before_rad": report.get("initial", {}).get("yaw_rad"),
            "yaw_after_rad": report.get("optimized", {}).get("yaw_rad"),
        }
    display_ready = bool(
        scene.get("readiness", {}).get("display_ready", False) and not reasons
    )
    return {
        "accepted": display_ready,
        "display_ready": display_ready,
        "collision_ready": bool(
            scene.get("readiness", {}).get("collision_ready", False)
        ),
        "motion_ready": bool(
            scene.get("readiness", {}).get("motion_ready", False)
        ),
        "reasons": reasons,
        "scene_readiness_reasons": list(
            scene.get("readiness", {}).get("reasons", ())
        ),
        "configured_objects": list(profile.get("objects", ())),
        "observed_objects": sorted(objects),
        "semantic_volume_optimization": optimization,
    }


def _commands(args, multiview_report: Path, scene_dir: Path) -> list[tuple[str, list[str]]]:
    commands = []
    if args.capture:
        reconstruct = [
            sys.executable,
            str(REPO_ROOT / "src" / "reconstruct_multiview_scene.py"),
            "--capture",
            str(Path(args.capture).resolve()),
            "--profile",
            str(Path(args.profile).resolve()),
            "--output-dir",
            str(multiview_report.parent.resolve()),
            "--sam-endpoint",
            args.sam_endpoint,
            "--attempt",
            str(args.attempt),
        ]
        for mask in args.mask:
            reconstruct.extend(["--mask", mask])
        commands.append(("reconstruct_multiview", reconstruct))
    complete = [
        sys.executable,
        str(REPO_ROOT / "src" / "build_semantic_scene.py"),
        "--multiview-report",
        str(multiview_report.resolve()),
        "--profile",
        str(Path(args.profile).resolve()),
        "--output-dir",
        str(scene_dir.resolve()),
        "--daily-scene",
        str(
            Path(args.daily_scene).resolve()
            if args.daily_scene
            else (scene_dir / "daily_scene.json").resolve()
        ),
    ]
    if args.calibration_report:
        complete.extend(
            [
                "--calibration-report",
                str(Path(args.calibration_report).resolve()),
            ]
        )
    if args.resume_confirmed:
        complete.append("--resume-confirmed")
    commands.append(("complete_semantic_mujoco", complete))
    return commands


def _serve(directory: Path, bind: str, port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((bind, int(port)), handler)
    print(
        json.dumps(
            {
                "status": "serving",
                "directory": str(directory.resolve()),
                "bind": bind,
                "port": int(port),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def run(args) -> dict:
    root = Path(args.output_dir).resolve()
    scene_dir = root / "scene"
    logs_dir = root / "logs"
    if args.capture:
        multiview_report = root / "multiview" / "multiview_report.json"
    else:
        multiview_report = Path(args.multiview_report).resolve()
    commands = _commands(args, multiview_report, scene_dir)
    if args.dry_run:
        return {
            "schema": SCHEMA,
            "status": "dry_run",
            "dry_run": True,
            "commands": [
                {"stage": name, "command": command}
                for name, command in commands
            ],
        }

    root.mkdir(parents=True, exist_ok=True)
    environment = dict(__import__("os").environ)
    environment.setdefault("MUJOCO_GL", "egl")
    stages = []
    for name, command in commands:
        record = _run_stage(
            name,
            command,
            logs_dir=logs_dir,
            environment=environment,
        )
        stages.append(record)
        if record["returncode"] != 0:
            failed = {
                "schema": SCHEMA,
                "status": "failed",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit(),
                "commands_sent": False,
                "stages": stages,
                "failure": {
                    "stage": name,
                    "returncode": record["returncode"],
                    "stderr_tail": record.get("stderr_tail"),
                },
            }
            report_path = root / "pipeline_report.json"
            _write_json(report_path, failed)
            raise RuntimeError(
                f"{name} failed with exit code {record['returncode']}; "
                f"see {report_path} and {record['stderr_log']}"
            )
    scene_path = scene_dir / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    validation = validate_completed_scene(scene, args.profile)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "commands_sent": False,
        "inputs": {
            "capture": (
                str(Path(args.capture).resolve()) if args.capture else None
            ),
            "multiview_report": {
                "path": str(multiview_report),
                "sha256": sha256_file(multiview_report),
            },
            "profile": {
                "path": str(Path(args.profile).resolve()),
                "sha256": sha256_file(args.profile),
            },
            "calibration_report": (
                str(Path(args.calibration_report).resolve())
                if args.calibration_report
                else None
            ),
        },
        "stages": stages,
        "validation": validation,
        "artifacts": scene.get("artifacts", {}),
        "scene": str(scene_path.resolve()),
    }
    report_path = root / "pipeline_report.json"
    _write_json(report_path, report)
    (root / "index.html").write_text(
        """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width">
<meta http-equiv="refresh" content="0; url=scene/index.html">
<a href="scene/index.html">semantic scene</a>
""",
        encoding="utf-8",
    )
    if not validation["accepted"]:
        raise RuntimeError(
            "pipeline validation failed: "
            + ", ".join(validation["reasons"])
            + f"; see {report_path}"
        )
    if args.require_collision_ready and not validation["collision_ready"]:
        raise RuntimeError(
            f"collision readiness was required but not achieved; see {report_path}"
        )
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture")
    source.add_argument("--multiview-report")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument(
        "--mask",
        action="append",
        default=[],
        help="accepted VIEW:LABEL=/absolute/mask.png for reconstruction",
    )
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--calibration-report")
    parser.add_argument("--daily-scene")
    parser.add_argument("--resume-confirmed", action="store_true")
    parser.add_argument("--require-collision-ready", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8784)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.serve and not args.dry_run:
        _serve(Path(args.output_dir).resolve(), args.bind, args.port)


if __name__ == "__main__":
    main()
