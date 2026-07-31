#!/usr/bin/env python3
"""One-command, observation-only reconstruction and measured replay pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "piper_robot.offline_scene_replay_run/v1"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _hash_path(hasher, path: Path) -> None:
    path = path.resolve()
    hasher.update(str(path).encode())
    if path.is_file():
        hasher.update(path.read_bytes())
        return
    if not path.is_dir():
        hasher.update(b"MISSING")
        return
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        hasher.update(str(child.relative_to(path)).encode())
        with child.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                hasher.update(block)


def _fingerprint(command: list[str], inputs: list[Path]) -> str:
    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    )
    for path in inputs:
        _hash_path(hasher, path)
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _validate_physical_arm_identity(
    *,
    production_calibration_mapping: dict,
    semantic_mapping: dict,
    carving: dict,
    target_profile: dict,
    replay_profile: dict,
    target_capture_manifests: list[dict],
    replay_capture_manifests: list[dict],
) -> dict:
    """Fail closed if sensing, kinematics, carving, and replay change arms."""

    bridge = target_profile["kinematic_bridge"]
    physical_arm = bridge["physical_arm"]
    target_branch = bridge["model_branch"]
    replay_branch = replay_profile["physical_right_model_branch"]
    carving_branch = carving["physical_right_model_branch"]
    if physical_arm != "right":
        raise ValueError(
            f"expected right-wrist evidence, got physical arm {physical_arm!r}"
        )
    if set(production_calibration_mapping) != {"left", "right"}:
        raise ValueError(
            "production calibration mapping must contain physical left/right"
        )
    if semantic_mapping != {"left": "left", "right": "right"}:
        raise ValueError(
            "semantic Piper branches must preserve physical identity; got "
            f"{semantic_mapping!r}"
        )
    authoritative_branch = semantic_mapping[physical_arm]
    branches = {
        "semantic_robot": authoritative_branch,
        "wrist_target": target_branch,
        "collision_carving": carving_branch,
        "trajectory_replay": replay_branch,
    }
    if len(set(branches.values())) != 1:
        raise ValueError(
            "physical-right branch mismatch: "
            + ", ".join(f"{name}={value}" for name, value in branches.items())
        )
    expected_prefix = f"{target_branch}/"
    prefixes = set(carving.get("robot_body_prefixes", []))
    if prefixes != {expected_prefix}:
        raise ValueError(
            "collision carving must be scoped only to physical-right branch "
            f"{expected_prefix!r}; got {sorted(prefixes)!r}"
        )
    target_capture_labels = [
        manifest.get("camera_label") for manifest in target_capture_manifests
    ]
    replay_capture_labels = [
        manifest.get("camera_label") for manifest in replay_capture_manifests
    ]
    capture_labels = target_capture_labels + replay_capture_labels
    if (
        not target_capture_labels
        or not replay_capture_labels
        or set(capture_labels) != {physical_arm}
    ):
        raise ValueError(
            "target/replay capture labels do not match the physical arm: "
            f"expected {physical_arm!r}, got {capture_labels!r}"
        )
    return {
        "accepted": True,
        "physical_arm": physical_arm,
        "model_branch": target_branch,
        "production_calibration_mapping": production_calibration_mapping,
        "semantic_mapping": semantic_mapping,
        "target_capture_labels": target_capture_labels,
        "replay_capture_labels": replay_capture_labels,
        "stages": branches,
        "policy": (
            "production calibration and semantic planning use separate branch "
            "namespaces; right-wrist RGB-D and right controller joints remain "
            "on semantic right through target, carving, and replay"
        ),
    }


def _run_stage(
    *,
    name: str,
    command: list[str],
    inputs: list[Path],
    outputs: list[Path],
    output_root: Path,
    force: bool,
    environment: dict[str, str] | None = None,
) -> dict:
    cache_dir = output_root / "cache"
    log_dir = output_root / "logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(command, inputs)
    cache_path = cache_dir / f"{name}.json"
    if (
        not force
        and cache_path.exists()
        and all(path.exists() for path in outputs)
    ):
        cache = json.loads(cache_path.read_text())
        expected_hashes = cache.get("output_sha256", {})
        outputs_match = bool(expected_hashes) and all(
            expected_hashes.get(str(path)) == _file_sha256(path)
            for path in outputs
        )
        if cache.get("fingerprint") == fingerprint and outputs_match:
            return {
                **cache,
                "status": "cached",
                "cache_hit": True,
                "duration_s": 0.0,
            }
    started = time.monotonic()
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    merged_environment = dict(__import__("os").environ)
    merged_environment.update(environment or {})
    existing_pythonpath = merged_environment.get("PYTHONPATH", "")
    merged_environment["PYTHONPATH"] = (
        str(ROOT)
        if not existing_pythonpath
        else str(ROOT) + __import__("os").pathsep + existing_pythonpath
    )
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            env=merged_environment,
            check=False,
        )
    duration = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"stage {name} failed ({completed.returncode}); "
            f"see {stderr_path}"
        )
    missing = [str(path) for path in outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"stage {name} omitted outputs: {missing}")
    cache = {
        "name": name,
        "status": "completed",
        "cache_hit": False,
        "fingerprint": fingerprint,
        "command": command,
        "inputs": [str(path) for path in inputs],
        "outputs": [str(path) for path in outputs],
        "output_sha256": {
            str(path): _file_sha256(path) for path in outputs
        },
        "duration_s": duration,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    cache_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return cache


def _write_mobile_index(output: Path, pipeline: dict) -> Path:
    target = pipeline["summary"]["latest_target_center_scene_m"]
    shift = pipeline["summary"]["recorded_to_latest_shift_m"]
    shift_mm = 1000.0 * sum(value * value for value in shift) ** 0.5
    moving_clear = pipeline["summary"]["moving_arm_path_clear"]
    global_clear = pipeline["summary"]["global_scene_home_clear"]
    current_overlay_link = (
        '<a href="current_scene/current_objects_overlay.png">'
        "<b>現在head SAMの皿・蓋</b></a>"
        if pipeline["summary"].get("current_object_refresh_accepted")
        else ""
    )
    page = f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Pasteur offline replay</title>
<style>
:root{{color-scheme:dark}}body{{margin:0;background:#08111f;color:#f8fafc;
font:16px/1.5 system-ui,sans-serif}}main{{max-width:760px;margin:auto;padding:20px}}
.card,a{{display:block;background:#172033;border-radius:14px;padding:14px;
margin:12px 0;color:#7dd3fc;text-decoration:none}}.ok{{color:#86efac}}
.warn{{color:#fbbf24}}video,img{{width:100%;border-radius:12px;background:#000}}
small{{color:#a8b4c5}}code{{overflow-wrap:anywhere}}
</style></head><body><main>
<h1>Pasteur オフライン再構成・軌道</h1>
<div class="card"><b class="ok">実機コマンド送信なし</b><br>
右腕の計画軌道: <b class="{'ok' if moving_clear else 'warn'}">
{'衝突なし' if moving_clear else '未承認'}</b><br>
シーン全体: <b class="{'ok' if global_clear else 'warn'}">
{'衝突なし' if global_clear else '静止左腕側に未解消接触あり'}</b></div>
<video controls playsinline preload="metadata"
 poster="render/recorded_replay_start.png">
<source src="render/recorded_replay.mp4" type="video/mp4"></video>
<small>シアンのrobot/軌道=物理右手、灰色robot=静止した物理左手、
黄=成功時対象、青=最新推定。記録点は実測、点間はMuJoCoで再計画。</small>
<a href="mobile/mujoco_home.html"><b>MuJoCo home（軽量3D）</b></a>
<a href="mobile/semantic_3d.html"><b>SAM意味付き3D（軽量）</b></a>
<a href="mobile/source_esdf_scene.html"><b>ESDF（軽量）</b></a>
<a href="render/recorded_replay_final.png"><b>最終姿勢画像</b></a>
{current_overlay_link}
<a href="target/overlays/05_latest_target_after_drop.png">
<b>最新対象のRGB-D検出</b></a>
<a href="pipeline_report.json"><b>機械可読パイプライン報告</b></a>
<div class="card"><small>最新中心 [m]</small><br><code>
{html.escape(json.dumps(target))}</code><br>
成功時から最新位置までのXY差: {shift_mm:.1f} mm</div>
</main></body></html>"""
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def run(
    config_path: Path,
    output: Path,
    *,
    force: bool,
    sam_endpoint: str | None = None,
) -> dict:
    started = time.monotonic()
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    stages = []

    base_config = config["base_scene"]
    multiview_report = _resolve(base_config["multiview_report"])
    semantic_profile = _resolve(base_config["profile"])
    semantic_profile_data = json.loads(semantic_profile.read_text())
    catalog = (
        semantic_profile.parent / semantic_profile_data["catalog"]
    ).resolve()
    base_dir = output / "base_scene"
    base_scene_json = base_dir / "scene.json"
    base_scene_model = base_dir / "scene.xml"
    base_positioned_robot = base_dir / "positioned_robot.xml"
    base_semantic_view = base_dir / "semantic_3d.html"
    base_esdf_view = base_dir / "source_esdf_scene.html"
    base_command = [
        python,
        str(ROOT / "src/build_semantic_scene.py"),
        "--multiview-report",
        str(multiview_report),
        "--profile",
        str(semantic_profile),
        "--output-dir",
        str(base_dir),
        "--daily-scene",
        str(base_dir / "daily_scene.json"),
    ]
    stages.append(
        _run_stage(
            name="base_semantic_scene",
            command=base_command,
            inputs=[
                multiview_report.parent,
                semantic_profile,
                catalog,
                ROOT / "src/build_semantic_scene.py",
                ROOT / "rollout/semantic_scene_pipeline.py",
            ],
            outputs=[
                base_scene_json,
                base_scene_model,
                base_positioned_robot,
                base_semantic_view,
                base_esdf_view,
            ],
            output_root=output,
            force=force,
        )
    )
    base_scene = json.loads(base_scene_json.read_text())
    if not (
        base_scene["readiness"]["display_ready"]
        and base_scene["mujoco_compile"]["ok"]
    ):
        raise ValueError("base semantic scene display/compile gates failed")

    alignment_config = config["alignment"]
    alignment_dir = output / "alignment"
    alignment_report = alignment_dir / "alignment_report.json"
    alignment_scene = alignment_dir / "scene.mjcf"
    alignment_command = [
        python,
        str(ROOT / "src/refine_scene_robot_alignment.py"),
        "--reference-report",
        str(_resolve(alignment_config["reference_report"])),
        "--reference-capture",
        str(_resolve(alignment_config["reference_capture"])),
        "--current-capture",
        str(_resolve(alignment_config["current_capture"])),
        "--robot-mask-dir",
        str(_resolve(alignment_config["robot_mask_dir"])),
        "--scene-json",
        str(base_scene_json),
        "--scene-model",
        str(base_scene_model),
        "--positioned-robot",
        str(base_positioned_robot),
        "--output-dir",
        str(alignment_dir),
        "--tag-id",
        str(alignment_config.get("tag_id", 3)),
        "--tag-size-m",
        str(alignment_config.get("tag_size_m", 0.06)),
        "--support-plane-z-m",
        str(alignment_config["support_plane_z_m"]),
        "--semantic-exclusion-margin-m",
        str(alignment_config.get("semantic_exclusion_margin_m", 0.02)),
        "--maximum-independent-base-translation-m",
        str(
            alignment_config.get(
                "maximum_independent_base_translation_m",
                semantic_profile_data.get(
                    "robot_alignment_refinement",
                    {},
                ).get("maximum_translation_m", 0.15),
            )
        ),
        "--minimum-base-nearest-ratio",
        str(alignment_config.get("minimum_base_nearest_ratio", 2.0)),
    ]
    if alignment_config.get("baseline_is_home", True):
        alignment_command.append("--baseline-is-home")
    # Expanded explicitly for clear provenance and deterministic directory hash.
    alignment_inputs = [
        ROOT / "src/refine_scene_robot_alignment.py",
        ROOT / "rollout/scene_registration.py",
        _resolve(alignment_config["reference_report"]),
        _resolve(alignment_config["reference_capture"]),
        _resolve(alignment_config["current_capture"]),
        _resolve(alignment_config["robot_mask_dir"]),
        base_scene_json,
        base_scene_model,
        base_positioned_robot,
    ]
    stages.append(
        _run_stage(
            name="alignment",
            command=alignment_command,
            inputs=alignment_inputs,
            outputs=[alignment_report, alignment_scene],
            output_root=output,
            force=force,
        )
    )
    alignment = json.loads(alignment_report.read_text())
    if not (
        alignment.get("accepted")
        and alignment["persistent_depth_robot_fit"]["accepted"]
        and alignment["home_pose_provenance"]["accepted"]
        and alignment.get("commands_sent") is False
    ):
        raise ValueError("depth-aware alignment gates failed")

    carving = config["collision_carving"]
    replay_config = _resolve(config["recorded_replay_config"])
    replay_profile = json.loads(replay_config.read_text())
    target_config = _resolve(config["wrist_target_config"])
    target_profile = json.loads(target_config.read_text())
    target_capture_manifests = [
        json.loads((_resolve(item["path"]) / "manifest.json").read_text())
        for item in target_profile["captures"]
    ]
    replay_capture_manifests = [
        json.loads((_resolve(item["capture"]) / "manifest.json").read_text())
        for item in replay_profile["measured_keyframes"]
    ]
    arm_identity = _validate_physical_arm_identity(
        production_calibration_mapping=semantic_profile_data[
            "robot_calibration"
        ][
            "physical_to_production_branch"
        ],
        semantic_mapping=semantic_profile_data["semantic_robot"][
            "physical_to_semantic_branch"
        ],
        carving=carving,
        target_profile=target_profile,
        replay_profile=replay_profile,
        target_capture_manifests=target_capture_manifests,
        replay_capture_manifests=replay_capture_manifests,
    )
    collision_dir = output / "collision_scene"
    collision_model = collision_dir / "scene.mjcf"
    carve_report = collision_dir / "carve_report.json"
    carve_command = [
        python,
        str(ROOT / "src/carve_robot_semantic_collision.py"),
        "--model",
        str(alignment_scene),
        "--alignment-report",
        str(alignment_report),
        "--output-model",
        str(collision_model),
        "--report",
        str(carve_report),
        "--verified-keyframes-config",
        str(replay_config),
        "--physical-right-model-branch",
        carving.get("physical_right_model_branch", "right"),
        "--robot-clearance-margin-m",
        str(carving.get("robot_clearance_margin_m", 0.03)),
        "--maximum-removed-fraction",
        str(carving.get("maximum_removed_fraction", 0.5)),
    ]
    allow_rejected_carving = bool(
        config.get("collision_carving_allow_rejected_display_only", False)
    )
    if allow_rejected_carving:
        carve_command.append("--allow-rejected-display-only")
    for prefix in carving["allowed_body_prefixes"]:
        carve_command.extend(["--allowed-body-prefix", prefix])
    for prefix in carving.get("robot_body_prefixes", ["left/"]):
        carve_command.extend(["--robot-body-prefix", prefix])
    stages.append(
        _run_stage(
            name="semantic_collision_carving",
            command=carve_command,
            inputs=[
                alignment_scene,
                alignment_report,
                replay_config,
                ROOT / "rollout/semantic_collision_carving.py",
                ROOT / "src/carve_robot_semantic_collision.py",
            ],
            outputs=[collision_model, carve_report],
            output_root=output,
            force=force,
        )
    )
    carve = json.loads(carve_report.read_text())
    if not carve.get("accepted") and not allow_rejected_carving:
        raise ValueError("semantic collision carving gate failed")

    target_dir = output / "target"
    target_report = target_dir / "wrist_target_report.json"
    object_scene = target_dir / "latest_target_scene.json"
    target_command = [
        python,
        str(ROOT / "src/calibrate_wrist_rgbd_target.py"),
        "--config",
        str(target_config),
        "--scene-model",
        str(collision_model),
        "--output-dir",
        str(target_dir),
    ]
    target_inputs = [
        target_config,
        collision_model,
        ROOT / "rollout/wrist_rgbd_target.py",
        ROOT / "src/calibrate_wrist_rgbd_target.py",
        *[_resolve(item["path"]) for item in target_profile["captures"]],
    ]
    stages.append(
        _run_stage(
            name="wrist_rgbd_target",
            command=target_command,
            inputs=target_inputs,
            outputs=[target_report, object_scene],
            output_root=output,
            force=force,
        )
    )
    target = json.loads(target_report.read_text())
    if not (
        target.get("accepted")
        and target["target_geometry_gate"]["accepted"]
        and target.get("commands_sent") is False
    ):
        raise ValueError("wrist RGB-D target gates failed")

    replay_model = collision_model
    replay_object_scene = object_scene
    current_refresh_report = None
    refresh_config = config.get("current_object_refresh")
    if refresh_config:
        current_dir = output / "current_scene"
        current_model = current_dir / "scene.mjcf"
        current_object_scene = current_dir / "latest_target_scene.json"
        current_report_path = current_dir / "current_object_report.json"
        current_overlay = current_dir / "current_objects_overlay.png"
        refresh_command = [
            python,
            str(ROOT / "src/update_current_semantic_objects.py"),
            "--config",
            str(config_path),
            "--model",
            str(collision_model),
            "--previous-object-scene",
            str(object_scene),
            "--output-dir",
            str(current_dir),
        ]
        effective_sam_endpoint = (
            sam_endpoint or refresh_config.get("sam_endpoint")
        )
        if effective_sam_endpoint:
            refresh_command.extend(
                ["--sam-endpoint", str(effective_sam_endpoint)]
            )
        refresh_inputs = [
            collision_model,
            object_scene,
            config_path,
            ROOT / "src/update_current_semantic_objects.py",
            ROOT / "rollout/sam_segmentation.py",
            ROOT / "rollout/scene_registration.py",
            _resolve(refresh_config["current_capture"]),
            _resolve(refresh_config["reference_report"]),
            _resolve(refresh_config["reference_capture"]),
        ]
        for record in refresh_config.get("accepted_masks", {}).values():
            refresh_inputs.append(_resolve(record["path"]))
        stages.append(
            _run_stage(
                name="current_semantic_objects",
                command=refresh_command,
                inputs=refresh_inputs,
                outputs=[
                    current_model,
                    current_object_scene,
                    current_report_path,
                    current_overlay,
                ],
                output_root=output,
                force=force,
            )
        )
        current_refresh_report = json.loads(current_report_path.read_text())
        if not (
            current_refresh_report.get("accepted")
            and current_refresh_report["model_validation"]["accepted"]
            and current_refresh_report.get("commands_sent") is False
        ):
            raise ValueError("current semantic object refresh gates failed")
        replay_model = current_model
        replay_object_scene = current_object_scene

    trajectory_dir = output / "trajectory"
    trajectory_path = trajectory_dir / "trajectory.json"
    display_only_on_collision = bool(
        config.get("recorded_replay_display_only_on_collision", False)
    )
    trajectory_command = [
        python,
        str(ROOT / "src/build_recorded_trajectory_replay.py"),
        "--config",
        str(replay_config),
        "--model",
        str(replay_model),
        "--object-scene",
        str(replay_object_scene),
        "--output",
        str(trajectory_path),
    ]
    if display_only_on_collision:
        trajectory_command.append("--display-only-on-collision")
    stages.append(
        _run_stage(
            name="recorded_trajectory",
            command=trajectory_command,
            inputs=[
                replay_config,
                replay_model,
                replay_object_scene,
                ROOT / "rollout/recorded_trajectory_replay.py",
                ROOT / "src/build_recorded_trajectory_replay.py",
            ],
            outputs=[trajectory_path],
            output_root=output,
            force=force,
        )
    )
    trajectory = json.loads(trajectory_path.read_text())
    if not (
        trajectory["validation"]["all_keyframes_exact"]
        and trajectory.get("commands_sent") is False
        and (
            trajectory["validation"]["moving_arm_path_clear"]
            or display_only_on_collision
        )
    ):
        raise ValueError("recorded trajectory gates failed")

    render_config = config["render"]
    render_dir = output / "render"
    render_report = render_dir / "render_report.json"
    render_command = [
        python,
        str(ROOT / "src/render_recorded_trajectory_replay.py"),
        "--model",
        str(replay_model),
        "--trajectory",
        str(trajectory_path),
        "--output-dir",
        str(render_dir),
        "--width",
        str(render_config.get("width", 640)),
        "--height",
        str(render_config.get("height", 480)),
        "--fps",
        str(render_config.get("fps", 15)),
        "--camera-azimuth-deg",
        str(render_config.get("camera_azimuth_deg", 90.0)),
        "--camera-elevation-deg",
        str(render_config.get("camera_elevation_deg", -28.0)),
        "--camera-distance-m",
        str(render_config.get("camera_distance_m", 1.65)),
    ]
    if display_only_on_collision:
        render_command.append("--allow-display-only-collision")
    stages.append(
        _run_stage(
            name="render_replay",
            command=render_command,
            inputs=[
                replay_model,
                trajectory_path,
                ROOT / "src/render_recorded_trajectory_replay.py",
            ],
            outputs=[
                render_report,
                render_dir / "recorded_replay.mp4",
                render_dir / "recorded_replay_start.png",
                render_dir / "recorded_replay_final.png",
            ],
            output_root=output,
            force=force,
            environment={"MUJOCO_GL": "egl"},
        )
    )

    views_dir = output / "views"
    full_home = views_dir / "mujoco_home_full.html"
    home_command = [
        python,
        str(ROOT / "src/render_mujoco_mobile.py"),
        "--model",
        str(replay_model),
        "--output",
        str(full_home),
        "--keyframe",
        "home",
        "--camera-eye",
        "-1.4",
        "1.6",
        "1.0",
    ]
    stages.append(
        _run_stage(
            name="render_home_3d",
            command=home_command,
            inputs=[replay_model, ROOT / "src/render_mujoco_mobile.py"],
            outputs=[full_home],
            output_root=output,
            force=force,
        )
    )

    mobile_config = config["mobile"]
    mobile_dir = output / "mobile"
    mobile_sources = {
        "mujoco_home": full_home,
        "semantic_3d": base_semantic_view,
        "source_esdf_scene": base_esdf_view,
    }
    for name, source in mobile_sources.items():
        mobile_output = mobile_dir / f"{name}.html"
        mobile_report = mobile_dir / f"{name}.report.json"
        command = [
            python,
            str(ROOT / "src/optimize_plotly_mobile.py"),
            "--source",
            str(source),
            "--output",
            str(mobile_output),
            "--maximum-faces",
            str(mobile_config.get("maximum_faces", 1200)),
            "--maximum-points",
            str(mobile_config.get("maximum_points", 2500)),
            "--report",
            str(mobile_report),
        ]
        stages.append(
            _run_stage(
                name=f"mobile_{name}",
                command=command,
                inputs=[source, ROOT / "src/optimize_plotly_mobile.py"],
                outputs=[mobile_output, mobile_report, mobile_dir / "plotly.min.js"],
                output_root=output,
                force=force,
            )
        )

    object_data = json.loads(replay_object_scene.read_text())
    object_record = object_data["objects"][0]
    latest_center = np.asarray(
        object_record["pose_scene"],
        dtype=float,
    )[:3, 3]
    successful = np.asarray(
        object_data["source"]["episode_targets_scene_xyz_m"][
            "successful_grasp_before_lift"
        ],
        dtype=float,
    )
    successful[2] = latest_center[2]
    summary = {
        "latest_target_center_scene_m": latest_center.tolist(),
        "recorded_success_target_center_scene_m": successful.tolist(),
        "recorded_to_latest_shift_m": (latest_center - successful).tolist(),
        "target_train_rms_m": target["fit"]["train_rms_m"],
        "target_holdout_max_error_m": target["fit"][
            "maximum_holdout_point_error_m"
        ],
        "all_measured_keyframes_exact": trajectory["validation"][
            "all_keyframes_exact"
        ],
        "moving_arm_path_clear": trajectory["validation"][
            "moving_arm_path_clear"
        ],
        "global_scene_home_clear": trajectory["validation"][
            "global_scene_home_clear"
        ],
        "trajectory_duration_s": trajectory["duration_s"],
        "trajectory_joint_path_length_rad": sum(
            item["joint_path_length_rad"] for item in trajectory["planners"]
        ),
        "continuous_joint_log_available": False,
        "commands_sent": False,
        "current_object_refresh_accepted": (
            None
            if current_refresh_report is None
            else current_refresh_report["accepted"]
        ),
    }
    report = {
        "schema": SCHEMA,
        "status": "offline_replay_ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands_sent": False,
        "hardware_motion_authorized": False,
        "config": str(config_path),
        "output": str(output),
        "stages": stages,
        "summary": summary,
        "physical_arm_identity": arm_identity,
        "limitations": [
            (
                "The exact stopped keyframes are measured; motion between "
                "them is reconstructed because no continuous joint log exists."
            ),
            *(
                []
                if summary["global_scene_home_clear"]
                else [
                    (
                        "Static physical-left contacts remain; only the moving "
                        "physical-right path passed."
                    )
                ]
            ),
        ],
        "duration_s": time.monotonic() - started,
    }
    report_path = output / "pipeline_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_mobile_index(output, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "src/configs/pasteur_offline_replay_20260730.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/runs/pasteur/offline_replay_20260730_v1",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--sam-endpoint",
        help="override current-object SAM endpoint from the replay profile",
    )
    args = parser.parse_args(argv)
    report = run(
        args.config,
        args.output_dir.resolve(),
        force=args.force,
        sam_endpoint=args.sam_endpoint,
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
