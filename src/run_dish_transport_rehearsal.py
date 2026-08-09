#!/usr/bin/env python3
"""Plan and optionally execute staged horizontal dish air transports.

Default operation is read-only: it compiles demonstrations, solves IK, audits
the robot plus a virtual dish in the calibrated MuJoCo scene, and writes a
manifest.  ``--execute`` uses the proven 30 Hz Cartesian teleop RPC path.  It
stops after source lift, route midpoint, and arrival hover, then waits for a
phone decision backed by fresh head and active-wrist photographs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.arm.home import physical_home_q
from robot.camera_id import load_camera_map
from robot.rpc import RPCClient
from rollout.camera import USBWristCameraFeedManager
from rollout.dish_transport_rehearsal import (
    CartesianAirTransportStreamer,
    TransportPlan,
    build_transport_plan,
    split_checkpoint_chunks,
)
from rollout.gripper_level import JawLevelReference, assess_jaw_level
from src.dish_transport_rehearsal_ui import CheckpointApprovalStore, make_server


DEFAULT_CONFIG = ROOT / "src/configs/pasteur_dish_transport_rehearsal.json"


def _plan_execution_ready(plan: TransportPlan) -> bool:
    audit = plan.collision_audit
    return bool(
        audit["accepted"]
        and audit.get("ik_proxy_jaw_level_accepted", False)
        and audit.get("ik_branch_continuity_accepted", False)
    )


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _load_json(path: str | Path) -> dict:
    with _resolve(path).open() as stream:
        return json.load(stream)


def _level_reference(path: str | Path) -> JawLevelReference:
    value = _load_json(path)
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


def compile_plans(config: dict) -> list[TransportPlan]:
    planning = config["planning"]
    geometry = config["geometry"]
    demo_directory = _resolve(config["demonstration_directory"])
    level = _level_reference(config["level_config"])
    plans = []
    for segment in config["segments"]:
        demonstrations = sorted(demo_directory.glob(segment["demonstration_glob"]))
        plans.append(
            build_transport_plan(
                name=segment["name"],
                source=segment["source"],
                destination=segment["destination"],
                physical_arm=segment["physical_arm"],
                demonstration_paths=demonstrations,
                production_model=_resolve(config["production_model"]),
                planning_model=_resolve(config["planning_model"]),
                level_reference=level,
                source_lift_m=float(planning["source_lift_m"]),
                arrival_hover_m=float(planning["arrival_hover_m"]),
                route_samples=int(planning["route_samples"]),
                maximum_cartesian_step_m=float(
                    planning["maximum_cartesian_step_m"]
                ),
                dish_radius_m=float(geometry["dish_diameter_m"]) / 2.0,
                dish_thickness_m=float(geometry["dish_thickness_m"]),
                dish_center_offset_ee_m=geometry["dish_center_offset_ee_m"],
                ignored_environment_bodies=geometry.get(
                    "ignored_absent_scene_bodies", []
                ),
                require_collision_free=bool(planning["require_collision_free"]),
            )
        )
    return plans


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n"
    )


def _write_plan_preview(path: Path, plans: list[TransportPlan]) -> None:
    """Write a dependency-free, phone-readable trajectory projection."""

    colors = {"right": "#18a6db", "left": "#ef70d5"}
    width, height = 1000, 760
    all_xyz = np.vstack([plan.poses_wxyz_xyz[:, 4:] for plan in plans])
    x_min, y_min = np.min(all_xyz[:, :2], axis=0) - 0.03
    x_max, y_max = np.max(all_xyz[:, :2], axis=0) + 0.03

    def point(xyz):
        x = 70 + (xyz[0] - x_min) / max(x_max - x_min, 1e-6) * (width - 140)
        y = height - 70 - (xyz[1] - y_min) / max(y_max - y_min, 1e-6) * (height - 140)
        return x, y

    paths = []
    marks = []
    legend = []
    for plan_index, plan in enumerate(plans):
        points = [point(xyz) for xyz in plan.poses_wxyz_xyz[:, 4:]]
        paths.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" '
            f'fill="none" stroke="{colors[plan.physical_arm]}" stroke-width="5"/>'
        )
        for checkpoint_number, checkpoint_index in enumerate(plan.checkpoint_indices):
            x, y = points[checkpoint_index]
            marks.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#ffe04a"/>'
                f'<text x="{x + 12:.1f}" y="{y - 8:.1f}" fill="white" font-size="18">'
                f'{plan_index + 1}.{checkpoint_number + 1}</text>'
            )
        legend.append(
            f'<div><span style="color:{colors[plan.physical_arm]}">●</span> '
            f'{plan_index + 1}. {plan.source} → {plan.destination} ({plan.physical_arm}) '
            f'/ 事前監査: {"OK" if _plan_execution_ready(plan) else "要確認"}</div>'
        )
    html = f"""<!doctype html><html lang=ja><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'><title>搬送軌道</title>
<body style='background:#111;color:#eee;font-family:-apple-system,sans-serif'>
<h2>水平エアー搬送計画（上面XY）</h2>{''.join(legend)}
<svg viewBox='0 0 {width} {height}' style='width:100%;background:#252933;border-radius:12px'>
{''.join(paths)}{''.join(marks)}</svg><p>黄色点で停止し、head＋使用手カメラを確認します。</p></body></html>"""
    path.write_text(html)


class SynchronizedCameraPair:
    """Keep head and active wrist alive together and return timestamp-matched RGB."""

    def __init__(self, head_name: str, wrist_name: str):
        camera_map = load_camera_map()
        self.events = {name: threading.Event() for name in (head_name, wrist_name)}
        self.managers = {
            name: USBWristCameraFeedManager(
                self.events[name],
                device_index=int(camera_map[name]),
                label=f"dish rehearsal {name}",
            )
            for name in (head_name, wrist_name)
        }
        self.head_name = head_name
        self.wrist_name = wrist_name

    def __enter__(self):
        for manager in self.managers.values():
            manager.start()
        return self

    def __exit__(self, _type, _value, _traceback):
        errors = []
        for manager in reversed(tuple(self.managers.values())):
            try:
                manager.stop()
            except BaseException as error:
                errors.append(error)
        if errors and _value is None:
            raise errors[0]

    @staticmethod
    def _bgr(rgb: np.ndarray) -> np.ndarray:
        rotated = cv2.rotate(np.asarray(rgb), cv2.ROTATE_90_CLOCKWISE)
        return cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)

    def capture(self, *, timeout_s: float, maximum_skew_s: float):
        started = time.time()
        deadline = time.monotonic() + float(timeout_s)
        latest = {}
        while time.monotonic() < deadline:
            for name, manager in self.managers.items():
                rgb, timestamp, _ = manager.get_latest_frame()
                if rgb is not None and timestamp is not None and timestamp >= started:
                    latest[name] = (rgb, float(timestamp))
            if len(latest) == 2:
                skew = abs(latest[self.head_name][1] - latest[self.wrist_name][1])
                if skew <= maximum_skew_s:
                    return {
                        "head_bgr": self._bgr(latest[self.head_name][0]),
                        "wrist_bgr": self._bgr(latest[self.wrist_name][0]),
                        "head_timestamp_s": latest[self.head_name][1],
                        "wrist_timestamp_s": latest[self.wrist_name][1],
                        "timestamp_skew_s": skew,
                    }
            time.sleep(0.02)
        raise RuntimeError(
            f"head/{self.wrist_name} did not yield a fresh synchronized pair"
        )


def _annotate(image: np.ndarray, lines: list[str]) -> np.ndarray:
    result = np.asarray(image).copy()
    y = 34
    for line in lines:
        cv2.putText(
            result,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return result


def _save_checkpoint_images(
    directory: Path,
    prefix: str,
    capture: dict,
    *,
    lines: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    head = _annotate(capture["head_bgr"], ["HEAD", *lines])
    wrist = _annotate(capture["wrist_bgr"], ["ACTIVE WRIST", *lines])
    head_path = directory / f"{prefix}_head.jpg"
    wrist_path = directory / f"{prefix}_wrist.jpg"
    cv2.imwrite(str(head_path), head)
    cv2.imwrite(str(wrist_path), wrist)
    target_height = max(head.shape[0], wrist.shape[0])

    def fit(image):
        scale = target_height / image.shape[0]
        return cv2.resize(image, (int(round(image.shape[1] * scale)), target_height))

    montage = np.hstack((fit(head), fit(wrist)))
    montage_path = directory / f"{prefix}_head_wrist.jpg"
    cv2.imwrite(str(montage_path), montage)
    return head, wrist, {
        "head_image": str(head_path),
        "wrist_image": str(wrist_path),
        "montage_image": str(montage_path),
    }


def _wait_home(rpc, side: str, tolerance_rad: float, timeout_s: float) -> dict:
    getter = getattr(rpc, f"get_{side}_joint_positions")
    deadline = time.monotonic() + float(timeout_s)
    maximum_error = float("inf")
    while time.monotonic() < deadline:
        measured = np.asarray(getter(), dtype=float)
        maximum_error = float(np.max(np.abs(measured - physical_home_q(side))))
        if maximum_error <= tolerance_rad:
            return {"accepted": True, "maximum_joint_error_rad": maximum_error}
        time.sleep(0.25)
    raise RuntimeError(
        f"{side} arm did not reach canonical home; error={maximum_error:.3f}rad"
    )


def _home_both(rpc, execution: dict) -> dict:
    rpc.home_left_arm()
    rpc.home_right_arm()
    return {
        side: _wait_home(
            rpc,
            side,
            float(execution["home_joint_tolerance_rad"]),
            float(execution["home_timeout_s"]),
        )
        for side in ("left", "right")
    }


def _ui_url(port: int) -> str:
    host = socket.gethostname()
    try:
        addresses = socket.gethostbyname_ex(host)[2]
        tailscale = next(address for address in addresses if address.startswith("100."))
        return f"http://{tailscale}:{port}/"
    except (OSError, StopIteration):
        return f"http://{host}:{port}/"


def execute_plans(
    config: dict,
    plans: list[TransportPlan],
    run_directory: Path,
    *,
    allow_audit_warnings: bool,
) -> dict:
    rejected = [plan.name for plan in plans if not _plan_execution_ready(plan)]
    if rejected and not allow_audit_warnings:
        raise RuntimeError(
            "MuJoCo collision audit warning blocks execution for "
            f"{rejected}; inspect the plan, then pass --allow-audit-warnings explicitly"
        )
    execution = config["execution"]
    rpc_config = config["rpc"]
    rpc = RPCClient(
        rpc_config["host"],
        int(rpc_config["port"]),
        timeout_ms=int(rpc_config["timeout_ms"]),
    )
    store = CheckpointApprovalStore()
    server = make_server(
        store,
        host=config["approval_ui"]["host"],
        port=int(config["approval_ui"]["port"]),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"checkpoint UI: {_ui_url(int(config['approval_ui']['port']))}", flush=True)
    report = {
        "schema": "piper_robot.dish_transport_air_rehearsal_run/v1",
        "started_at_s": time.time(),
        "allow_audit_warnings": bool(allow_audit_warnings),
        "plans": [plan.to_dict() for plan in plans],
        "checkpoints": [],
        "status": "running",
    }
    _write_json(run_directory / "run.json", report)
    try:
        if execution["home_before_start"]:
            report["initial_home"] = _home_both(rpc, execution)
        level = _level_reference(config["level_config"])
        camera_names = config["camera_names"]
        for segment_number, plan in enumerate(plans, start=1):
            chunks = split_checkpoint_chunks(plan)
            streamer = CartesianAirTransportStreamer(
                rpc,
                plan.physical_arm,
                torque_limit_nm=execution["torque_limits_nm"][plan.physical_arm],
                torque_samples=int(execution["torque_consecutive_samples"]),
                control_hz=float(execution["control_hz"]),
                preview_time_s=float(execution["preview_time_s"]),
                tracking_interval=int(execution["tracking_check_interval"]),
                maximum_tracking_position_error_m=float(
                    execution["maximum_tracking_position_error_m"]
                ),
                maximum_tracking_rotation_error_rad=float(
                    execution["maximum_tracking_rotation_error_rad"]
                ),
                final_settle_s=float(execution["endpoint_settle_s"]),
            )
            with SynchronizedCameraPair(
                camera_names["head"], camera_names[plan.physical_arm]
            ) as cameras:
                for checkpoint_number in range(3):
                    checkpoint = plan.checkpoint(checkpoint_number)
                    motion = streamer.execute(
                        chunks[checkpoint_number],
                        speed_m_s=float(execution["speed_m_s"]),
                        gripper_open_ratio=float(execution["air_gripper_open_ratio"]),
                        stage=f"{plan.name}/{checkpoint.name}",
                    )
                    refinement_config = execution["jaw_level_refinement"]
                    refinement = None
                    if refinement_config.get("enabled", True):
                        try:
                            refinement = streamer.refine_jaw_level(
                                level,
                                gripper_open_ratio=float(
                                    execution["air_gripper_open_ratio"]
                                ),
                                maximum_attempts=int(
                                    refinement_config["maximum_attempts"]
                                ),
                                maximum_correction_deg=float(
                                    refinement_config["maximum_correction_deg"]
                                ),
                                maximum_total_correction_deg=float(
                                    refinement_config["maximum_total_correction_deg"]
                                ),
                                correction_duration_s=float(
                                    refinement_config["correction_duration_s"]
                                ),
                                settle_s=float(refinement_config["settle_s"]),
                                maximum_xyz_drift_m=float(
                                    refinement_config["maximum_xyz_drift_m"]
                                ),
                                maximum_xyz_correction_per_attempt_m=float(
                                    refinement_config[
                                        "maximum_xyz_correction_per_attempt_m"
                                    ]
                                ),
                                maximum_xyz_command_offset_m=float(
                                    refinement_config[
                                        "maximum_xyz_command_offset_m"
                                    ]
                                ),
                                hard_xyz_drift_m=float(
                                    refinement_config["hard_xyz_drift_m"]
                                ),
                            )
                        except BaseException:
                            streamer.hold_measured()
                            raise
                    measured = streamer.measured_pose()
                    jaw = assess_jaw_level(measured, level, planned=False)
                    capture = cameras.capture(
                        timeout_s=float(execution["camera_timeout_s"]),
                        maximum_skew_s=float(
                            execution["maximum_camera_timestamp_skew_s"]
                        ),
                    )
                    prefix = f"{segment_number:02d}_{checkpoint_number + 1:02d}_{checkpoint.name}"
                    head, wrist, images = _save_checkpoint_images(
                        run_directory / "checkpoints",
                        prefix,
                        capture,
                        lines=[
                            f"{plan.source} -> {plan.destination}",
                            f"{checkpoint.name} / {plan.physical_arm}",
                        ],
                    )
                    record = {
                        "segment": plan.name,
                        "checkpoint": checkpoint.to_dict(),
                        "physical_arm": plan.physical_arm,
                        "motion": motion,
                        "jaw_level_refinement": refinement,
                        "measured_pose_wxyz_xyz": measured.tolist(),
                        "jaw_level": jaw.to_dict(),
                        "camera": {
                            key: value
                            for key, value in capture.items()
                            if not key.endswith("_bgr")
                        },
                        **images,
                    }
                    report["checkpoints"].append(record)
                    _write_json(run_directory / "run.json", report)
                    store.publish(
                        segment=f"{segment_number}/{len(plans)} {plan.source} → {plan.destination}",
                        checkpoint=f"{checkpoint_number + 1}/3 {checkpoint.name}",
                        physical_arm=plan.physical_arm,
                        metrics={
                            "jaw_level": jaw.to_dict(),
                            "camera_timestamp_skew_s": capture["timestamp_skew_s"],
                            "preflight_audit": _plan_execution_ready(plan),
                            "torque_warning_count": motion["torque_warning_count"],
                        },
                        head_bgr=head,
                        wrist_bgr=wrist,
                        continue_allowed=jaw.accepted,
                    )
                    decision = store.wait(
                        float(execution["operator_approval_timeout_s"])
                    )
                    record["operator_decision"] = decision
                    _write_json(run_directory / "run.json", report)
                    if decision != "continue":
                        report["status"] = decision
                        if decision == "abort_home":
                            report["abort_home"] = _home_both(rpc, execution)
                        return report
                # Each air segment is independent; return along the audited
                # path, then restore the physical canonical home before the
                # other arm/next demonstration is selected.
                report.setdefault("returns", []).append(
                    streamer.execute(
                        chunks[3],
                        speed_m_s=float(execution["speed_m_s"]),
                        gripper_open_ratio=float(execution["air_gripper_open_ratio"]),
                        stage=f"{plan.name}/return_home",
                    )
                )
            getattr(rpc, f"home_{plan.physical_arm}_arm")()
            _wait_home(
                rpc,
                plan.physical_arm,
                float(execution["home_joint_tolerance_rad"]),
                float(execution["home_timeout_s"]),
            )
        if execution["home_after_completion"]:
            report["final_home"] = _home_both(rpc, execution)
        report["status"] = "completed"
        return report
    except BaseException as error:
        report["status"] = "failed_hold_current"
        report["error"] = repr(error)
        raise
    finally:
        report["finished_at_s"] = time.time()
        _write_json(run_directory / "run.json", report)
        server.shutdown()
        server.server_close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-audit-warnings", action="store_true")
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="execute only this named segment (repeatable); planning still audits all",
    )
    args = parser.parse_args(argv)
    config = _load_json(args.config)
    plans = compile_plans(config)
    output = (
        _resolve(args.output)
        if args.output
        else ROOT
        / "data/runs/pasteur/dish_transport_rehearsal"
        / time.strftime("%Y%m%dT%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "piper_robot.dish_transport_air_rehearsal_manifest/v1",
        "config": str(_resolve(args.config)),
        "created_at_s": time.time(),
        "execution_requested": bool(args.execute),
        "plans": [plan.to_dict() for plan in plans],
    }
    _write_json(output / "plan.json", manifest)
    _write_plan_preview(output / "plan_preview.html", plans)
    print(f"plan: {output / 'plan.json'}")
    print(f"preview: {output / 'plan_preview.html'}")
    for plan in plans:
        print(
            f"{plan.name}: arm={plan.physical_arm}, medoid={Path(plan.medoid_hdf5).name}, "
            f"preflight={'OK' if _plan_execution_ready(plan) else 'WARNING'}"
        )
    if args.execute:
        selected = (
            [plan for plan in plans if plan.name in set(args.segment)]
            if args.segment
            else plans
        )
        missing = sorted(set(args.segment) - {plan.name for plan in selected})
        if missing:
            parser.error(f"unknown --segment values: {missing}")
        report = execute_plans(
            config,
            selected,
            output,
            allow_audit_warnings=bool(args.allow_audit_warnings),
        )
        print(f"run status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
