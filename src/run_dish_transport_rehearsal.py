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
from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess
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
from rollout.camera import CameraFeedManager, USBWristCameraFeedManager
from rollout.dish_transport_rehearsal import (
    CartesianAirTransportStreamer,
    StoppedObserverPlan,
    TransportPlan,
    build_stopped_observer_plan,
    build_transport_plan,
    split_checkpoint_chunks,
)
from rollout.gripper_level import JawLevelReference, assess_jaw_level
from rollout.wrist_observer_tracking import (
    blue_components,
    compare_side_view_shape,
    describe_blue_component,
)
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


def _validate_scene_layout_contract(config: dict) -> None:
    """Reject a mirrored/stale display scene before any route is compiled."""

    report_path = config.get("semantic_scene_report")
    contract = config.get("scene_layout_contract")
    if not report_path or not contract:
        return
    report = _load_json(report_path)
    objects = {
        value["semantic_name"]: np.asarray(
            value["geometry"]["center_xyz_m"], dtype=float
        )
        for value in report["objects"]
    }
    bases = {
        side.split("/")[0]: np.asarray(xyz, dtype=float)
        for side, xyz in report["robot_placement"]["base_xyz_level_m"].items()
    }
    for station, expected_side in contract["station_in_front_of_arm"].items():
        if station not in objects or expected_side not in bases:
            raise ValueError(f"scene contract is missing {station}/{expected_side}")
        other_side = "right" if expected_side == "left" else "left"
        expected_distance = float(np.linalg.norm(objects[station][:2] - bases[expected_side][:2]))
        other_distance = float(np.linalg.norm(objects[station][:2] - bases[other_side][:2]))
        if expected_distance >= other_distance:
            raise ValueError(
                f"scene is mirrored/stale: {station} is not in front of "
                f"physical {expected_side} arm"
            )
        if objects[station][1] <= bases[expected_side][1]:
            raise ValueError(
                f"scene is reversed: {station} is not forward of physical "
                f"{expected_side} arm"
            )


def compile_plans(
    config: dict, *, selected_names: set[str] | None = None
) -> list[TransportPlan]:
    _validate_scene_layout_contract(config)
    planning = config["planning"]
    geometry = config["geometry"]
    demo_directory = _resolve(config["demonstration_directory"])
    level = _level_reference(config["level_config"])
    plans = []
    for segment in config["segments"]:
        if selected_names and segment["name"] not in selected_names:
            continue
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
                minimum_dish_clearance_m=float(
                    planning["minimum_dish_clearance_m"]
                ),
                low_route_search_step_m=float(
                    planning["low_route_search_step_m"]
                ),
                maximum_low_route_lift_m=float(
                    planning["maximum_low_route_lift_m"]
                ),
                maximum_ik_joint_step_rad=float(
                    planning["maximum_ik_joint_step_rad"]
                ),
            )
        )
    return plans


def compile_observer_plans(
    config: dict, plans: list[TransportPlan]
) -> dict[str, StoppedObserverPlan]:
    observer = config.get("observer", {})
    if not observer.get("enabled", False):
        return {}
    reference = _load_json(observer["reference_config"])
    safe_waypoint = (
        _load_json(observer["safe_waypoint_config"])["ee_pose_wxyz_xyz"]
        if observer.get("safe_waypoint_config")
        else None
    )
    planning = config["planning"]
    geometry = config["geometry"]
    result = {}
    for plan in plans:
        if plan.physical_arm != "right":
            continue
        result[plan.name] = build_stopped_observer_plan(
            plan,
            production_model=_resolve(config["production_model"]),
            planning_model=_resolve(config["planning_model"]),
            reference_observer_pose_wxyz_xyz=reference[
                "observer_pose_wxyz_xyz"
            ],
            reference_carrier_pose_wxyz_xyz=reference[
                "observed_right_pose_wxyz_xyz"
            ],
            safe_waypoint_pose_wxyz_xyz=safe_waypoint,
            maximum_cartesian_step_m=float(
                planning["maximum_cartesian_step_m"]
            ),
            maximum_ik_joint_step_rad=float(
                planning["maximum_ik_joint_step_rad"]
            ),
            dish_radius_m=float(geometry["dish_diameter_m"]) / 2.0,
            dish_thickness_m=float(geometry["dish_thickness_m"]),
            dish_center_offset_ee_m=geometry["dish_center_offset_ee_m"],
            ignored_environment_bodies=geometry.get(
                "ignored_absent_scene_bodies", []
            ),
            minimum_dish_clearance_m=float(
                planning["minimum_dish_clearance_m"]
            ),
        )
    return result


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


class SynchronizedCameraSet:
    """Keep head and both wrists alive and return one timestamp-matched capture."""

    def __init__(self, head_name: str, left_name: str, right_name: str):
        camera_map = load_camera_map()
        names = (head_name, left_name, right_name)
        if len(set(names)) != 3:
            raise ValueError("head/left/right camera names must be distinct")
        self.events = {name: threading.Event() for name in names}
        self.managers = {
            head_name: CameraFeedManager(
                self.events[head_name], display=False, head_stream=False
            ),
            **{
                name: USBWristCameraFeedManager(
                self.events[name],
                device_index=int(camera_map[name]),
                label=f"dish rehearsal {name}",
            )
                for name in (left_name, right_name)
            },
        }
        self.head_name = head_name
        self.left_name = left_name
        self.right_name = right_name

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
                rgb, timestamp, depth = manager.get_latest_frame()
                if rgb is not None and timestamp is not None and timestamp >= started:
                    latest[name] = (rgb, float(timestamp), depth)
            if len(latest) == 3:
                timestamps = [latest[name][1] for name in self.managers]
                skew = max(timestamps) - min(timestamps)
                if skew <= maximum_skew_s:
                    return {
                        "head_bgr": self._bgr(latest[self.head_name][0]),
                        "left_bgr": self._bgr(latest[self.left_name][0]),
                        "right_bgr": self._bgr(latest[self.right_name][0]),
                        "head_depth_m": latest[self.head_name][2],
                        "head_timestamp_s": latest[self.head_name][1],
                        "left_timestamp_s": latest[self.left_name][1],
                        "right_timestamp_s": latest[self.right_name][1],
                        "timestamp_skew_s": skew,
                    }
            time.sleep(0.02)
        raise RuntimeError(
            "head/left/right did not yield a fresh synchronized capture"
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    head = _annotate(capture["head_bgr"], ["HEAD", *lines])
    left = _annotate(capture["left_bgr"], ["LEFT OBSERVER", *lines])
    right = _annotate(capture["right_bgr"], ["RIGHT CARRIER", *lines])
    head_path = directory / f"{prefix}_head.jpg"
    left_path = directory / f"{prefix}_left.jpg"
    right_path = directory / f"{prefix}_right.jpg"
    depth_path = directory / f"{prefix}_head_depth.npy"
    cv2.imwrite(str(head_path), head)
    cv2.imwrite(str(left_path), left)
    cv2.imwrite(str(right_path), right)
    if capture.get("head_depth_m") is not None:
        np.save(depth_path, np.asarray(capture["head_depth_m"], dtype=np.float32))
    target_height = max(head.shape[0], left.shape[0], right.shape[0])

    def fit(image):
        scale = target_height / image.shape[0]
        return cv2.resize(image, (int(round(image.shape[1] * scale)), target_height))

    montage = np.hstack((fit(head), fit(left), fit(right)))
    montage_path = directory / f"{prefix}_head_left_right.jpg"
    cv2.imwrite(str(montage_path), montage)
    return head, left, right, {
        "head_image": str(head_path),
        "left_image": str(left_path),
        "right_image": str(right_path),
        "head_depth": str(depth_path) if depth_path.exists() else None,
        "montage_image": str(montage_path),
    }


def _assess_left_side_view(left_bgr: np.ndarray, reference: dict) -> dict:
    """Select positive blue jaw evidence and reject dark shadows."""

    components = blue_components(left_bgr)
    expected = reference["shape_reference"]
    from rollout.wrist_observer_tracking import BlueShapeDescriptor

    expected_descriptor = BlueShapeDescriptor(
        principal_axis_deg=float(expected["principal_axis_deg"]),
        elongation=float(expected["elongation"]),
        fill_fraction=float(expected["fill_fraction"]),
        area_fraction=float(expected["area_fraction"]),
        centroid_normalized_xy=tuple(expected["centroid_normalized_xy"]),
        bbox_normalized_xywh=tuple(expected["bbox_normalized_xywh"]),
    )
    candidates = []
    for component in components:
        descriptor = describe_blue_component(left_bgr, component)
        match = compare_side_view_shape(descriptor, expected_descriptor)
        score = (
            match.principal_axis_difference_deg / 5.0
            + match.elongation_log_ratio / 0.35
            + match.fill_fraction_difference / 0.20
        )
        candidates.append(
            {
                "accepted": match.accepted,
                "score": float(score),
                "bbox_xywh": list(component.bbox_xywh),
                "area_fraction": descriptor.area_fraction,
                "principal_axis_difference_deg": (
                    match.principal_axis_difference_deg
                ),
                "elongation_log_ratio": match.elongation_log_ratio,
                "fill_fraction_difference": match.fill_fraction_difference,
                "reasons": list(match.reasons),
            }
        )
    if not candidates:
        return {
            "accepted": False,
            "positive_blue_component_count": 0,
            "reason": "no_positive_blue_jaw_evidence",
        }
    selected = min(candidates, key=lambda value: value["score"])
    return {
        "accepted": bool(selected["accepted"]),
        "positive_blue_component_count": len(candidates),
        "selected": selected,
        "policy": "positive_blue_shape_only_dark_shadow_never_target",
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
    observer_plans: dict[str, StoppedObserverPlan],
    run_directory: Path,
    *,
    allow_audit_warnings: bool,
) -> dict:
    rejected = [plan.name for plan in plans if not _plan_execution_ready(plan)]
    observer_rejected = [
        f"{name}/left_observer"
        for name, observer in observer_plans.items()
        if not observer.audit.get("accepted", False)
    ]
    if observer_rejected:
        raise RuntimeError(
            "coordinated two-arm collision audit blocks execution for "
            f"{observer_rejected}; this gate has no warning override"
        )
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
        "observer_plans": {
            name: observer.to_dict() for name, observer in observer_plans.items()
        },
        "checkpoints": [],
        "status": "running",
    }
    _write_json(run_directory / "run.json", report)
    try:
        if execution["home_before_start"]:
            report["initial_home"] = _home_both(rpc, execution)
        level = _level_reference(config["level_config"])
        checkpoint_level = replace(
            level,
            maximum_checkpoint_tilt_deg=float(
                execution["maximum_checkpoint_tilt_deg"]
            ),
        )
        side_view_reference = (
            _load_json(config["observer"]["reference_config"])
            if config.get("observer", {}).get("enabled", False)
            else None
        )
        camera_names = config["camera_names"]
        for segment_number, plan in enumerate(plans, start=1):
            chunks = split_checkpoint_chunks(plan)
            observer_plan = observer_plans.get(plan.name)
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
            observer_streamer = (
                CartesianAirTransportStreamer(
                    rpc,
                    "left",
                    torque_limit_nm=execution["torque_limits_nm"]["left"],
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
                if observer_plan is not None
                else None
            )
            with SynchronizedCameraSet(
                camera_names["head"], camera_names["left"], camera_names["right"]
            ) as cameras:
                checkpoint_count = len(plan.checkpoint_indices)
                for checkpoint_number in range(checkpoint_count):
                    checkpoint = plan.checkpoint(checkpoint_number)
                    motion = streamer.execute(
                        chunks[checkpoint_number],
                        speed_m_s=float(execution["speed_m_s"]),
                        gripper_open_ratio=float(execution["air_gripper_open_ratio"]),
                        stage=f"{plan.name}/{checkpoint.name}",
                    )
                    observer_motion = None
                    if observer_plan is not None:
                        observer_motion = observer_streamer.execute(
                            observer_plan.transition_pose_paths[checkpoint_number],
                            speed_m_s=float(execution["observer_speed_m_s"]),
                            gripper_open_ratio=float(
                                execution["air_gripper_open_ratio"]
                            ),
                            stage=f"{plan.name}/{checkpoint.name}/left_observer",
                        )
                    refinement_config = execution["jaw_level_refinement"]
                    refinement = None
                    if (
                        checkpoint_number == 0
                        and refinement_config.get("enabled", True)
                    ):
                        try:
                            refinement_level = replace(
                                level,
                                maximum_checkpoint_tilt_deg=float(
                                    refinement_config["target_maximum_tilt_deg"]
                                ),
                                maximum_tip_height_difference_m=float(
                                    refinement_config[
                                        "target_maximum_tip_height_difference_m"
                                    ]
                                ),
                            )
                            refinement = streamer.refine_jaw_level(
                                refinement_level,
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
                    jaw = assess_jaw_level(
                        measured, checkpoint_level, planned=False
                    )
                    capture = cameras.capture(
                        timeout_s=float(execution["camera_timeout_s"]),
                        maximum_skew_s=float(
                            execution["maximum_camera_timestamp_skew_s"]
                        ),
                    )
                    side_view = (
                        _assess_left_side_view(
                            capture["left_bgr"], side_view_reference
                        )
                        if observer_plan is not None
                        else {"accepted": True, "policy": "observer_disabled"}
                    )
                    prefix = f"{segment_number:02d}_{checkpoint_number + 1:02d}_{checkpoint.name}"
                    head, left, right, images = _save_checkpoint_images(
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
                        "observer_motion": observer_motion,
                        "jaw_level_refinement": refinement,
                        "measured_pose_wxyz_xyz": measured.tolist(),
                        "jaw_level": jaw.to_dict(),
                        "left_side_view": side_view,
                        "camera": {
                            key: value
                            for key, value in capture.items()
                            if not key.endswith("_bgr") and key != "head_depth_m"
                        },
                        **images,
                    }
                    report["checkpoints"].append(record)
                    _write_json(run_directory / "run.json", report)
                    store.publish(
                        segment=f"{segment_number}/{len(plans)} {plan.source} → {plan.destination}",
                        checkpoint=(
                            f"{checkpoint_number + 1}/{checkpoint_count} "
                            f"{checkpoint.name}"
                        ),
                        physical_arm=plan.physical_arm,
                        metrics={
                            "jaw_level": jaw.to_dict(),
                            "left_side_view": side_view,
                            "camera_timestamp_skew_s": capture["timestamp_skew_s"],
                            "preflight_audit": bool(
                                _plan_execution_ready(plan)
                                and (
                                    observer_plan is None
                                    or observer_plan.audit.get("accepted", False)
                                )
                            ),
                            "torque_warning_count": motion["torque_warning_count"],
                        },
                        head_bgr=head,
                        left_bgr=left,
                        right_bgr=right,
                        continue_allowed=bool(
                            jaw.accepted and side_view["accepted"]
                        ),
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
                # Move the observer home while the carrier holds at arrival;
                # only then return the carrier, preserving one-arm-at-a-time.
                if observer_plan is not None:
                    report.setdefault("observer_returns", []).append(
                        observer_streamer.execute(
                            observer_plan.return_pose_path,
                            speed_m_s=float(execution["observer_speed_m_s"]),
                            gripper_open_ratio=float(
                                execution["air_gripper_open_ratio"]
                            ),
                            stage=f"{plan.name}/left_observer_return_home",
                        )
                    )
                    rpc.home_left_arm()
                    _wait_home(
                        rpc,
                        "left",
                        float(execution["home_joint_tolerance_rad"]),
                        float(execution["home_timeout_s"]),
                    )
                # Each air segment is independent; return along the audited
                # path, then restore the physical canonical home before the
                # other arm/next demonstration is selected.
                report.setdefault("returns", []).append(
                    streamer.execute(
                        chunks[-1],
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
    parser.add_argument(
        "--carrier-only",
        action="store_true",
        help=(
            "keep the observer arm at home and execute only the audited "
            "carrier path (air rehearsal only)"
        ),
    )
    parser.add_argument("--allow-audit-warnings", action="store_true")
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="execute only this named segment (repeatable); planning still audits all",
    )
    args = parser.parse_args(argv)
    config = _load_json(args.config)
    known_segments = {segment["name"] for segment in config["segments"]}
    missing = sorted(set(args.segment) - known_segments)
    if missing:
        parser.error(f"unknown --segment values: {missing}")
    plans = compile_plans(
        config, selected_names=set(args.segment) if args.segment else None
    )
    observer_plans = (
        {} if args.carrier_only else compile_observer_plans(config, plans)
    )
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
        "carrier_only": bool(args.carrier_only),
        "plans": [plan.to_dict() for plan in plans],
        "observer_plans": {
            name: observer.to_dict() for name, observer in observer_plans.items()
        },
    }
    _write_json(output / "plan.json", manifest)
    _write_plan_preview(output / "plan_preview.html", plans)
    video_path = output / "plan_preview.mp4"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/render_dish_transport_plan.py"),
            "--plan",
            str(output / "plan.json"),
            "--output",
            str(video_path),
        ],
        check=True,
    )
    print(f"plan: {output / 'plan.json'}")
    print(f"preview: {output / 'plan_preview.html'}")
    print(f"video: {video_path}")
    for plan in plans:
        observer = observer_plans.get(plan.name)
        ready = bool(
            _plan_execution_ready(plan)
            and (observer is None or observer.audit.get("accepted", False))
        )
        print(
            f"{plan.name}: arm={plan.physical_arm}, medoid={Path(plan.medoid_hdf5).name}, "
            f"preflight={'OK' if ready else 'BLOCKED'}"
        )
    if args.execute:
        report = execute_plans(
            config,
            plans,
            observer_plans,
            output,
            allow_audit_warnings=bool(args.allow_audit_warnings),
        )
        print(f"run status: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
