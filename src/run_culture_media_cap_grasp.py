#!/usr/bin/env python3
"""Home-seeded, adapter-driven grasp of a culture-media bottle cap.

The successful thin-object pipeline remains authoritative for trajectory
streaming, FK, and Cartesian corrections.  Only semantic target identity and
the cylindrical side-pinch contact geometry live here.  A Petri-lid demo pixel
goal is never loaded by this entrypoint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.arm.home import physical_home_q
from robot.rpc import RPCClient
from rollout.grasp_window import ToolImageFrame, detect_light_pad_tool_frame
from rollout.media_cap_target import (
    MediaCapTargetAdapter,
    detect_coloured_support_anchor,
    detect_media_cap,
    detect_open_jaw_center_head,
    fixed_target_in_jaw_segment,
)
from rollout.target_adapter import (
    render_target_and_contact_goal,
    stable_observation,
)
from rollout.teleop_trajectory_stream import ProductionRightFK
from rollout.teleop_trajectory_stream import sample_joint_knots
from rollout.rgbd_target_scene import (
    selected_head_rgbd,
    target_surface_in_scene,
    write_completed_vertical_object_scene,
)
from src.run_codexless_thin_object_grasp import (
    LiveCamera,
    _cartesian_move,
    _execute_direct_joint_samples,
    _joint_bounds,
    _move_between_hovers,
    _right_joint_path_contact_audit,
)


@dataclass(frozen=True)
class CapObservation:
    """Compatibility view retained for analysis tools from the first trial."""

    center_px: tuple[float, float]
    center_uv: tuple[float, float]
    component_pixels: int
    component_area_per_tool_scale_sq: float


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def load_task_profile(path: str | Path) -> tuple[dict, dict]:
    task = _load(path)
    if task.get("schema") != "piper_robot.adapter_grasp_task/v1":
        raise ValueError("unsupported adapter grasp task profile")
    base = _load(task["base_profile"])
    if task.get("target_adapter") != "culture_media_cap":
        raise ValueError("this entrypoint requires culture_media_cap adapter")
    if "canonical_hover_goal_uv" in task or "canonical_preclose_goal_uv" in task:
        raise ValueError("thin-object demo goals are forbidden in a cap task")
    return task, base


def observe_media_cap(image_bgr: np.ndarray, tool_frame: ToolImageFrame):
    """Backward-compatible one-frame observation using the target adapter."""

    observed = MediaCapTargetAdapter().observe(image_bgr, tool_frame)
    overlay = render_target_and_contact_goal(
        image_bgr, observed, tool_frame, observed.center_uv
    )
    compatibility = CapObservation(
        center_px=observed.center_px,
        center_uv=observed.center_uv,
        component_pixels=observed.component_pixels,
        component_area_per_tool_scale_sq=float(
            observed.component_pixels / tool_frame.scale_px**2
        ),
    )
    return compatibility, overlay, dict(observed.diagnostics)


def _home_gate(rpc, tolerance_rad: float) -> dict:
    measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    expected = physical_home_q("right")
    error = float(np.max(np.abs(measured - expected)))
    return {
        "accepted": bool(error <= float(tolerance_rad)),
        "maximum_joint_error_rad": error,
        "measured_q_physical_rad": measured.tolist(),
        "expected_q_physical_rad": expected.tolist(),
    }


def _joint_margin(profile: dict, fk, measured_q) -> dict:
    measured = np.asarray(measured_q, dtype=float)
    lower, upper = _joint_bounds(profile, fk)
    margins = np.minimum(measured - lower, upper - measured)
    return {
        "minimum_margin_rad": float(np.min(margins)),
        "per_joint_margin_rad": margins.tolist(),
    }


def _tap_anchor_px(task: dict, image: np.ndarray) -> np.ndarray:
    identity = _load(task["tap_identity"])
    uv = np.asarray(identity["tap"]["uv"], dtype=float)
    if uv.shape != (2,) or np.any(uv < 0.0) or np.any(uv > 1.0):
        raise ValueError("tap identity must contain normalized uv")
    return uv * np.asarray([image.shape[1], image.shape[0]], dtype=float)


def _refresh_planning_scene(task: dict, profile: dict, output: Path):
    """Replace stale movable-object poses with the current RGB-D target."""

    settings = task["scene_refresh"]
    image, depth, confidence, meta, manifest = selected_head_rgbd(
        settings["capture"]
    )
    anchor = _tap_anchor_px(task, image)
    mask, center, detection = detect_media_cap(
        image,
        identity_anchor_px=anchor,
        maximum_anchor_displacement_diagonal_fraction=float(
            task["head_coarse"][
                "maximum_tap_displacement_diagonal_fraction"
            ]
        ),
    )
    _, jaw_mask, _ = detect_open_jaw_center_head(image)

    def median_mask_depth(mask_rgb):
        mask_depth = cv2.resize(
            np.asarray(mask_rgb, dtype=np.uint8) * 255,
            (depth.shape[1], depth.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        valid = (
            (mask_depth >= 64)
            & np.isfinite(depth)
            & (depth > 0)
            & (confidence >= 1)
        )
        if int(np.count_nonzero(valid)) < 8:
            raise ValueError("head target/tool mask lacks metric depth support")
        return float(np.median(depth[valid])), int(np.count_nonzero(valid))

    target_depth, target_depth_pixels = median_mask_depth(mask)
    jaw_depth, jaw_depth_pixels = median_mask_depth(jaw_mask)
    surface, registration = target_surface_in_scene(
        image_bgr=image,
        depth_landscape_m=depth,
        confidence_landscape=confidence,
        meta=meta,
        mask_rgb=mask,
        reference_report=settings["reference_report"],
        reference_capture=settings["reference_capture"],
        tag_id=int(settings["tag_id"]),
        tag_size_m=float(settings["tag_size_m"]),
    )
    model_path = output / "current_target_scene.mjcf"
    completion = write_completed_vertical_object_scene(
        source_model=profile["planning_model"],
        output_model=model_path,
        body_name=settings["body_name"],
        visible_top_scene_xyz_m=surface,
        diameter_m=float(settings["catalog_size_m"][0]),
        height_m=float(settings["catalog_size_m"][2]),
        top_feature_diameter_m=float(settings["cap_size_m"][0]),
        top_feature_height_m=float(settings["cap_size_m"][2]),
        unobserved_dynamic_bodies=tuple(
            settings.get("unobserved_dynamic_bodies", ())
        ),
    )
    refreshed = deepcopy(profile)
    refreshed["planning_model"] = str(model_path)
    report = {
        "schema": "piper_robot.current_target_scene/v1",
        "capture": str(Path(settings["capture"]).resolve()),
        "capture_stationary": bool(
            manifest["robot_state"]["stability"]["stationary"]
        ),
        "target_center_px": center.tolist(),
        "detection": detection,
        "head_depth_separation": {
            "target_median_depth_m": target_depth,
            "open_jaw_median_depth_m": jaw_depth,
            "jaw_behind_target_m": jaw_depth - target_depth,
            "target_depth_pixels": target_depth_pixels,
            "jaw_depth_pixels": jaw_depth_pixels,
            "interpretation": "positive means the open jaw is farther from the head camera",
        },
        "registration": registration,
        "completion": completion,
        "support_relation": settings.get(
            "support_relation", "unknown_support_relation"
        ),
        "support_relation_interpretation": (
            "the completed bottle extends into the recess; both platform "
            "edges remain fixed collision geometry"
        ),
        "paper_fiducials_are_collision_geometry": False,
    }
    (output / "current_target_scene.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return refreshed, report


def _measured_q_from_run(path: str | Path, iteration: int = 0) -> np.ndarray:
    payload = _load(path)
    q = np.asarray(
        payload["iterations"][int(iteration)]["measured_right_q_physical_rad"],
        dtype=float,
    )
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError(f"{path}: empirical view waypoint is invalid")
    return q


def _move_to_empirical_view(task, profile, rpc, fk, scene_refresh):
    settings = task["empirical_view_anchor"]
    observed_target = np.asarray(
        scene_refresh["registration"]["surface_scene_xyz_m"], dtype=float
    )
    anchor_target = np.asarray(settings["target_scene_xyz_m"], dtype=float)
    displacement = float(np.linalg.norm(observed_target[:2] - anchor_target[:2]))
    if displacement > float(settings["maximum_target_displacement_m"]):
        raise RuntimeError(
            "current target is outside the hardware-observed view anchor"
        )
    start = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    waypoints = [
        _measured_q_from_run(item["run"], int(item.get("iteration", 0)))
        for item in settings["waypoints"]
    ]
    knots = [
        {
            "stage": "measured_home_open",
            "right_q_physical_rad": start.tolist(),
            "right_gripper_open_ratio": 1.0,
            "minimum_duration_s": 0.1,
        }
    ]
    for index, (definition, q) in enumerate(zip(settings["waypoints"], waypoints)):
        knots.append(
            {
                "stage": f"empirical_target_view_{index}",
                "right_q_physical_rad": q.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": float(definition["duration_s"]),
            }
        )
    samples = sample_joint_knots(knots)
    path = np.vstack((start, *[sample.right_q_physical_rad for sample in samples]))
    collision = _right_joint_path_contact_audit(profile, path)
    motion = _execute_direct_joint_samples(
        profile,
        rpc,
        fk,
        samples,
        final_tolerance_rad=float(settings["final_tolerance_rad"]),
        endpoint_correction_gain=0.0,
        settle_timeout_s=float(settings["settle_timeout_s"]),
        require_final_convergence=False,
    )
    return {
        "method": "target_registered_hardware_observed_view_anchor",
        "target_displacement_from_anchor_m": displacement,
        "waypoint_sources": settings["waypoints"],
        "model_collision_audit": collision,
        "model_collision_interpretation": (
            "current absolute scene registration has a known false overlap; "
            "the exact high-view waypoints are retained as hardware observations"
        ),
        "motion": motion,
        "descent_authorized": False,
        "closure_authorized": False,
    }


def _observe_head(camera: LiveCamera, task: dict, *, frames: int = 3):
    records = []
    for _ in range(int(frames)):
        image, timestamp = camera.frame()
        anchor = _tap_anchor_px(task, image)
        try:
            cap_mask, detected_center, cap_diag = detect_media_cap(
                image,
                identity_anchor_px=anchor,
                maximum_anchor_displacement_diagonal_fraction=float(
                    task["head_coarse"][
                        "maximum_tap_displacement_diagonal_fraction"
                    ]
                ),
            )
        except RuntimeError as error:
            cap_mask = np.zeros(image.shape[:2], dtype=bool)
            detected_center = anchor.copy()
            cap_diag = {"detection_error": str(error)}
        cap_center = (
            anchor.copy()
            if bool(task["head_coarse"].get("fixed_camera_stationary_target", False))
            else detected_center
        )
        support_center, support_diag = detect_coloured_support_anchor(
            image, target_anchor_px=anchor
        )
        jaw_center, jaw_mask, jaw_diag = detect_open_jaw_center_head(image)
        records.append(
            (
                image,
                timestamp,
                cap_mask,
                cap_center,
                jaw_mask,
                jaw_center,
                cap_diag,
                jaw_diag,
                support_center,
                support_diag,
            )
        )
    cap_centers = np.asarray([item[3] for item in records], dtype=float)
    jaw_centers = np.asarray([item[5] for item in records], dtype=float)
    cap_median = np.median(cap_centers, axis=0)
    jaw_median = np.median(jaw_centers, axis=0)
    representative = min(
        records,
        key=lambda item: float(
            np.linalg.norm(item[3] - cap_median)
            + np.linalg.norm(item[5] - jaw_median)
        ),
    )
    (
        image,
        timestamp,
        cap_mask,
        _,
        jaw_mask,
        _,
        cap_diag,
        jaw_diag,
        _,
        support_diag,
    ) = representative
    support_median = np.median(
        np.asarray([item[8] for item in records], dtype=float), axis=0
    )
    span = float(jaw_diag["jaw_span_px"])
    error = (jaw_median - cap_median) / max(span, 1.0)
    overlay = image.copy()
    overlay[cap_mask] = (0, 255, 0)
    overlay[jaw_mask] = (255, 255, 0)
    cap_px = tuple(np.rint(cap_median).astype(int))
    jaw_px = tuple(np.rint(jaw_median).astype(int))
    cv2.circle(overlay, cap_px, 14, (0, 0, 255), -1)
    cv2.circle(overlay, jaw_px, 14, (0, 255, 255), -1)
    cv2.line(overlay, jaw_px, cap_px, (0, 255, 255), 5, cv2.LINE_AA)
    return {
        "image": image,
        "overlay": overlay,
        "timestamp": float(timestamp),
        "cap_center_px": cap_median,
        "jaw_center_px": jaw_median,
        "error_jaw_spans": error,
        "error_norm": float(np.linalg.norm(error)),
        "cap": cap_diag,
        "jaw": jaw_diag,
        "support_center_px": support_median,
        "support": support_diag,
    }


def _save_head(output: Path, index: int, observed: dict) -> dict:
    raw = output / f"head_{index:02d}.png"
    overlay = output / f"head_{index:02d}_overlay.png"
    cv2.imwrite(str(raw), observed["image"])
    cv2.imwrite(str(overlay), observed["overlay"])
    return {
        "raw": str(raw),
        "overlay": str(overlay),
        "timestamp": observed["timestamp"],
        "cap_center_px": observed["cap_center_px"].tolist(),
        "jaw_center_px": observed["jaw_center_px"].tolist(),
        "error_jaw_spans": observed["error_jaw_spans"].tolist(),
        "error_norm": observed["error_norm"],
        "cap": observed["cap"],
        "jaw": observed["jaw"],
        "support_center_px": observed["support_center_px"].tolist(),
        "support": observed["support"],
    }


def _move_cartesian_delta(profile, rpc, fk, delta_xyz, *, stage: str):
    start = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    target = start.copy()
    target[4:] += np.asarray(delta_xyz, dtype=float)
    return _cartesian_move(
        profile,
        rpc,
        fk,
        target_pose=target,
        duration_s=1.5,
        aperture=1.0,
        stage=stage,
        settle_s=0.35,
    )


def _move_validated_fixed_head_approach(task, profile, rpc, fk):
    """Replay only hardware-observed open-jaw waypoints to the preclose view."""

    settings = task["validated_fixed_head_approach"]
    start = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    knots = [
        {
            "stage": "measured_home_open",
            "right_q_physical_rad": start.tolist(),
            "right_gripper_open_ratio": 1.0,
            "minimum_duration_s": 0.1,
        }
    ]
    for index, waypoint in enumerate(settings["waypoints"]):
        q = np.asarray(waypoint["right_q_physical_rad"], dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("validated fixed-head waypoint must contain six joints")
        knots.append(
            {
                "stage": f"fixed_head_open_approach_{index}",
                "right_q_physical_rad": q.tolist(),
                "right_gripper_open_ratio": 1.0,
                "minimum_duration_s": float(waypoint["duration_s"]),
            }
        )
    samples = sample_joint_knots(knots)
    path = np.vstack(
        (start, *[sample.right_q_physical_rad for sample in samples])
    )
    collision = _right_joint_path_contact_audit(profile, path)
    motion = _execute_direct_joint_samples(
        profile,
        rpc,
        fk,
        samples,
        final_tolerance_rad=float(settings["final_tolerance_rad"]),
        endpoint_correction_gain=0.0,
        settle_timeout_s=float(settings["settle_timeout_s"]),
        require_final_convergence=False,
    )
    return {
        "method": "hardware_observed_open_jaw_fixed_head_approach",
        "model_collision_audit": collision,
        "motion": motion,
    }


def _support_shift_fraction(before: dict, after: dict) -> float:
    distance = float(
        np.linalg.norm(
            np.asarray(after["support_center_px"], dtype=float)
            - np.asarray(before["support_center_px"], dtype=float)
        )
    )
    scale = max(
        float(before["support"]["component_diagonal_px"]),
        float(after["support"]["component_diagonal_px"]),
        1.0,
    )
    return distance / scale


def _fixed_head_preclose_servo(task, profile, rpc, fk, output, report):
    """Align the open-jaw midpoint to the immutable initial head tap."""

    settings = task["fixed_head_preclose"]
    jacobian = np.asarray(
        settings["jaw_error_per_meter_jacobian"], dtype=float
    )
    if jacobian.shape != (2, 3) or not np.all(np.isfinite(jacobian)):
        raise ValueError("fixed-head jaw Jacobian must be finite 2x3")
    camera = LiveCamera("head")
    camera.__enter__()
    try:
        baseline = _observe_head(camera, task)
        report["head_observations"].append(
            _save_head(output, len(report["head_observations"]), baseline)
        )
        for iteration in range(int(settings["maximum_iterations"]) + 1):
            error = np.asarray(baseline["error_jaw_spans"], dtype=float)
            if float(np.linalg.norm(error)) <= float(
                settings["preclose_tolerance_jaw_spans"]
            ):
                return baseline
            if iteration >= int(settings["maximum_iterations"]):
                break
            delta = -np.linalg.pinv(jacobian, rcond=1e-4) @ error
            norm = float(np.linalg.norm(delta))
            maximum = float(settings["maximum_step_m"])
            if norm > maximum:
                delta *= maximum / norm
            motion = _move_cartesian_delta(
                profile,
                rpc,
                fk,
                delta,
                stage="culture_cap_fixed_head_midpoint_servo",
            )
            observed = _observe_head(camera, task)
            shift = _support_shift_fraction(baseline, observed)
            saved = _save_head(
                output, len(report["head_observations"]), observed
            )
            saved["commanded_delta_xyz_m"] = delta.tolist()
            saved["motion"] = motion
            saved["support_shift_fraction"] = shift
            report["head_observations"].append(saved)
            if shift > float(settings["maximum_support_shift_fraction"]):
                _move_cartesian_delta(
                    profile,
                    rpc,
                    fk,
                    -delta,
                    stage="culture_cap_support_motion_backtrack",
                )
                raise RuntimeError("coloured support moved during preclose servo")
            if observed["error_norm"] >= baseline["error_norm"]:
                _move_cartesian_delta(
                    profile,
                    rpc,
                    fk,
                    -delta,
                    stage="culture_cap_nonimproving_backtrack",
                )
                raise RuntimeError("fixed-head preclose correction did not improve")
            baseline = observed
        raise RuntimeError("fixed-head preclose servo did not reach the tap")
    finally:
        camera.__exit__(None, None, None)


def _close_and_validate_fixed_head(task, profile, rpc, fk, output, report, preclose):
    settings = task["closure"]
    pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
    report["closure_commanded"] = True
    report["closure_motion"] = _cartesian_move(
        profile,
        rpc,
        fk,
        target_pose=pose,
        duration_s=float(settings["duration_s"]),
        aperture=0.0,
        stage="culture_cap_close_at_fixed_tap",
        settle_s=float(settings["settle_s"]),
    )
    camera = LiveCamera("head")
    camera.__enter__()
    try:
        closed = _observe_head(camera, task)
    finally:
        camera.__exit__(None, None, None)
    report["head_observations"].append(
        _save_head(output, len(report["head_observations"]), closed)
    )
    anchor = _tap_anchor_px(task, closed["image"])
    segment = fixed_target_in_jaw_segment(
        anchor,
        closed["jaw"]["jaw_centers_px"],
        maximum_perpendicular_span_fraction=float(
            settings["maximum_target_perpendicular_span_fraction"]
        ),
    )
    support_shift = _support_shift_fraction(preclose, closed)
    aperture = float(rpc.get_right_gripper_exact())
    accepted = bool(
        segment["accepted"]
        and support_shift <= float(settings["maximum_support_shift_fraction"])
        and float(settings["minimum_contact_open_ratio"])
        <= aperture
        <= float(settings["maximum_contact_open_ratio"])
    )
    validation = {
        "accepted": accepted,
        "fixed_target_in_jaw_segment": segment,
        "support_shift_fraction": support_shift,
        "gripper_open_ratio": aperture,
    }
    report["closure_validation"] = validation
    if not accepted:
        raise RuntimeError("closed-jaw image/contact validation failed")
    return closed


def _head_coarse_servo(task, profile, rpc, fk, output, report):
    config = task["head_coarse"]
    camera = LiveCamera("head")
    camera.__enter__()
    try:
        baseline = _observe_head(camera, task)
        report["head_observations"].append(_save_head(output, 0, baseline))
        if baseline["error_norm"] <= float(config["tolerance_jaw_spans"]):
            return baseline
        current = baseline
        search_origin_xy = np.asarray(
            rpc.get_right_ee_pose().parameters(), dtype=float
        )[4:6]
        step = float(config["probe_m"])
        minimum_step = float(config["minimum_step_m"])
        maximum_step = float(config["maximum_step_m"])
        minimum_progress = float(config["minimum_progress"])
        directions = [
            np.asarray([1.0, 0.0]),
            np.asarray([-1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, -1.0]),
        ]
        preferred = 0
        image_index = 1
        trials = 0
        while trials < int(config["maximum_iterations"]):
            if current["error_norm"] <= float(config["tolerance_jaw_spans"]):
                report["head_search_status"] = "aligned"
                return current
            accepted = False
            order = [preferred] + [i for i in range(4) if i != preferred]
            for direction_index in order:
                if trials >= int(config["maximum_iterations"]):
                    break
                delta = step * directions[direction_index]
                motion = _move_cartesian_delta(
                    profile,
                    rpc,
                    fk,
                    [delta[0], delta[1], 0.0],
                    stage="head_pattern_search_trial",
                )
                observed = _observe_head(camera, task)
                saved = _save_head(output, image_index, observed)
                saved["commanded_delta_xy_m"] = delta.tolist()
                saved["motion"] = motion
                improvement = float(current["error_norm"] - observed["error_norm"])
                saved["improvement"] = improvement
                saved["trial_accepted"] = bool(improvement >= minimum_progress)
                report["head_observations"].append(saved)
                image_index += 1
                trials += 1
                if improvement >= minimum_progress:
                    current = observed
                    preferred = direction_index
                    step = min(maximum_step, step * 1.25)
                    current_xy = np.asarray(
                        rpc.get_right_ee_pose().parameters(), dtype=float
                    )[4:6]
                    displacement = float(np.linalg.norm(current_xy - search_origin_xy))
                    saved["search_displacement_m"] = displacement
                    if displacement > float(config["maximum_search_displacement_m"]):
                        raise RuntimeError("head pattern search reached its displacement bound")
                    margin = _joint_margin(
                        profile, fk, rpc.get_right_joint_positions()
                    )
                    saved["joint_margin"] = margin
                    if margin["minimum_margin_rad"] < float(
                        config["minimum_joint_margin_rad"]
                    ):
                        raise RuntimeError("head pattern search reached a joint margin")
                    accepted = True
                    break
                undo = _move_cartesian_delta(
                    profile,
                    rpc,
                    fk,
                    [-delta[0], -delta[1], 0.0],
                    stage="head_pattern_search_backtrack",
                )
                restored = _observe_head(camera, task)
                restored_saved = _save_head(output, image_index, restored)
                restored_saved["backtracked_delta_xy_m"] = (-delta).tolist()
                restored_saved["motion"] = undo
                restored_saved["trial_accepted"] = True
                restored_saved["role"] = "restored_search_center"
                report["head_observations"].append(restored_saved)
                image_index += 1
                current = restored
            if not accepted:
                step *= 0.5
                if step < minimum_step:
                    break
        report["head_search_status"] = "stopped_without_alignment"
        raise RuntimeError("bounded head pattern search found no improving move")
    finally:
        camera.__exit__(None, None, None)


def _observe_right(camera, adapter, task):
    first_image, _ = camera.frame()
    tool_frame = detect_light_pad_tool_frame(first_image)
    observed, image, timestamp = stable_observation(camera, adapter, tool_frame)
    goal = np.asarray(task["right_alignment"]["contact_center_uv"], dtype=float)
    overlay = render_target_and_contact_goal(image, observed, tool_frame, goal)
    return observed, image, overlay, timestamp, tool_frame, goal


def _right_align(task, profile, rpc, fk, output, report, *, execute: bool):
    adapter = MediaCapTargetAdapter(
        identity_anchor_uv=task["right_alignment"].get("identity_anchor_uv"),
        maximum_anchor_displacement_diagonal_fraction=float(
            task["right_alignment"].get(
                "maximum_identity_displacement_diagonal_fraction", 0.25
            )
        ),
    )
    camera = LiveCamera("right")
    camera.__enter__()
    fixed_orientation = None
    baseline_xy = baseline_error = x_xy = x_error = jacobian = None
    previous_error = None
    stale = 0
    try:
        for index in range(int(task["right_alignment"]["maximum_iterations"]) + 1):
            observed, image, overlay, timestamp, frame, goal = _observe_right(
                camera, adapter, task
            )
            raw = output / f"right_{index:02d}.png"
            shown = output / f"right_{index:02d}_overlay.png"
            cv2.imwrite(str(raw), image)
            cv2.imwrite(str(shown), overlay)
            error = np.asarray(observed.center_uv) - goal
            measured_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
            pose = np.asarray(rpc.get_right_ee_pose().parameters(), dtype=float)
            if fixed_orientation is None:
                fixed_orientation = pose[:4].copy()
            margin = _joint_margin(profile, fk, measured_q)
            record = {
                "index": index,
                "raw": str(raw),
                "overlay": str(shown),
                "timestamp": timestamp,
                "observation": {**observed.diagnostics, "tool_frame": asdict(frame)},
                "error_uv": error.tolist(),
                "error_norm": float(np.linalg.norm(error)),
                "joint_margin": margin,
                "measured_q_physical_rad": measured_q.tolist(),
            }
            report["right_observations"].append(record)
            if margin["minimum_margin_rad"] < float(
                task["right_alignment"]["minimum_joint_margin_rad"]
            ):
                raise RuntimeError("right alignment reached the configured joint margin")
            if float(np.linalg.norm(error)) <= float(
                task["right_alignment"]["tolerance_tool_units"]
            ):
                report["right_alignment_status"] = "aligned"
                return observed, frame
            if not execute:
                report["right_alignment_status"] = "plan_only"
                return observed, frame
            xy = pose[4:6].copy()
            probe = float(task["right_alignment"]["probe_m"])
            if baseline_xy is None:
                baseline_xy, baseline_error = xy, error
                delta = np.asarray([probe, 0.0])
                method = "right_axis_x_probe"
            elif x_xy is None:
                x_xy, x_error = xy, error
                delta = np.asarray([0.0, probe])
                method = "right_axis_y_probe"
            else:
                if jacobian is None:
                    dx = np.column_stack((x_xy - baseline_xy, xy - x_xy))
                    de = np.column_stack((x_error - baseline_error, error - x_error))
                    jacobian = de @ np.linalg.inv(dx)
                    if not np.all(np.isfinite(jacobian)) or float(
                        np.linalg.cond(jacobian)
                    ) > float(task["right_alignment"]["maximum_jacobian_condition"]):
                        raise RuntimeError("right-camera probes produced invalid Jacobian")
                raw_delta = -np.linalg.pinv(jacobian, rcond=1e-4) @ error
                maximum = float(task["right_alignment"]["maximum_step_m"])
                norm = float(np.linalg.norm(raw_delta))
                delta = raw_delta if norm <= maximum else raw_delta * maximum / norm
                method = "right_frozen_jacobian_servo"
            record["correction"] = {"method": method, "delta_xy_m": delta.tolist()}
            motion = _move_between_hovers(
                profile,
                rpc,
                fk,
                measured_q,
                selected_delta_xy_m=delta,
                fixed_orientation_wxyz=fixed_orientation,
                allow_branch_escape=False,
            )
            record["motion"] = motion
            if motion.get("method") == "fixed_camera_workspace_boundary_hold":
                raise RuntimeError("workspace boundary requires a new high-view plan")
            norm_error = float(np.linalg.norm(error))
            if previous_error is not None:
                improvement = previous_error - norm_error
                stale = stale + 1 if improvement <= float(
                    task["right_alignment"]["minimum_progress_tool_units"]
                ) else 0
                if stale >= 2:
                    raise RuntimeError("right alignment failed to improve twice")
            previous_error = norm_error
            time.sleep(0.15)
        raise RuntimeError("right alignment reached its iteration limit")
    finally:
        camera.__exit__(None, None, None)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-profile",
        default="src/configs/pasteur_culture_media_cap_grasp.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/runs/pasteur/culture_media_cap_adapter_latest",
    )
    parser.add_argument(
        "--stage", choices=("observe", "coarse", "align", "grasp"), default="observe"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    task, profile = load_task_profile(args.task_profile)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "piper_robot.adapter_grasp_run/v1",
        "target_adapter": task["target_adapter"],
        "requested_stage": args.stage,
        "execute": bool(args.execute),
        "fiducials": {
            "role": "localization_only",
            "included_in_collision_geometry": False,
        },
        "head_observations": [],
        "right_observations": [],
        "descent_commanded": False,
        "closure_commanded": False,
    }
    rpc = RPCClient("127.0.0.1", port=8081, timeout_ms=15000)
    try:
        profile, scene_refresh = _refresh_planning_scene(
            task, profile, output
        )
        report["scene_refresh"] = scene_refresh
        fk = ProductionRightFK(profile["production_model"])
        home = _home_gate(rpc, float(task["home"]["maximum_joint_error_rad"]))
        report["home_gate"] = home
        if args.execute and not home["accepted"]:
            raise RuntimeError("adapter grasp execution must start from physical right home")
        home_q = np.asarray(home["measured_q_physical_rad"], dtype=float)
        home_collision = _right_joint_path_contact_audit(
            profile, np.vstack((home_q, home_q))
        )
        report["home_collision_gate"] = home_collision
        separation = float(
            scene_refresh["head_depth_separation"]["jaw_behind_target_m"]
        )
        initial_egress = {
            "accepted": bool(
                home_collision["accepted"]
                or separation
                >= float(task["scene_refresh"]["minimum_initial_egress_depth_m"])
            ),
            "method": (
                "mujoco_home_clear"
                if home_collision["accepted"]
                else "rgbd_depth_separated_vertical_egress"
            ),
            "jaw_behind_target_m": separation,
            "motion_restriction": "vertical_up_only_until_home_overlap_is_cleared",
        }
        report["initial_egress_gate"] = initial_egress
        if args.execute and not initial_egress["accepted"]:
            raise RuntimeError(
                "neither MuJoCo nor current RGB-D authorizes initial egress"
            )
        if args.stage == "observe":
            camera = LiveCamera("head")
            camera.__enter__()
            try:
                observed = _observe_head(camera, task)
                report["head_observations"].append(_save_head(output, 0, observed))
                report["status"] = "observed"
            finally:
                camera.__exit__(None, None, None)
        else:
            if not args.execute:
                raise RuntimeError("coarse/align/grasp stages require --execute")
            if args.stage == "grasp":
                report["validated_fixed_head_approach"] = (
                    _move_validated_fixed_head_approach(
                        task, profile, rpc, fk
                    )
                )
                preclose = _fixed_head_preclose_servo(
                    task, profile, rpc, fk, output, report
                )
                report["status"] = "fixed_head_tap_aligned_open"
                _close_and_validate_fixed_head(
                    task, profile, rpc, fk, output, report, preclose
                )
                report["status"] = "culture_media_cap_grasped_and_held"
            else:
                report["empirical_view_motion"] = _move_to_empirical_view(
                    task, profile, rpc, fk, scene_refresh
                )
                report["status"] = "hardware_observed_right_view_reached"
                if args.stage == "coarse":
                    _right_align(
                        task, profile, rpc, fk, output, report, execute=False
                    )
                if args.stage == "align":
                    _right_align(
                        task, profile, rpc, fk, output, report, execute=True
                    )
                    report["status"] = "right_aligned_open"
        Path(output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        Path(output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        rpc.socket.close(linger=0)
        rpc.context.term()


if __name__ == "__main__":
    main()
