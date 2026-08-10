#!/usr/bin/env python3
"""Observe or empirically correct the physical blue gripper with head RGB-D."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot.rpc import RPCClient
from rollout.camera import CameraFeedManager
from rollout.dish_transport_rehearsal import ProductionArmKinematics
from rollout.piper_realtime_motion import PiperRealtimeMotionPreparation
from rollout.rgbd_gripper_level import (
    confirm_stopped_level_bursts,
    empirical_correction_from_probe,
    measure_blue_gripper_level,
    plan_translation_null_level_probe,
    robust_signed_angle,
)


DEFAULT_CONFIG = ROOT / "src/configs/pasteur_rgbd_gripper_level.json"


def _load(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _camera_pose(value):
    return {
        "translation_xyz_m": [float(value.tx), float(value.ty), float(value.tz)],
        "quaternion_xyzw": [
            float(value.qx),
            float(value.qy),
            float(value.qz),
            float(value.qw),
        ],
    }


def _capture(stage: str, output_dir: Path, settings: dict):
    stop = threading.Event()
    camera = CameraFeedManager(stop, display=False, head_stream=False)
    try:
        camera.start()
        deadline = time.monotonic() + 12.0
        samples = []
        rejected_frames = []
        last_timestamp = None
        while time.monotonic() < deadline:
            rgb, timestamp, depth = camera.get_latest_frame()
            if (
                rgb is None
                or depth is None
                or not np.asarray(depth).size
                or timestamp == last_timestamp
            ):
                time.sleep(0.04)
                continue
            last_timestamp = timestamp
            coefficients = camera.session.get_intrinsic_mat()
            pose = camera.session.get_camera_pose()
            matrix = np.array(
                [
                    [float(coefficients.fx), 0.0, float(coefficients.tx)],
                    [0.0, float(coefficients.fy), float(coefficients.ty)],
                    [0.0, 0.0, 1.0],
                ]
            )
            pose_value = _camera_pose(pose)
            try:
                measurement = measure_blue_gripper_level(
                    rgb,
                    depth,
                    matrix,
                    camera_pose=pose_value,
                    blue_hue_range=settings["blue_hue_range"],
                    minimum_blue_area_fraction=float(
                        settings["minimum_blue_area_fraction"]
                    ),
                    minimum_blue_depth_points=int(
                        settings["minimum_blue_depth_points"]
                    ),
                    minimum_support_points=int(settings["minimum_support_points"]),
                    maximum_support_gravity_disagreement_deg=float(
                        settings["maximum_support_gravity_disagreement_deg"]
                    ),
                    maximum_accepted_angle_deg=float(
                        settings["maximum_accepted_angle_deg"]
                    ),
                )
            except ValueError as error:
                rejected_frames.append(str(error))
                time.sleep(0.01)
                continue
            structural_reasons = set(measurement.reasons) - {
                "physical_blue_jaw_not_horizontal"
            }
            if structural_reasons:
                rejected_frames.append(
                    f"structural: {sorted(structural_reasons)}"
                )
                time.sleep(0.01)
                continue
            samples.append(
                {
                    "rgb": np.asarray(rgb, dtype=np.uint8).copy(),
                    "depth": np.asarray(depth, dtype=np.float32).copy(),
                    "timestamp": float(timestamp),
                    "matrix": matrix.copy(),
                    "pose": pose_value,
                    "measurement": measurement,
                }
            )
            if len(samples) < int(settings["burst_frames"]):
                time.sleep(0.01)
                continue
            burst = robust_signed_angle(
                [
                    sample["measurement"].signed_long_axis_angle_deg
                    for sample in samples
                ],
                maximum_mad_deg=float(settings["maximum_burst_mad_deg"]),
            )
            representative = min(
                samples,
                key=lambda sample: abs(
                    sample["measurement"].signed_long_axis_angle_deg
                    - burst["median_deg"]
                ),
            )
            measurement = representative["measurement"]
            reasons = tuple(
                reason
                for reason in measurement.reasons
                if reason != "physical_blue_jaw_not_horizontal"
            )
            if abs(burst["median_deg"]) > float(
                settings["maximum_accepted_angle_deg"]
            ):
                reasons += ("physical_blue_jaw_not_horizontal",)
            measurement = replace(
                measurement,
                accepted=not reasons,
                signed_long_axis_angle_deg=float(burst["median_deg"]),
                absolute_long_axis_angle_deg=abs(float(burst["median_deg"])),
                reasons=reasons,
            )
            rgb = representative["rgb"]
            depth = representative["depth"]
            timestamp = representative["timestamp"]
            matrix = representative["matrix"]
            pose_value = representative["pose"]
            stage_dir = output_dir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                stage_dir / "frame.npz",
                rgb=np.asarray(rgb, dtype=np.uint8),
                depth_m=np.asarray(depth, dtype=np.float32),
                camera_matrix_rgb=matrix,
                timestamp_s=float(timestamp),
            )
            report = {
                "stage": stage,
                "timestamp_s": float(timestamp),
                "camera_pose": pose_value,
                "measurement": measurement.to_dict(),
                "angle_burst": burst,
                "rejected_frame_count": len(rejected_frames),
                "rejected_frame_reasons": sorted(set(rejected_frames)),
            }
            _write(stage_dir / "measurement.json", report)
            bgr = cv2.cvtColor(
                cv2.rotate(np.asarray(rgb), cv2.ROTATE_90_CLOCKWISE),
                cv2.COLOR_RGB2BGR,
            )
            lines = [
                stage,
                f"physical blue jaw={measurement.signed_long_axis_angle_deg:+.2f} deg",
                f"support/gravity={measurement.support_gravity_disagreement_deg:.2f} deg",
                "PASS" if measurement.accepted else "NOT LEVEL",
            ]
            for index, line in enumerate(lines):
                cv2.putText(
                    bgr,
                    line,
                    (18, 34 + 31 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 0, 0),
                    5,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr,
                    line,
                    (18, 34 + 31 * index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imwrite(str(stage_dir / "head_overlay.jpg"), bgr)
            report["overlay"] = str(stage_dir / "head_overlay.jpg")
            return report
        raise RuntimeError("head Record3D did not provide a fresh RGB-D frame")
    finally:
        stop.set()
        camera.stop()


def _stream(rpc, q, duration_s, gripper, config, check_torque=None):
    target = np.asarray(q, dtype=float).reshape(6)
    duration = float(duration_s)
    rate_hz = float(config["motion"]["rate_hz"])
    maximum_tracking_error = math.radians(
        float(config["motion"]["maximum_tracking_error_deg"])
    )
    start = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    steps = max(1, int(math.ceil(duration * rate_hz)))
    started = time.monotonic()
    for index in range(1, steps + 1):
        if check_torque is not None:
            check_torque()
        fraction = index / steps
        blend = fraction**3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
        command = start + blend * (target - start)
        accepted = rpc.set_right_joint_target(
            command,
            gripper_target=float(gripper),
            preview_time=max(0.05, 2.0 / rate_hz),
        )
        if accepted is False:
            raise RuntimeError("right joint probe command was rejected")
        measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        error = float(np.max(np.abs(measured - command)))
        if error > maximum_tracking_error:
            rpc.set_right_joint_target(
                measured, gripper_target=float(gripper), preview_time=0.2
            )
            raise RuntimeError(
                f"right joint probe tracking error {error:.3f}rad exceeds "
                f"{maximum_tracking_error:.3f}rad"
            )
        delay = started + index / rate_hz - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
    tolerance = math.radians(float(config["motion"]["endpoint_tolerance_deg"]))
    required = int(config["motion"]["endpoint_consecutive_samples"])
    deadline = time.monotonic() + float(config["motion"]["endpoint_timeout_s"])
    integral_gain = float(config["motion"]["endpoint_integral_gain"])
    maximum_bias = math.radians(
        float(config["motion"]["maximum_endpoint_bias_deg"])
    )
    command_bias = np.zeros(6, dtype=float)
    consecutive = 0
    while time.monotonic() < deadline:
        if check_torque is not None:
            check_torque()
        measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        target_error = target - measured
        command_bias = np.clip(
            command_bias + integral_gain * target_error,
            -maximum_bias,
            maximum_bias,
        )
        accepted = rpc.set_right_joint_target(
            target + command_bias,
            gripper_target=float(gripper),
            preview_time=max(0.05, 2.0 / rate_hz),
        )
        if accepted is False:
            raise RuntimeError("right joint endpoint settle command was rejected")
        error = float(np.max(np.abs(measured - target)))
        consecutive = consecutive + 1 if error <= tolerance else 0
        if consecutive >= required:
            rpc.set_right_joint_target(
                measured, gripper_target=float(gripper), preview_time=0.2
            )
            return measured
        time.sleep(1.0 / rate_hz)
    measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    rpc.set_right_joint_target(
        measured, gripper_target=float(gripper), preview_time=0.2
    )
    raise RuntimeError(
        "right joint endpoint did not settle; "
        f"error={float(np.max(np.abs(measured - target))):.4f}rad"
    )


def _signed_angle(capture: dict) -> float:
    return float(capture["measurement"]["signed_long_axis_angle_deg"])


def _confirm_level(
    first_capture: dict,
    *,
    stage_prefix: str,
    output_dir: Path,
    config: dict,
) -> tuple[list[dict], dict]:
    """Open independent Record3D connections and require their consensus."""

    settings = config["confirmation"]
    captures = [first_capture]
    for index in range(1, int(settings["independent_bursts"])):
        captures.append(
            _capture(
                f"{stage_prefix}_confirm_{index}",
                output_dir,
                config["measurement"],
            )
        )
    consensus = confirm_stopped_level_bursts(
        [_signed_angle(capture) for capture in captures],
        [float(capture["angle_burst"]["mad_deg"]) for capture in captures],
        minimum_bursts=int(settings["independent_bursts"]),
        maximum_accepted_angle_deg=float(
            config["measurement"]["maximum_accepted_angle_deg"]
        ),
        maximum_interburst_range_deg=float(
            settings["maximum_interburst_range_deg"]
        ),
        maximum_individual_mad_deg=float(
            config["measurement"]["maximum_burst_mad_deg"]
        ),
    )
    return captures[1:], consensus.to_dict()


def _torque_checker(rpc, limit_nm: float):
    limit = float(limit_nm)

    def check():
        sample = np.asarray(rpc.get_right_joint_torque(), dtype=float)
        if sample.shape != (6,) or not np.all(np.isfinite(sample)):
            raise RuntimeError("invalid right joint torque sample")
        if float(np.max(np.abs(sample))) > limit:
            raise RuntimeError(
                f"right joint torque exceeded {limit:.1f}Nm: {sample.tolist()}"
            )

    return check


def run(config: dict, output_dir: Path, *, execute: bool) -> dict:
    if config["arm"] != "right":
        raise ValueError("the current calibrated profile is physical-right only")
    rpc = RPCClient(
        config["rpc"]["host"],
        int(config["rpc"]["port"]),
        timeout_ms=int(config["rpc"]["timeout_ms"]),
    )
    if execute and not bool(config.get("automatic_correction_enabled", False)):
        raise RuntimeError("automatic physical-level correction is disabled")
    gripper = float(rpc.get_right_gripper_exact())
    kinematics = ProductionArmKinematics(ROOT / config["production_model"], "right")
    level_geometry = _load(ROOT / config["level_config"])
    check_torque = _torque_checker(
        rpc, float(config["motion"]["maximum_torque_nm"])
    )
    preparer = PiperRealtimeMotionPreparation(rpc, "right") if execute else None
    report = {
        "schema": config["schema"],
        "started_at_s": time.time(),
        "execute": bool(execute),
        "gripper_open_ratio": gripper,
        "status": "observing",
        "captures": [],
        "iterations": [],
    }
    _write(output_dir / "run.json", report)
    try:
        if preparer is not None:
            preparer.prepare(check_torque)
        base_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        report["base_q_rad"] = base_q.tolist()
        base = _capture("00_base", output_dir, config["measurement"])
        report["captures"].append(base)
        if not execute:
            report["status"] = "observation_complete"
            return report

        if abs(_signed_angle(base)) <= float(
            config["confirmation"]["trigger_absolute_angle_deg"]
        ):
            extra, consensus = _confirm_level(
                base,
                stage_prefix="00_base",
                output_dir=output_dir,
                config=config,
            )
            report["captures"].extend(extra)
            report["level_consensus"] = consensus
            if consensus["accepted"]:
                report["status"] = "already_level_consensus_hold_current"
                report["calibrated_q_rad"] = np.asarray(
                    rpc.get_right_joint_positions(), dtype=float
                ).tolist()
                report["calibrated_pose_wxyz_xyz"] = np.asarray(
                    rpc.get_right_ee_pose().parameters(), dtype=float
                ).tolist()
                return report
            if "independent_rgbd_bursts_disagree" in consensus["reasons"]:
                report["status"] = "base_consensus_disagreed_hold_current"
                return report
            # These independent bursts agree that the jaw is tilted. Capture
            # a new local anchor so a probe is never scaled from an earlier
            # connection or directly from the consensus median.
            base = _capture(
                "00_base_reanchor", output_dir, config["measurement"]
            )
            report["captures"].append(base)

        maximum_iterations = int(config["correction"]["maximum_iterations"])
        for iteration_index in range(maximum_iterations):
            prefix = f"iter_{iteration_index:02d}"
            anchor_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
            if iteration_index:
                base = _capture(
                    f"{prefix}_base", output_dir, config["measurement"]
                )
                report["captures"].append(base)
            structural_reasons = set(base["measurement"]["reasons"]) - {
                "physical_blue_jaw_not_horizontal"
            }
            if structural_reasons:
                raise RuntimeError(
                    f"RGB-D measurement gate failed: {sorted(structural_reasons)}"
                )
            probe = plan_translation_null_level_probe(
                kinematics.pose,
                anchor_q,
                approach_axis_ee=level_geometry["approach_axis_ee"],
                baseline_axis_ee=level_geometry["tip_baseline_ee"],
                target_probe_angle_deg=float(config["probe"]["target_angle_deg"]),
                maximum_joint_delta_rad=math.radians(
                    float(config["probe"]["maximum_joint_delta_deg"])
                ),
            )
            if not probe.accepted:
                raise RuntimeError(f"joint probe plan rejected: {probe.reasons}")
            iteration = {
                "index": iteration_index,
                "anchor_q_rad": anchor_q.tolist(),
                "base_stage": base["stage"],
                "base_angle_deg": _signed_angle(base),
                "probe_plan": probe.to_dict(),
                "fresh_probe_required": True,
            }
            report["iterations"].append(iteration)
            if iteration_index == 0:
                report["probe_plan"] = probe.to_dict()

            probe_q = anchor_q + np.asarray(probe.probe_delta_q_rad)
            _stream(
                rpc,
                probe_q,
                config["probe"]["duration_s"],
                gripper,
                config,
                check_torque,
            )
            probe_capture = _capture(
                f"{prefix}_probe", output_dir, config["measurement"]
            )
            report["captures"].append(probe_capture)
            _stream(
                rpc,
                anchor_q,
                config["probe"]["return_duration_s"],
                gripper,
                config,
                check_torque,
            )
            returned = _capture(
                f"{prefix}_return", output_dir, config["measurement"]
            )
            report["captures"].append(returned)
            return_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
            return_joint_error = float(np.max(np.abs(return_q - anchor_q)))
            return_angle_error = abs(_signed_angle(returned) - _signed_angle(base))
            iteration["return_gate"] = {
                "maximum_joint_error_rad": return_joint_error,
                "angle_error_deg": return_angle_error,
            }
            if return_joint_error > math.radians(
                float(config["probe"]["maximum_return_joint_error_deg"])
            ):
                raise RuntimeError("probe did not return to its fresh anchor branch")
            if return_angle_error > float(
                config["probe"]["maximum_return_angle_error_deg"]
            ):
                raise RuntimeError("RGB-D angle was not repeatable after probe return")

            correction = empirical_correction_from_probe(
                _signed_angle(base),
                _signed_angle(probe_capture),
                probe.probe_delta_q_rad,
                maximum_correction_angle_deg=float(
                    config["correction"]["maximum_angle_per_iteration_deg"]
                ),
                minimum_observed_probe_change_deg=float(
                    config["probe"]["minimum_observed_change_deg"]
                ),
                maximum_joint_delta_rad=math.radians(
                    float(config["correction"]["maximum_joint_delta_deg"])
                ),
            )
            iteration["empirical_correction_q_rad"] = correction.tolist()
            _stream(
                rpc,
                anchor_q + correction,
                config["correction"]["duration_s"],
                gripper,
                config,
                check_torque,
            )
            final = _capture(
                f"{prefix}_corrected", output_dir, config["measurement"]
            )
            report["captures"].append(final)
            improvement = abs(_signed_angle(base)) - abs(_signed_angle(final))
            iteration["corrected_angle_deg"] = _signed_angle(final)
            iteration["improvement_deg"] = improvement

            if improvement < float(config["correction"]["minimum_improvement_deg"]):
                _stream(
                    rpc,
                    anchor_q,
                    config["probe"]["return_duration_s"],
                    gripper,
                    config,
                    check_torque,
                )
                rollback = _capture(
                    f"{prefix}_rollback", output_dir, config["measurement"]
                )
                report["captures"].append(rollback)
                iteration["rollback_angle_deg"] = _signed_angle(rollback)
                report["status"] = "correction_rejected_returned_to_fresh_anchor"
                break

            if abs(_signed_angle(final)) <= float(
                config["confirmation"]["trigger_absolute_angle_deg"]
            ):
                extra, consensus = _confirm_level(
                    final,
                    stage_prefix=f"{prefix}_corrected",
                    output_dir=output_dir,
                    config=config,
                )
                report["captures"].extend(extra)
                iteration["level_consensus"] = consensus
                if consensus["accepted"]:
                    report["level_consensus"] = consensus
                    report["calibrated_q_rad"] = np.asarray(
                        rpc.get_right_joint_positions(), dtype=float
                    ).tolist()
                    report["calibrated_pose_wxyz_xyz"] = np.asarray(
                        rpc.get_right_ee_pose().parameters(), dtype=float
                    ).tolist()
                    report["status"] = "level_consensus_accepted_hold_current"
                    break
                if "independent_rgbd_bursts_disagree" in consensus["reasons"]:
                    _stream(
                        rpc,
                        anchor_q,
                        config["probe"]["return_duration_s"],
                        gripper,
                        config,
                        check_torque,
                    )
                    report["status"] = (
                        "candidate_consensus_disagreed_returned_to_fresh_anchor"
                    )
                    break

            if iteration_index + 1 == maximum_iterations:
                report["status"] = "improved_fresh_probe_required_hold_current"
            else:
                report["status"] = "improved_reprobing_from_current"
    except BaseException:
        measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        rpc.set_right_joint_target(measured, gripper_target=gripper, preview_time=0.2)
        report["status"] = "failed_hold_current"
        raise
    finally:
        if preparer is not None:
            preparer.finish()
        report["finished_at_s"] = time.time()
        _write(output_dir / "run.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = _load(args.config)
    result = run(config, Path(args.output_dir), execute=args.execute)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
