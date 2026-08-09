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
from rollout.branch_locked_contact import JointWaypoint, stream_joint_waypoint
from rollout.camera import CameraFeedManager
from rollout.dish_transport_rehearsal import ProductionArmKinematics
from rollout.rgbd_gripper_level import (
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
            structural_reasons = set(measurement.reasons) - {
                "physical_blue_jaw_not_horizontal"
            }
            if structural_reasons:
                raise RuntimeError(
                    f"RGB-D structural measurement failed: {sorted(structural_reasons)}"
                )
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


def _stream(rpc, q, duration_s, gripper, config):
    return stream_joint_waypoint(
        rpc,
        arm="right",
        waypoint=JointWaypoint(np.asarray(q, dtype=float), float(duration_s)),
        gripper_target=float(gripper),
        rate_hz=float(config["motion"]["rate_hz"]),
        maximum_tracking_error_rad=math.radians(
            float(config["motion"]["maximum_tracking_error_deg"])
        ),
    )


def run(config: dict, output_dir: Path, *, execute: bool) -> dict:
    if config["arm"] != "right":
        raise ValueError("the current calibrated profile is physical-right only")
    rpc = RPCClient(
        config["rpc"]["host"],
        int(config["rpc"]["port"]),
        timeout_ms=int(config["rpc"]["timeout_ms"]),
    )
    q0 = np.asarray(rpc.get_right_joint_positions(), dtype=float)
    gripper = float(rpc.get_right_gripper_exact())
    kinematics = ProductionArmKinematics(ROOT / config["production_model"], "right")
    probe = plan_translation_null_level_probe(
        kinematics.pose,
        q0,
        target_probe_angle_deg=float(config["probe"]["target_angle_deg"]),
        maximum_joint_delta_rad=math.radians(
            float(config["probe"]["maximum_joint_delta_deg"])
        ),
    )
    report = {
        "schema": config["schema"],
        "started_at_s": time.time(),
        "execute": bool(execute),
        "base_q_rad": q0.tolist(),
        "gripper_open_ratio": gripper,
        "probe_plan": probe.to_dict(),
        "status": "observing",
    }
    _write(output_dir / "run.json", report)
    base = _capture("00_base", output_dir, config["measurement"])
    report["captures"] = [base]
    if not execute:
        report["status"] = "observation_complete"
        report["finished_at_s"] = time.time()
        _write(output_dir / "run.json", report)
        return report
    structural_reasons = set(base["measurement"]["reasons"]) - {
        "physical_blue_jaw_not_horizontal"
    }
    if structural_reasons:
        raise RuntimeError(f"RGB-D measurement gate failed: {sorted(structural_reasons)}")
    if not probe.accepted:
        raise RuntimeError(f"joint probe plan rejected: {probe.reasons}")
    try:
        probe_q = q0 + np.asarray(probe.probe_delta_q_rad)
        _stream(
            rpc,
            probe_q,
            config["probe"]["duration_s"],
            gripper,
            config,
        )
        probe_capture = _capture("01_probe", output_dir, config["measurement"])
        report["captures"].append(probe_capture)
        _stream(
            rpc,
            q0,
            config["probe"]["return_duration_s"],
            gripper,
            config,
        )
        returned = _capture("02_return", output_dir, config["measurement"])
        report["captures"].append(returned)
        return_q = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        return_joint_error = float(np.max(np.abs(return_q - q0)))
        return_angle_error = abs(
            float(returned["measurement"]["signed_long_axis_angle_deg"])
            - float(base["measurement"]["signed_long_axis_angle_deg"])
        )
        report["return_gate"] = {
            "maximum_joint_error_rad": return_joint_error,
            "angle_error_deg": return_angle_error,
        }
        if return_joint_error > math.radians(
            float(config["probe"]["maximum_return_joint_error_deg"])
        ):
            raise RuntimeError("probe did not return to the starting joint branch")
        if return_angle_error > float(
            config["probe"]["maximum_return_angle_error_deg"]
        ):
            raise RuntimeError("RGB-D angle was not repeatable after probe return")
        correction = empirical_correction_from_probe(
            float(base["measurement"]["signed_long_axis_angle_deg"]),
            float(probe_capture["measurement"]["signed_long_axis_angle_deg"]),
            probe.probe_delta_q_rad,
            maximum_correction_angle_deg=float(
                config["correction"]["maximum_angle_per_run_deg"]
            ),
            minimum_observed_probe_change_deg=float(
                config["probe"]["minimum_observed_change_deg"]
            ),
            maximum_joint_delta_rad=math.radians(
                float(config["correction"]["maximum_joint_delta_deg"])
            ),
        )
        report["empirical_correction_q_rad"] = correction.tolist()
        _stream(
            rpc,
            q0 + correction,
            config["correction"]["duration_s"],
            gripper,
            config,
        )
        final = _capture("03_corrected", output_dir, config["measurement"])
        report["captures"].append(final)
        improvement = abs(float(base["measurement"]["signed_long_axis_angle_deg"])) - abs(
            float(final["measurement"]["signed_long_axis_angle_deg"])
        )
        report["improvement_deg"] = improvement
        if improvement < float(config["correction"]["minimum_improvement_deg"]):
            _stream(
                rpc,
                q0,
                config["probe"]["return_duration_s"],
                gripper,
                config,
            )
            report["status"] = "correction_rejected_returned_to_base"
        else:
            report["status"] = (
                "level_accepted_hold_current"
                if final["measurement"]["accepted"]
                else "improved_hold_current"
            )
    except BaseException:
        measured = np.asarray(rpc.get_right_joint_positions(), dtype=float)
        rpc.set_right_joint_target(measured, gripper_target=gripper, preview_time=0.2)
        report["status"] = "failed_hold_current"
        raise
    finally:
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
