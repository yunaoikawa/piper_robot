#!/usr/bin/env python3
"""SAM-only staged pregrasp: horizontal alignment before any descent."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_realtime_sam_grasp import LiveSamGrasp
from robot.camera_id import load_camera_map
from rollout.camera import USBWristCameraFeedManager
from rollout.sam_segmentation import (
    choose_lid_candidate,
    enhance_low_light,
)


def fit_horizontal_jacobian(robot_deltas, feature_deltas, rcond=0.12):
    """Return image XY change per robot XY metre from horizontal probes."""

    robot = np.asarray(robot_deltas, dtype=float)
    feature = np.asarray(feature_deltas, dtype=float)
    if (
        robot.ndim != 2
        or feature.ndim != 2
        or robot.shape != feature.shape
        or robot.shape[1] != 3
        or robot.shape[0] < 2
    ):
        raise ValueError("at least two paired XYZ observations are required")
    horizontal_robot = robot[:, :2]
    horizontal_feature = feature[:, :2]
    if np.linalg.matrix_rank(horizontal_robot, tol=5e-4) < 2:
        raise ValueError("live probes span fewer than two horizontal directions")
    jacobian = horizontal_feature.T @ np.linalg.pinv(
        horizontal_robot.T, rcond=rcond
    )
    return jacobian


def estimate_horizontal_displacement(jacobian, feature_error):
    """Map SAM pixel error into local robot XY without requesting descent."""

    jacobian = np.asarray(jacobian, dtype=float).reshape(2, 2)
    error = np.asarray(feature_error, dtype=float).reshape(3)[:2]
    horizontal = np.linalg.lstsq(jacobian, error, rcond=0.12)[0]
    return np.array([horizontal[0], horizontal[1], 0.0])


def bound_horizontal_step(displacement, max_norm_m, max_axis_m):
    step = np.zeros(3, dtype=float)
    step[:2] = np.asarray(displacement, dtype=float).reshape(3)[:2]
    peak = float(np.max(np.abs(step[:2])))
    if peak > max_axis_m:
        step[:2] *= max_axis_m / peak
    norm = float(np.linalg.norm(step[:2]))
    if norm > max_norm_m:
        step[:2] *= max_norm_m / norm
    return step


class RightLidObserver:
    """Independent live SAM check from the right wrist camera."""

    def __init__(self, runner: LiveSamGrasp, output_dir: Path):
        camera_map = load_camera_map()
        self.runner = runner
        self.camera = USBWristCameraFeedManager(
            runner.stop_event,
            device_index=camera_map.get("right", 2),
            label="right wrist",
        )
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.previous_center = None
        self.sequence = 0

    def start(self):
        self.camera.start()
        deadline = time.time() + 12.0
        while time.time() < deadline:
            rgb, timestamp, _ = self.camera.get_latest_frame()
            if rgb is not None and timestamp is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("right wrist camera frame unavailable")

    def stop(self):
        self.camera.stop()

    def observe(self, *, require_lid=True):
        rgb, timestamp, _ = self.camera.get_latest_frame()
        if rgb is None or timestamp is None:
            raise RuntimeError("right wrist camera frame disappeared")
        if time.time() - float(timestamp) > 0.5:
            raise RuntimeError("stale right wrist camera frame")
        image = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sequence = self.sequence
        self.sequence += 1
        raw_path = self.output_dir / f"{sequence:03d}_raw.png"
        cv2.imwrite(str(raw_path), image)
        sam_image = (
            enhance_low_light(image) if float(image.mean()) < 35.0 else image
        )
        input_path = self.output_dir / f"{sequence:03d}_sam_input.png"
        cv2.imwrite(str(input_path), sam_image)
        selected = None
        attempts = []
        for prompt in (
            "transparent round petri dish lid with blue cross",
            "petri dish lid",
            "round transparent plastic dish",
        ):
            result = self.runner.sam.segment(
                sam_image,
                frame_id=self.runner.frame_id,
                timestamp=timestamp,
                prompt=prompt,
                confidence_threshold=0.05,
            )
            self.runner.frame_id += 1
            attempts.append((prompt, len(result.candidates)))
            selected = choose_lid_candidate(
                result.candidates,
                image_bgr=sam_image,
                previous_center_px=self.previous_center,
                require_blue_cross=True,
            )
            if selected is not None:
                break
        if selected is None:
            if require_lid:
                raise RuntimeError(
                    "right SAM did not identify the blue-cross lid; "
                    f"attempts={attempts}, raw={raw_path}, input={input_path}"
                )
            overlay = sam_image.copy()
            label = f"RIGHT lid outside view; attempts={attempts}"
            cv2.putText(
                overlay,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                overlay,
                label,
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            path = self.output_dir / f"{sequence:03d}.png"
            cv2.imwrite(str(path), overlay)
            return None, None, str(path), float(timestamp)
        candidate, geometry = selected
        self.previous_center = geometry.center_px.copy()
        overlay = sam_image.copy()
        mask = np.asarray(candidate.mask, dtype=bool)
        tint = np.zeros_like(overlay)
        tint[:] = (0, 255, 0)
        overlay[mask] = cv2.addWeighted(
            overlay[mask], 0.55, tint[mask], 0.45, 0
        )
        center = tuple(np.rint(geometry.center_px).astype(int))
        cv2.drawMarker(
            overlay, center, (0, 255, 0), cv2.MARKER_CROSS, 30, 3
        )
        label = (
            f"RIGHT live SAM score={candidate.score:.3f} "
            f"center=({center[0]},{center[1]})"
        )
        cv2.putText(
            overlay,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            overlay,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        path = self.output_dir / f"{sequence:03d}.png"
        cv2.imwrite(str(path), overlay)
        return geometry, candidate, str(path), float(timestamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-endpoint", default="tcp://127.0.0.1:15563")
    parser.add_argument(
        "--torque-config", default="src/configs/pasteur_lid_torque.json"
    )
    parser.add_argument(
        "--scene-config",
        default="src/configs/pasteur_lid_scene3d.json",
    )
    parser.add_argument("--reference-points")
    parser.add_argument(
        "--output-dir", default="/tmp/realtime_sam_horizontal"
    )
    parser.add_argument("--depth-frames", type=int, default=15)
    parser.add_argument("--preview-time", type=float, default=2.0)
    parser.add_argument("--probe-m", type=float, default=0.006)
    parser.add_argument("--max-step-m", type=float, default=0.008)
    parser.add_argument("--max-axis-m", type=float, default=0.006)
    parser.add_argument("--horizontal-tolerance-m", type=float, default=0.006)
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument(
        "--execute-horizontal",
        action="store_true",
        help="allow right-arm horizontal motion; default is camera-only dry-run",
    )
    args = parser.parse_args()

    # LiveSamGrasp owns camera, SAM, torque monitoring, and right-only RPC.
    args.control_space = "cartesian"
    args.minimum_progress = 0.0
    args.joint_minimum_progress = 0.0
    runner = LiveSamGrasp(args)
    right = RightLidObserver(
        runner, Path(args.output_dir) / "right"
    )
    samples_robot = []
    samples_feature = []
    moved = False
    try:
        runner.start()
        right.start()
        runner.observe(0.0)  # warm Record3D temporal depth
        feature, raw_error, head_path, timestamp = runner.observe(0.0)
        registration = runner.check_scene_registration(
            args.reference_points
        )
        right_geometry, right_candidate, right_path, right_timestamp = (
            right.observe(require_lid=False)
        )
        print(
            json.dumps(
                {
                    "mode": (
                        "HORIZONTAL_ONLY"
                        if args.execute_horizontal
                        else "DRY_RUN_NO_MOTION"
                    ),
                    "feature_error": raw_error.round(2).tolist(),
                    "target_camera_xyz_m": (
                        runner.last_target_3d.point_camera_xyz_m.round(4).tolist()
                    ),
                    "support_plane_confidence": (
                        runner.last_target_3d.confidence
                    ),
                    "geometry_quality": {
                        "accepted": runner.last_geometry_quality.accepted,
                        "view_angle_deg": (
                            runner.last_geometry_quality.view_angle_deg
                        ),
                        "native_depth_pixel_footprint_mm": [
                            runner.last_geometry_quality.native_pixel_footprint_x_m
                            * 1000.0,
                            runner.last_geometry_quality.native_pixel_footprint_y_m
                            * 1000.0,
                        ],
                        "reasons": list(
                            runner.last_geometry_quality.reasons
                        ),
                    },
                    "proximity_warning_only_m": (
                        runner.last_proximity_m
                    ),
                    "registration": registration,
                    "head_image": head_path,
                    "head_timestamp": timestamp,
                    "right_lid_center_px": (
                        right_geometry.center_px.round(2).tolist()
                        if right_geometry is not None
                        else None
                    ),
                    "right_lid_score": (
                        float(right_candidate.score)
                        if right_candidate is not None
                        else None
                    ),
                    "right_image": right_path,
                    "right_timestamp": right_timestamp,
                }
            ),
            flush=True,
        )
        if not args.execute_horizontal:
            return
        if not runner.last_geometry_quality.accepted:
            raise RuntimeError(
                "camera geometry is too oblique for horizontal execution: "
                + "; ".join(runner.last_geometry_quality.reasons)
            )
        origin = np.asarray(
            runner.rpc.get_right_ee_pose().parameters(), dtype=float
        )

        # Horizontal calibration only.  The old version probed Z here, which
        # violated the required align-first/descend-later state ordering.
        for axis in range(2):
            before, _, _, _ = runner.observe(0.0)
            current = np.asarray(
                runner.rpc.get_right_ee_pose().parameters(), dtype=float
            )
            request = np.zeros(3, dtype=float)
            request[axis] = args.probe_m
            if axis < 2:
                # Horizontal probes may experience a little IK sag. Ask the
                # controller to restore, never lower, the starting height.
                request[2] = max(0.0, float(origin[6] - current[6]))
            moved = True
            actual = runner.move_cartesian_delta(
                request, minimum_progress=0.0
            )
            after, _, path, _ = runner.observe(0.0)
            delta_feature = (
                after.gripper_feature - before.gripper_feature
            )
            if np.linalg.norm(actual) < 5e-4:
                raise RuntimeError(
                    f"Cartesian probe {axis} produced no motion"
                )
            samples_robot.append(actual)
            samples_feature.append(delta_feature)
            print(
                json.dumps(
                    {
                        "probe_axis": axis,
                        "actual_xyz_m": actual.round(5).tolist(),
                        "feature_delta": delta_feature.round(2).tolist(),
                        "image": path,
                    }
                ),
                flush=True,
            )

        worse = 0
        previous_horizontal = None
        final_path = None
        for iteration in range(args.max_iters):
            feature, raw_error, path, timestamp = runner.observe(0.0)
            (
                right_geometry,
                right_candidate,
                right_path,
                right_timestamp,
            ) = right.observe(require_lid=False)
            jacobian = fit_horizontal_jacobian(
                samples_robot, samples_feature
            )
            displacement = estimate_horizontal_displacement(
                jacobian, raw_error
            )
            horizontal_norm = float(
                np.linalg.norm(displacement[:2])
            )
            report = {
                "stage": "HORIZONTAL_ONLY",
                "iteration": iteration,
                "feature_error": raw_error.round(2).tolist(),
                "estimated_robot_displacement_m": (
                    displacement.round(4).tolist()
                ),
                "horizontal_norm_m": horizontal_norm,
                "descent_not_executed_m": float(displacement[2]),
                "image": path,
                "timestamp": timestamp,
                "right_lid_center_px": (
                    right_geometry.center_px.round(2).tolist()
                    if right_geometry is not None
                    else None
                ),
                "right_lid_score": (
                    float(right_candidate.score)
                    if right_candidate is not None
                    else None
                ),
                "right_image": right_path,
                "right_timestamp": right_timestamp,
            }
            print(json.dumps(report), flush=True)
            final_path = path
            if horizontal_norm <= args.horizontal_tolerance_m:
                runner.hold_measured()
                if right_candidate is None:
                    print(
                        json.dumps(
                            {
                                "status": (
                                    "HORIZONTAL_ALIGNED_"
                                    "RIGHT_CONFIRMATION_MISSING"
                                ),
                                "image": path,
                                "right_image": right_path,
                                "estimated_descent_m": float(
                                    displacement[2]
                                ),
                            }
                        ),
                        flush=True,
                    )
                    return
                print(
                    json.dumps(
                        {
                            "status": "HORIZONTAL_ALIGNED_DESCENT_PAUSED",
                            "image": path,
                            "estimated_descent_m": float(
                                displacement[2]
                            ),
                        }
                    ),
                    flush=True,
                )
                return

            if (
                previous_horizontal is not None
                and horizontal_norm > 1.25 * previous_horizontal
            ):
                worse += 1
                if worse >= 2:
                    runner.hold_measured()
                    raise RuntimeError(
                        "horizontal-only SAM estimate diverged"
                    )
            else:
                worse = 0

            step = bound_horizontal_step(
                displacement,
                max_norm_m=args.max_step_m,
                max_axis_m=args.max_axis_m,
            )
            current = np.asarray(
                runner.rpc.get_right_ee_pose().parameters(), dtype=float
            )
            # Correct only upward for any IK sag; this stage never requests
            # descent below its starting height.
            step[2] = np.clip(
                max(0.0, float(origin[6] - current[6])),
                0.0,
                0.002,
            )
            before_gripper = feature.gripper_feature.copy()
            moved = True
            actual = runner.move_cartesian_delta(
                step, minimum_progress=0.0
            )
            after, _, _, _ = runner.observe(0.0)
            if np.linalg.norm(actual) >= 5e-4:
                samples_robot.append(actual)
                samples_feature.append(
                    after.gripper_feature - before_gripper
                )
                samples_robot[:] = samples_robot[-8:]
                samples_feature[:] = samples_feature[-8:]
            previous_horizontal = horizontal_norm

        runner.hold_measured()
        raise RuntimeError(
            f"horizontal-only alignment did not converge; image={final_path}"
        )
    except BaseException:
        if moved:
            try:
                runner.hold_measured()
            except Exception:
                pass
        raise
    finally:
        try:
            right.stop()
        except Exception:
            pass
        runner.stop()


if __name__ == "__main__":
    main()
