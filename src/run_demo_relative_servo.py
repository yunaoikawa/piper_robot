#!/usr/bin/env python3
"""Run the generic demo-relative controller on the right arm.

The process is intentionally quasi-static: capture a fresh synchronized
observation, compute one bounded decision, execute it, settle, and observe
again.  It never homes the robot and it pauses before demonstrated contact
unless ``--auto-contact`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robot.camera_id import load_camera_map
from robot.rpc import RPCClient
from rollout.apriltag_retarget import detect_tags, render_tags
from rollout.camera import CameraFeedManager, USBWristCameraFeedManager
from rollout.lid_task import LidTaskAdapter
from rollout.lid_vision import inspect_lid, render_inspection
from rollout.visual_servo import (
    DemoRelativeServo,
    ManipulationTemplate,
    ServoAction,
    ServoConfig,
    ServoPhase,
    StampedObservation,
)


def camera_bgr(frame):
    if frame is None or np.asarray(frame).size == 0:
        return None
    return cv2.cvtColor(np.rot90(frame, k=3), cv2.COLOR_RGB2BGR)


def usable_frame(frame):
    if frame is None or np.asarray(frame).size == 0:
        return False
    # Record3D can report callbacks while the phone stream is an all-black
    # placeholder.  Treat it as unavailable instead of feeding it to vision.
    return float(np.percentile(np.asarray(frame), 99)) > 5.0


class LiveSource:
    def __init__(self, rpc, output_dir):
        self.rpc = rpc
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        camera_map = load_camera_map()
        self.head = CameraFeedManager(
            self.stop_event, display=False, head_stream=False
        )
        self.right = USBWristCameraFeedManager(
            self.stop_event,
            device_index=camera_map.get("right", 2),
            label="right wrist",
        )

    def start(self, timeout_s=15.0):
        self.head.start()
        self.right.start()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            head, _, _ = self.head.get_latest_frame()
            right, _, _ = self.right.get_latest_frame()
            if usable_frame(head) and usable_frame(right):
                return
            time.sleep(0.1)
        raise RuntimeError("head/right Record3D frames are not both available")

    def stop(self):
        self.stop_event.set()
        self.head.stop()
        self.right.stop()

    def observe(self):
        head_rgb, head_ts, head_depth = self.head.get_latest_frame()
        right_rgb, right_ts, right_depth = self.right.get_latest_frame()
        if head_rgb is None or right_rgb is None:
            raise RuntimeError("camera frame disappeared")
        pose = self.rpc.get_right_ee_pose()
        joints = np.asarray(self.rpc.get_right_joint_positions(), dtype=float)
        torque = np.asarray(self.rpc.get_right_joint_torque(), dtype=float)
        ratio = float(self.rpc.get_right_gripper_exact())
        # Movement is quasi-static, so the newest of the two receive timestamps
        # is paired with the state sampled immediately afterwards.
        timestamp = min(float(head_ts), float(right_ts))
        depths = {}
        if head_depth is not None:
            depths["head"] = np.rot90(head_depth, k=3)
        if right_depth is not None:
            depths["right"] = np.rot90(right_depth, k=3)
        return StampedObservation(
            timestamp=timestamp,
            ee_pose=np.asarray(pose.parameters(), dtype=float),
            joint_positions=joints,
            gripper_ratio=ratio,
            images={"head": camera_bgr(head_rgb), "right": camera_bgr(right_rgb)},
            depths=depths,
            torque=torque,
        )

    def save_overlay(self, sequence, observation, adapter, decision):
        head = observation.images["head"]
        detections = detect_tags(head, adapter.tag_profile.family)
        head_overlay = render_tags(head, detections)
        tracker = adapter.last_tracker
        if tracker is not None:
            center = tuple(np.rint(tracker["center"]).astype(int))
            cv2.circle(head_overlay, center, 16, (0, 255, 255), 3)
            cv2.putText(
                head_overlay,
                "LID BLUE CROSS",
                (center[0] + 18, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        text = f"{decision.phase.value}: {decision.action.value} {decision.reason}"
        cv2.putText(
            head_overlay, text, (10, head_overlay.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3,
        )
        cv2.putText(
            head_overlay, text, (10, head_overlay.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1,
        )
        right = observation.images["right"]
        right_result = inspect_lid(right, adapter.wrist_profile)
        right_overlay = render_inspection(right, right_result, "RIGHT")
        head_path = self.output_dir / f"{sequence:03d}_head.png"
        right_path = self.output_dir / f"{sequence:03d}_right.png"
        head_raw_path = self.output_dir / f"{sequence:03d}_head_raw.png"
        right_raw_path = self.output_dir / f"{sequence:03d}_right_raw.png"
        cv2.imwrite(str(head_path), head_overlay)
        cv2.imwrite(str(right_path), right_overlay)
        cv2.imwrite(str(head_raw_path), head)
        cv2.imwrite(str(right_raw_path), right)
        return {
            "head": str(head_path),
            "right": str(right_path),
            "head_raw": str(head_raw_path),
            "right_raw": str(right_raw_path),
        }


def execute(rpc, decision, dry_run):
    if dry_run or decision.action in (ServoAction.HOLD, ServoAction.COMPLETE):
        return
    current = rpc.get_right_ee_pose()
    if decision.action == ServoAction.MOVE:
        requested = np.asarray(decision.target_pose, dtype=float).copy()
        delta = requested[4:7] - current.translation()
        # Piper accepts but does not physically execute very small joint
        # commands. Extrapolate only Cartesian translation; never multiply the
        # orientation correction or feed the undershot pose back as a goal.
        if (
            decision.phase != ServoPhase.WRIST_SERVO
            and 5e-4 < np.linalg.norm(delta) < 0.010
        ):
            requested[4:7] = current.translation() + 2.0 * delta
        rpc.set_right_ee_target(
            mink.SE3(requested),
            gripper_target=1.0,
            preview_time=0.8,
        )
    elif decision.action == ServoAction.CLOSE:
        rpc.set_right_ee_target(current, gripper_target=0.0, preview_time=0.2)
    elif decision.action == ServoAction.OPEN:
        rpc.set_right_ee_target(current, gripper_target=1.0, preview_time=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tag-profile", default="src/configs/pasteur_lid_tags.json"
    )
    ap.add_argument(
        "--vision-profile", default="src/configs/pasteur_lid_vision.json"
    )
    ap.add_argument(
        "--template", default="src/configs/pasteur_lid_grasp_template.json"
    )
    ap.add_argument("--output-dir", default="/tmp/demo_relative_servo")
    ap.add_argument("--max-cycles", type=int, default=30)
    ap.add_argument("--settle-s", type=float, default=0.9)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--auto-contact", action="store_true")
    args = ap.parse_args()

    template = ManipulationTemplate.load(args.template)
    adapter = LidTaskAdapter.load(
        args.tag_profile,
        args.vision_profile,
        empty_close_ratio=template.empty_close_ratio,
    )
    servo = DemoRelativeServo(
        adapter,
        template,
        ServoConfig(require_contact_confirmation=not args.auto_contact),
    )
    rpc = RPCClient("localhost", 8081, timeout_ms=10000)
    rpc.init()
    source = LiveSource(rpc, args.output_dir)
    source.start()
    try:
        for sequence in range(args.max_cycles):
            observation = source.observe()
            decision = servo.step(observation)
            paths = source.save_overlay(sequence, observation, adapter, decision)
            print(json.dumps({
                "cycle": sequence,
                "phase": decision.phase.value,
                "action": decision.action.value,
                "reason": decision.reason,
                "target_pose": (
                    None if decision.target_pose is None
                    else np.asarray(decision.target_pose).round(6).tolist()
                ),
                "diagnostics": dict(decision.diagnostics),
                "images": paths,
                "actual_ee_pose": observation.ee_pose.round(6).tolist(),
                "actual_joint_positions": observation.joint_positions.round(5).tolist(),
                "torque_log_only": observation.torque.round(4).tolist(),
            }), flush=True)
            if (
                decision.action == ServoAction.HOLD
                and "contact confirmation" in decision.reason
            ):
                print(
                    "CONTACT PAUSE: inspect the saved overlays, then rerun with "
                    "--auto-contact to descend and close.",
                    flush=True,
                )
                return
            execute(rpc, decision, args.dry_run)
            if decision.action == ServoAction.COMPLETE:
                return
            time.sleep(args.settle_s)
    finally:
        source.stop()


if __name__ == "__main__":
    main()
