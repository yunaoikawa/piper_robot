#!/usr/bin/env python3
"""
Bimanual teleop collector with a sequential named-step pipeline.

Camera identification:
  Run `python robot/camera_id.py` first to map physical cameras to labels.
  The mapping is saved to robot/camera_map.json and loaded automatically.

Controller mapping (NORMAL - no swap):
  Left controller (X/Y)  -> Left arm
  Right controller (A/B) -> Right arm

Episode logic (per step):
  ANY arm start -> recording begins
  ALL arms stop -> recording ends + save, then advance to the next step

Output layout (--mode):
  steps  (default) Each episode is saved into DATA_DIR/<step_name>/. Steps run
                   in a fixed order (STEPS). The loop ends after the final step
                   completes, or when Enter is pressed (any in-progress episode
                   is finished and saved first).
                   --start-step begins from a step (name or 0-based index).

  parity           Each episode is saved into DATA_DIR/type_even/ or
                   DATA_DIR/type_odd/ by episode-number parity, named
                   episode_NNNN_<timestamp>. The loop runs until Enter.

  --repeat-step    With --mode steps, keep collecting the selected
                   --start-step instead of advancing through STEPS.

  --task-sequence  Select a comma-separated subset/order of named tasks.
  --loop-sequence  Cycle that sequence until Enter instead of stopping after
                   one pass.
"""
import argparse
import atexit
import queue
import sys
import threading
import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import imageio
import mink
import numpy as np
import zmq
from loop_rate_limiters import RateLimiter
from record3d import Record3DStream

from robot.rpc import RPCClient
from robot.arm.startup import prepare_arms_for_manipulation
from robot.teleop.oculus_msgs import parse_controller_state
from robot.camera_id import load_camera_map
from rollout.calibration_keyboard_jog import load_torque_thresholds
from rollout.head_stream import HeadCameraStreamServer
from rollout.recovery_teleop_safety import (
    RecoveryTorqueGuard,
    extend_fallback_threshold_for_stationary_pose,
)
from rollout.safety import SafetyLayer

DEFAULT_VR_TCP_HOST = "192.168.1.106"
DEFAULT_VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"
CONTROL_FREQ = 30
DATA_DIR = Path("./teleop_demonstrations")
CAMERA_LABELS = ["head", "right", "left"]

STEPS = [
    "door_open", "petri2microscope", "petri2bench", "lid_open", "bottle_open",
    "pipette", "lid_close", "bottle_close", "petri2incubator", "door_close",
]

RecordingSample = namedtuple("RecordingSample", [
    "timestamp",
    "left_ee_pose", "right_ee_pose",
    "left_gripper", "right_gripper",
    "head_rgb", "head_depth", "head_rgb_ts",
    "left_wrist_rgb", "left_wrist_depth", "left_wrist_rgb_ts",
    "right_wrist_rgb", "right_wrist_depth", "right_wrist_rgb_ts",
])


def resolve_task_sequence(value):
    """Return the requested ordered task names, validating against STEPS."""
    if value is None:
        return list(STEPS)
    tasks = [item.strip() for item in value.split(",") if item.strip()]
    if not tasks:
        raise SystemExit("--task-sequence must contain at least one task name")
    unknown = [task for task in tasks if task not in STEPS]
    if unknown:
        raise SystemExit(
            f"Unknown task(s) in --task-sequence: {unknown}. Choose from {STEPS}"
        )
    if len(set(tasks)) != len(tasks):
        raise SystemExit("--task-sequence must not contain duplicate task names")
    return tasks


def resolve_start_step(value, steps=None):
    """Map a --start-step value (name or 0-based index) to a step index."""
    steps = STEPS if steps is None else steps
    if value is None:
        return 0
    if value.isdigit():
        i = int(value)
    elif value in steps:
        i = steps.index(value)
    else:
        raise SystemExit(
            f"Unknown start step '{value}'. Choose from {steps} or "
            f"0..{len(steps) - 1}"
        )
    if not (0 <= i < len(steps)):
        raise SystemExit(f"start-step index out of range 0..{len(steps) - 1}")
    return i


def advance_step_index(current, step_count, *, repeat_step, loop_sequence):
    """Return ``(next_index, finished)`` after a completed step episode."""
    if repeat_step:
        return current, False
    next_index = current + 1
    if next_index < step_count:
        return next_index, False
    if loop_sequence:
        return 0, False
    return next_index, True


class CameraStream:
    def __init__(self, device, index, label, stop_event):
        self.device = device
        self.index = index
        self.label = label
        self.stop_event = stop_event
        self.lock = threading.Lock()
        self.frame_event = threading.Event()
        self.session = None
        self.connected = False
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_ts = None

    def start(self):
        self.session = Record3DStream()
        self.session.on_new_frame = lambda: self.frame_event.set()
        self.session.on_stream_stopped = lambda: print(f"[{self.label}] Stream stopped")
        try:
            self.session.connect(self.device)
            self.connected = True
            print(f"[{self.label}] Connected (device {self.index})")
        except Exception as e:
            print(f"[{self.label}] Connect failed: {e}")

    def run(self):
        if not self.connected:
            return
        while not self.stop_event.is_set():
            self.frame_event.wait(timeout=0.1)
            if self.session is None:
                continue
            try:
                rgb = np.array(self.session.get_rgb_frame())
                try:
                    depth = np.array(self.session.get_depth_frame())
                except Exception:
                    depth = None
                ts = time.time()
                with self.lock:
                    self.latest_rgb = rgb
                    self.latest_depth = depth
                    self.latest_ts = ts
            except Exception:
                pass
            self.frame_event.clear()

    def get_latest(self):
        with self.lock:
            rgb = np.array(self.latest_rgb) if self.latest_rgb is not None else None
            depth = np.array(self.latest_depth) if self.latest_depth is not None else None
            ts = self.latest_ts
        return rgb, depth, ts


class VideoWriterSet:
    def __init__(self, base_path, fps=30):
        self.writers = {}
        self.base_path = base_path
        self.fps = fps
        self.frame_counts = {}

    def open(self, label):
        path = f"{self.base_path}_{label}.mp4"
        self.writers[label] = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=8)
        self.frame_counts[label] = 0
        return path

    def write_frame(self, label, frame):
        if label not in self.writers:
            return
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.writers[label].append_data(frame)
        self.frame_counts[label] += 1

    def close_all(self):
        for label, w in self.writers.items():
            try:
                w.close()
            except Exception as e:
                print(f"[VideoWriter] Error closing {label}: {e}")
        paths = {label: f"{self.base_path}_{label}.mp4" for label in self.writers}
        counts = dict(self.frame_counts)
        self.writers.clear()
        self.frame_counts.clear()
        return paths, counts


class MinimalTeleopCollector:
    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.save_dir = DATA_DIR
        if self.mode != "recovery":
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "parity":
            for sub in ("type_even", "type_odd"):
                (self.save_dir / sub).mkdir(parents=True, exist_ok=True)

        self.steps = args.task_sequence
        self.step_index = args.start_index
        self.repeat_step = args.repeat_step
        self.loop_sequence = args.loop_sequence
        self._quit_requested = False
        self._stopped = False

        self.start_teleop_left = False
        self.start_teleop_right = False
        self.H = mink.SE3.from_rotation(
            mink.SO3.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]))
        )
        self.X_Cinit_left = self.X_ee_init_left = None
        self.X_Cinit_right = self.X_ee_init_right = None

        self.robot = RPCClient("localhost", 8081)
        self.robot_rpc_lock = threading.Lock()
        with self.robot_rpc_lock:
            # Connect without moving first, then run the same explicit
            # machine-zero -> manipulation-home sequence as inference.
            self.robot.init(reset_arms=False)
            if self.mode != "recovery" and not args.attach_current:
                prepare_arms_for_manipulation(
                    self.robot,
                    context="teleop startup",
                )
            else:
                print("[INIT] Preserving the current arm poses (no machine-zero "
                      "or manipulation-home move).")

        self.recovery_safety = SafetyLayer.from_config(args.safety_config)
        self.recovery_torque_guard = None
        if self.mode == "recovery":
            thresholds, torque_provenance = load_torque_thresholds(
                args.torque_config,
                allow_symmetric_left_fallback=(
                    args.allow_symmetric_left_torque_fallback
                ),
            )
            fallback = torque_provenance.get("fallback")
            if fallback is not None:
                with self.robot_rpc_lock:
                    stationary_extension = (
                        extend_fallback_threshold_for_stationary_pose(
                            self.robot,
                            thresholds,
                            arm=fallback["arm"],
                        )
                    )
                torque_provenance["stationary_fallback_extension"] = (
                    stationary_extension
                )
                print(
                    "[RECOVERY] Audited stationary-pose extension for "
                    f"{fallback['arm']} fallback torque envelope; changed joints="
                    f"{stationary_extension['changed_joints']}",
                    flush=True,
                )
            self.recovery_torque_guard = RecoveryTorqueGuard(
                self.robot,
                thresholds,
                consecutive_samples=torque_provenance["consecutive_samples"],
                audit_path=args.recovery_audit_log,
                provenance=torque_provenance,
                enforce=getattr(
                    args,
                    "enforce_recovery_torque_stop",
                    False,
                ),
                **torque_provenance["recovery_teleop"],
            )
            print(
                "[RECOVERY] Torque observer active"
                + (
                    " with stop authority; "
                    if self.recovery_torque_guard.enforce
                    else " without stop authority; "
                )
                + f"audit={Path(args.recovery_audit_log).resolve()}",
                flush=True,
            )

        self.latest_controller_state = None
        self.controller_state_lock = threading.Lock()
        self.latest_left_ee_pose = self.latest_right_ee_pose = None
        self.robot_state_lock = threading.Lock()

        self.stop_event = threading.Event()

        self.cameras = {}
        self.camera_threads = []
        self._init_cameras()

        self.head_stream = None
        if not args.no_head_stream:
            self.head_stream = HeadCameraStreamServer(
                self._head_frame,
                host=args.head_stream_host,
                port=args.head_stream_port,
                token=args.head_stream_token,
                fps=args.head_stream_fps,
                status_provider=self._head_stream_status,
            )
            self.head_stream.start()

        self.is_recording = False
        self._recording_start_time = 0
        self.episode_count = 0
        self._current_step = None
        self.episode_data = None
        self.video_writers = None
        self.recording_queue = queue.Queue(maxsize=300)
        self.episode_lock = threading.Lock()

        self.oculus_thread = threading.Thread(target=self._oculus_thread, daemon=True)
        self.robot_state_thread = threading.Thread(target=self._robot_state_thread, daemon=True)
        self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self.display_thread = None
        if not args.no_display:
            self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.enter_thread = threading.Thread(target=self._enter_listener, daemon=True)
        self.oculus_thread.start()
        self.robot_state_thread.start()
        self.recording_thread.start()
        if self.display_thread:
            self.display_thread.start()
        self.enter_thread.start()

    def _head_frame(self):
        camera = self.cameras.get("head")
        if camera is None:
            return None
        rgb, _, _ = camera.get_latest()
        return rgb

    def _head_stream_status(self):
        if self.mode == "recovery":
            if (
                self.recovery_torque_guard is not None
                and self.recovery_torque_guard.latched
            ):
                stopped = "+".join(
                    arm.upper()
                    for arm in sorted(self.recovery_torque_guard.latched)
                )
                return f"RECOVERY TORQUE STOP {stopped}"
            arms = []
            if self.start_teleop_left:
                arms.append("LEFT")
            if self.start_teleop_right:
                arms.append("RIGHT")
            return "RECOVERY " + ("+".join(arms) if arms else "ARMED / HOLDING")
        if getattr(self, "is_recording", False):
            current_step = getattr(self, "_current_step", None)
            episode_count = getattr(self, "episode_count", 0)
            return f"REC {current_step or ('EPISODE ' + str(episode_count))}"
        return "LIVE / NOT RECORDING"

    def _step_subdir(self, step_index):
        d = self.save_dir / self.steps[step_index]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _episode_subdir(self, episode_number):
        d = self.save_dir / ("type_even" if episode_number % 2 == 0 else "type_odd")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _init_cameras(self):
        try:
            devs = Record3DStream.get_connected_devices()
        except Exception as e:
            print(f"[Cameras] Discovery failed: {e}")
            return
        print(f"[Cameras] {len(devs)} device(s) found")
        if not devs:
            print("[Cameras] No devices found.")
            return

        cam_map = load_camera_map()
        print(f"[Cameras] Using map: {cam_map}")

        for label in CAMERA_LABELS:
            idx = cam_map.get(label)
            if idx is None or idx >= len(devs):
                print(f"[Cameras] {label}: no valid device index in camera_map.json")
                continue
            stream = CameraStream(devs[idx], idx, label, self.stop_event)
            stream.start()
            if stream.connected:
                self.cameras[label] = stream
                t = threading.Thread(target=stream.run, daemon=True)
                t.start()
                self.camera_threads.append(t)
        print(f"[Cameras] Connected: {list(self.cameras.keys())}")

    def _enter_listener(self):
        while not self.stop_event.is_set() and not self._quit_requested:
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if line == "":  # EOF
                return
            print("[LOOP] Enter received. Finishing after the current episode.")
            self._quit_requested = True
            return

    def _oculus_thread(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 2000)
        if self.args.use_relay:
            host, port = self.args.relay_host, self.args.relay_port
            topic = self.args.relay_topic.encode("utf-8")
        else:
            host, port, topic = (
                self.args.quest_host,
                self.args.quest_port,
                VR_CONTROLLER_TOPIC,
            )
        endpoint = f"tcp://{host}:{port}"
        sock.connect(endpoint)
        sock.setsockopt(zmq.SUBSCRIBE, topic)
        time.sleep(0.3)
        last_msg = time.time()
        warned = False
        while not self.stop_event.is_set():
            try:
                parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                payload = parts[1] if len(parts) >= 2 else parts[0]
                state = parse_controller_state(payload.decode(errors="replace"))
                with self.controller_state_lock:
                    self.latest_controller_state = state
                last_msg = time.time()
                warned = False
            except zmq.Again:
                if (time.time() - last_msg) > 2.0 and not warned:
                    print(f"[VR] WARNING: no messages for >2s from {endpoint}")
                    warned = True
                time.sleep(0.005)
            except Exception as e:
                print(f"[VR] ERROR: {e}")
                time.sleep(0.2)
        sock.close(0)
        ctx.destroy(linger=0)

    def _robot_state_thread(self):
        while not self.stop_event.is_set():
            with self.robot_rpc_lock:
                left = self.robot.get_left_ee_pose()
                right = self.robot.get_right_ee_pose()
            with self.robot_state_lock:
                self.latest_left_ee_pose = left
                self.latest_right_ee_pose = right
            time.sleep(0.01)

    def _reset_recovery_torque_guard(self, arm):
        """Reset an engagement baseline without racing the state reader.

        ``RecoveryTorqueGuard`` and ``_robot_state_thread`` intentionally share
        the robot RPC client.  Its pyzmq REQ socket permits only one in-flight
        request, so every access must use the same lock.
        """
        with self.robot_rpc_lock:
            self.recovery_torque_guard.reset(arm)

    def _recording_worker(self):
        while not self.stop_event.is_set():
            try:
                sample = self.recording_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self.episode_lock:
                if self.episode_data is None:
                    self.recording_queue.task_done()
                    continue

                self.episode_data["timestamps"].append(sample.timestamp)
                self.episode_data["left_ee_pos"].append(sample.left_ee_pose.translation())
                self.episode_data["left_ee_quat"].append(sample.left_ee_pose.rotation().wxyz)
                self.episode_data["left_gripper"].append(sample.left_gripper)
                self.episode_data["right_ee_pos"].append(sample.right_ee_pose.translation())
                self.episode_data["right_ee_quat"].append(sample.right_ee_pose.rotation().wxyz)
                self.episode_data["right_gripper"].append(sample.right_gripper)

                def _rotate(frame):
                    if frame is None:
                        return np.zeros((480, 640, 3), dtype=np.uint8)
                    return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                def _rotate_depth(frame):
                    if frame is None:
                        return np.zeros((480, 640), dtype=np.float32)
                    try:
                        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    except Exception:
                        return np.zeros((480, 640), dtype=np.float32)

                head_depth = _rotate_depth(sample.head_depth)
                self.episode_data["depth_frames"].append(head_depth)
                self.episode_data["rgb_frame_timestamps"].append(
                    sample.head_rgb_ts if sample.head_rgb_ts is not None else sample.timestamp
                )
                if self.video_writers:
                    self.video_writers.write_frame("head", _rotate(sample.head_rgb))
                    self.video_writers.write_frame("left", _rotate(sample.left_wrist_rgb))
                    self.video_writers.write_frame("right", _rotate(sample.right_wrist_rgb))

            self.recording_queue.task_done()

    def _new_episode_data(self):
        return {
            "timestamps": [],
            "left_ee_pos": [], "left_ee_quat": [], "left_gripper": [],
            "right_ee_pos": [], "right_ee_quat": [], "right_gripper": [],
            "depth_frames": [], "rgb_frame_timestamps": [],
        }

    def _start_episode(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.mode == "parity":
            step = None
            name = f"episode_{self.episode_count:04d}_{ts}"
            episode_dir = self._episode_subdir(self.episode_count)
            print(f"[SAVE] Episode {self.episode_count} -> {episode_dir}")
        else:
            step = self.steps[self.step_index]
            name = f"{step}_{ts}"
            episode_dir = self._step_subdir(self.step_index)
            print(f"[SAVE] Step {self.step_index + 1}/{len(self.steps)} ({step}) -> {episode_dir}")
        base_path = str(episode_dir / name)

        with self.episode_lock:
            self.episode_data = self._new_episode_data()
            self.video_writers = VideoWriterSet(base_path, fps=CONTROL_FREQ)
            for label in CAMERA_LABELS:
                self.video_writers.open(label)
            self._current_base_path = base_path
            self._current_step = step

        self.is_recording = True
        self._recording_start_time = time.time()
        if self.mode == "parity":
            print(f"\n=== RECORDING STARTED (episode {self.episode_count}) ===\n")
        else:
            print(f"\n=== RECORDING STARTED (step {self.step_index + 1}/{len(self.steps)}: {step}) ===\n")

    def _drain_queue(self):
        try:
            self.recording_queue.join()
        except Exception:
            pass

    def _end_episode_and_save(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self._drain_queue()

        with self.episode_lock:
            if self.episode_data is None or len(self.episode_data["timestamps"]) == 0:
                print("[SAVE] Episode had 0 samples; nothing saved.")
                if self.video_writers:
                    self.video_writers.close_all()
                self.episode_data = None
                self.video_writers = None
                return

            n = len(self.episode_data["timestamps"])
            video_paths, video_counts = {}, {}
            if self.video_writers:
                video_paths, video_counts = self.video_writers.close_all()
                self.video_writers = None

            h5_path = f"{self._current_base_path}.hdf5"
            try:
                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("timestamps", data=np.array(self.episode_data["timestamps"]))
                    f.create_dataset("left_ee_pos", data=np.array(self.episode_data["left_ee_pos"]))
                    f.create_dataset("left_ee_quat", data=np.array(self.episode_data["left_ee_quat"]))
                    f.create_dataset("left_gripper", data=np.array(self.episode_data["left_gripper"]))
                    f.create_dataset("right_ee_pos", data=np.array(self.episode_data["right_ee_pos"]))
                    f.create_dataset("right_ee_quat", data=np.array(self.episode_data["right_ee_quat"]))
                    f.create_dataset("right_gripper", data=np.array(self.episode_data["right_gripper"]))
                    f.create_dataset("rgb_frame_timestamps",
                                     data=np.array(self.episode_data["rgb_frame_timestamps"]))
                    depth_arr = np.array(self.episode_data["depth_frames"], dtype=np.float32)
                    f.create_dataset("depth_frames", data=depth_arr, compression="gzip")
                    f.attrs["num_samples"] = n
                    if self.mode == "parity":
                        f.attrs["episode_number"] = self.episode_count
                    else:
                        f.attrs["step_name"] = self._current_step
                        f.attrs["step_index"] = self.step_index
                    f.attrs["control_frequency_hz"] = CONTROL_FREQ
            except Exception as e:
                print(f"[SAVE] HDF5 error: {e}")

            self.episode_data = None

        if self.mode == "parity":
            print(f"\n=== RECORDING STOPPED (episode {self.episode_count}) ===")
        else:
            print(f"\n=== RECORDING STOPPED (step {self.step_index + 1}/{len(self.steps)}: {self._current_step}) ===")
        print(f"[SAVE] HDF5 : {h5_path} ({n} samples)")
        for label, path in video_paths.items():
            print(f"[SAVE] MP4  : {path} ({video_counts.get(label, 0)} frames)")
        print()

        self.episode_count += 1
        if self.mode == "parity":
            nxt = "type_even" if self.episode_count % 2 == 0 else "type_odd"
            print(f"[LOOP] Next episode {self.episode_count} -> {nxt}. "
                  f"Trigger a controller to record; press Enter to quit.")
            return

        if self.repeat_step:
            step = self.steps[self.step_index]
            print(
                f"[LOOP] Repeating step {self.step_index + 1}/{len(self.steps)}: "
                f"{step}. Trigger a controller to record; press Enter to quit."
            )
            return

        self.step_index, finished = advance_step_index(
            self.step_index,
            len(self.steps),
            repeat_step=False,
            loop_sequence=self.loop_sequence,
        )
        if finished:
            print("[LOOP] All steps complete. Finishing.")
            self.stop_event.set()
        else:
            nxt = self.steps[self.step_index]
            cycle = " (new cycle)" if self.step_index == 0 and self.loop_sequence else ""
            print(f"[LOOP] Next step {self.step_index + 1}/{len(self.steps)}: {nxt}. "
                  f"Trigger a controller to record; press Enter to quit.{cycle}")

    def _display_loop(self):
        windows = {}
        while not self.stop_event.is_set():
            for label in CAMERA_LABELS:
                if label in self.cameras:
                    rgb, _, _ = self.cameras[label].get_latest()
                    if rgb is not None and rgb.size > 0:
                        frame = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        if label not in windows:
                            cv2.namedWindow(label, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(label, 640, 480)
                            windows[label] = True
                        if self.is_recording:
                            elapsed = time.time() - self._recording_start_time
                            step = self._current_step or ""
                            cv2.putText(frame, f"REC {step} {elapsed:.1f}s", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow(label, frame)
                else:
                    if label not in windows:
                        cv2.namedWindow(label, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(label, 640, 480)
                        windows[label] = True
                    black = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black, f"Waiting for {label}...", (100, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.imshow(label, black)
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                self._quit_requested = True
        cv2.destroyAllWindows()

    def control_loop(self):
        rate = RateLimiter(CONTROL_FREQ)
        prev_any_teleop = False
        prev_all_stopped = True

        if self.mode == "recovery":
            print("[LOOP] Mode: remote recovery (no recording, no automatic home)")
            print("[LOOP] X/A engages an arm; Y/B disengages it and holds position.")
            print(f"[LOOP] Quest messages older than {self.args.vr_timeout:.2f}s disengage both arms.")
        elif self.mode == "parity":
            print("[LOOP] Mode: parity (even -> type_even, odd -> type_odd)")
            print(f"[LOOP] Starting at episode {self.episode_count}")
            print("[LOOP] Trigger a controller (X/Y or A/B) to record. Enter to quit.")
        else:
            print(f"[LOOP] Steps: {self.steps}")
            print(f"[LOOP] Starting at step {self.step_index + 1}/{len(self.steps)}: {self.steps[self.step_index]}")
            if self.repeat_step:
                print(f"[LOOP] Repeat enabled: {self.steps[self.step_index]} only")
            elif self.loop_sequence:
                print("[LOOP] Sequence loop enabled; collection continues until Enter.")
            print("[LOOP] Trigger a controller (X/Y or A/B) to record each step. Enter to quit early.")

        while not self.stop_event.is_set():
            if self._quit_requested and not self.is_recording:
                break

            with self.controller_state_lock:
                cs = self.latest_controller_state
            if cs is None:
                rate.sleep()
                continue

            if (self.mode == "recovery" and
                    (time.time() - cs.created_timestamp) > self.args.vr_timeout):
                if self.start_teleop_left or self.start_teleop_right:
                    print("[VR] Controller connection stale; disengaging both arms and holding.")
                    self.start_teleop_left = False
                    self.start_teleop_right = False
                    self.recovery_safety.reset()
                prev_any_teleop = False
                prev_all_stopped = True
                rate.sleep()
                continue

            with self.robot_state_lock:
                eeL = self.latest_left_ee_pose
                eeR = self.latest_right_ee_pose
            if eeL is None or eeR is None:
                rate.sleep()
                continue

            if cs.left_x and not self.start_teleop_left:
                self.X_Cinit_left = cs.left_SE3
                self.X_ee_init_left = eeL
                self.start_teleop_left = True
                if self.mode == "recovery":
                    self.recovery_safety.reset("left")
                    self._reset_recovery_torque_guard("left")
            if cs.left_y:
                self.start_teleop_left = False

            if cs.right_a and not self.start_teleop_right:
                self.X_Cinit_right = cs.right_SE3
                self.X_ee_init_right = eeR
                self.start_teleop_right = True
                if self.mode == "recovery":
                    self.recovery_safety.reset("right")
                    self._reset_recovery_torque_guard("right")
            if cs.right_b:
                self.start_teleop_right = False

            any_teleop = self.start_teleop_left or self.start_teleop_right
            all_stopped = (not self.start_teleop_left) and (not self.start_teleop_right)

            # Do not begin a new step episode once a quit has been requested.
            if self.mode == "recovery":
                if any_teleop and not prev_any_teleop:
                    print("[RECOVERY] Control engaged.")
                if all_stopped and not prev_all_stopped:
                    print("[RECOVERY] Control disengaged; holding current poses.")
                    self.recovery_safety.reset()
            else:
                if any_teleop and not prev_any_teleop and not self._quit_requested:
                    self._start_episode()
                if all_stopped and not prev_all_stopped:
                    with self.robot_rpc_lock:
                        self.robot.home_left_arm()
                        self.robot.home_right_arm()
                    self._end_episode_and_save()

            prev_any_teleop = any_teleop
            prev_all_stopped = all_stopped

            with self.robot_rpc_lock:
                if self.start_teleop_left and self.X_Cinit_left is not None and self.X_ee_init_left is not None:
                    torque_ok = (
                        self.mode != "recovery"
                        or self.recovery_torque_guard.check("left")
                    )
                    if not torque_ok:
                        self.start_teleop_left = False
                        print(
                            "[RECOVERY] LEFT torque stop latched; "
                            "measured joint hold sent.",
                            flush=True,
                        )
                    else:
                        Xd = self.X_Cinit_left.inverse().multiply(cs.left_SE3)
                        Rd = self.H.inverse() @ Xd @ self.H
                        p = self.X_ee_init_left.translation() + Rd.translation()
                        R = Rd.rotation() @ self.X_ee_init_left.rotation()
                        gr = 1.0 if cs.left_index_trigger < 0.5 else 0.0
                        p = self.recovery_safety.check("left", p) if self.mode == "recovery" else p
                        if p is not None:
                            self.robot.set_left_ee_target(
                                ee_target=mink.SE3(np.concatenate([R.wxyz, p])),
                                gripper_target=gr, preview_time=0.05,
                            )
                        elif self.mode == "recovery":
                            # A discontinuous Quest target is rejected for this
                            # frame.  Drop engagement so the next held X frame
                            # re-anchors controller and robot poses instead of
                            # comparing forever against the stale target.
                            self.start_teleop_left = False
                            self.X_Cinit_left = self.X_ee_init_left = None
                            print(
                                "[RECOVERY] LEFT target rejected; "
                                "re-anchoring without motion.",
                                flush=True,
                            )
                if self.start_teleop_right and self.X_Cinit_right is not None and self.X_ee_init_right is not None:
                    torque_ok = (
                        self.mode != "recovery"
                        or self.recovery_torque_guard.check("right")
                    )
                    if not torque_ok:
                        self.start_teleop_right = False
                        print(
                            "[RECOVERY] RIGHT torque stop latched; "
                            "measured joint hold sent.",
                            flush=True,
                        )
                    else:
                        Xd = self.X_Cinit_right.inverse().multiply(cs.right_SE3)
                        Rd = self.H.inverse() @ Xd @ self.H
                        p = self.X_ee_init_right.translation() + Rd.translation()
                        R = Rd.rotation() @ self.X_ee_init_right.rotation()
                        gr = 1.0 if cs.right_index_trigger < 0.5 else 0.0
                        p = self.recovery_safety.check("right", p) if self.mode == "recovery" else p
                        if p is not None:
                            self.robot.set_right_ee_target(
                                ee_target=mink.SE3(np.concatenate([R.wxyz, p])),
                                gripper_target=gr, preview_time=0.05,
                            )
                        elif self.mode == "recovery":
                            self.start_teleop_right = False
                            self.X_Cinit_right = self.X_ee_init_right = None
                            print(
                                "[RECOVERY] RIGHT target rejected; "
                                "re-anchoring without motion.",
                                flush=True,
                            )

            if self.is_recording:
                now = time.time()
                with self.robot_state_lock:
                    ee_left = self.latest_left_ee_pose
                    ee_right = self.latest_right_ee_pose

                def _get_cam(label):
                    if label in self.cameras:
                        return self.cameras[label].get_latest()
                    return None, None, None

                h_rgb, h_depth, h_ts = _get_cam("head")
                l_rgb, l_depth, l_ts = _get_cam("left")
                r_rgb, r_depth, r_ts = _get_cam("right")

                sample = RecordingSample(
                    timestamp=now,
                    left_ee_pose=ee_left, right_ee_pose=ee_right,
                    left_gripper=1.0 if cs.left_index_trigger < 0.5 else 0.0,
                    right_gripper=1.0 if cs.right_index_trigger < 0.5 else 0.0,
                    head_rgb=h_rgb, head_depth=h_depth, head_rgb_ts=h_ts,
                    left_wrist_rgb=l_rgb, left_wrist_depth=l_depth, left_wrist_rgb_ts=l_ts,
                    right_wrist_rgb=r_rgb, right_wrist_depth=r_depth, right_wrist_rgb_ts=r_ts,
                )
                try:
                    self.recording_queue.put_nowait(sample)
                except queue.Full:
                    pass

            rate.sleep()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        if self.is_recording:
            self._end_episode_and_save()
        self.stop_event.set()
        if self.head_stream:
            self.head_stream.stop()
        for t in [self.oculus_thread, self.robot_state_thread, self.recording_thread,
                  self.display_thread] + self.camera_threads:
            if t is None:
                continue
            try:
                t.join(timeout=1.0)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-relay", action="store_true")
    ap.add_argument(
        "--quest-host",
        default=DEFAULT_VR_TCP_HOST,
        help=(
            "Direct Quest controller host when --use-relay is not set "
            f"(default: {DEFAULT_VR_TCP_HOST})."
        ),
    )
    ap.add_argument(
        "--quest-port",
        type=int,
        default=DEFAULT_VR_TCP_PORT,
        help=f"Direct Quest controller port (default: {DEFAULT_VR_TCP_PORT}).",
    )
    ap.add_argument("--relay-host", default="100.125.255.41")
    ap.add_argument("--relay-port", type=int, default=6006)
    ap.add_argument("--relay-topic", default="oculus_controller")
    ap.add_argument("--mode", choices=("steps", "parity", "recovery"), default="steps",
                    help="steps: save into DATA_DIR/<step_name>/ following STEPS in order. "
                         "parity: save by episode parity. recovery: teleoperate without "
                         "recording or automatic homing.")
    ap.add_argument("--start-step", default=None,
                    help=f"Name or 0-based index of the step to start from. Steps: {STEPS}")
    ap.add_argument(
        "--task-sequence",
        default=None,
        help=(
            "Comma-separated task names to record in order. Defaults to all "
            f"tasks: {STEPS}"
        ),
    )
    ap.add_argument(
        "--loop-sequence",
        action="store_true",
        help="Repeat --task-sequence indefinitely instead of stopping after one pass.",
    )
    ap.add_argument(
        "--repeat-step",
        action="store_true",
        help="Repeat --start-step indefinitely instead of advancing to the next step.",
    )
    ap.add_argument("--vr-timeout", type=float, default=0.75,
                    help="Disengage teleop when Quest messages are this old (default: 0.75s).")
    ap.add_argument("--safety-config", default=None,
                    help="Optional keep-out-zone JSON for recovery mode (see rollout/safety.py).")
    ap.add_argument("--torque-config", default="src/configs/pasteur_lid_torque.json",
                    help="Joint-torque thresholds for recovery mode.")
    ap.add_argument("--allow-symmetric-left-torque-fallback", action="store_true",
                    help="Explicitly reuse right-arm torque thresholds for the identical left Piper.")
    ap.add_argument(
        "--enforce-recovery-torque-stop",
        action="store_true",
        help=(
            "Let torque telemetry latch a recovery-teleop hold. Disabled by "
            "default because unmodeled pose load caused false stops."
        ),
    )
    ap.add_argument(
        "--recovery-audit-log",
        default=(
            "/var/tmp/piper-recovery-teleop/"
            + datetime.now().strftime("torque-%Y%m%dT%H%M%S.jsonl")
        ),
        help="Append-only recovery torque audit log.",
    )
    ap.add_argument("--no-display", action="store_true",
                    help="Disable local OpenCV windows (useful on a headless robot PC).")
    ap.add_argument(
        "--attach-current",
        action="store_true",
        help=(
            "Preserve current arm poses at startup; skip both true machine-zero "
            "and manipulation-home moves."
        ),
    )
    ap.add_argument("--no-head-stream", action="store_true",
                    help="Disable the phone/Quest head-camera web stream.")
    ap.add_argument("--head-stream-host", default="0.0.0.0",
                    help="Head stream listen address (default: all interfaces).")
    ap.add_argument("--head-stream-port", type=int, default=8080,
                    help="Head stream HTTP port (default: 8080).")
    ap.add_argument("--head-stream-token", default=None,
                    help="Optional token required as ?token=... in the stream URL.")
    ap.add_argument("--head-stream-fps", type=float, default=15.0,
                    help="Maximum remote head stream rate (default: 15 FPS).")
    args = ap.parse_args()
    if args.use_relay and not args.relay_host:
        raise SystemExit("ERROR: --relay-host is required when --use-relay is set.")
    if not args.use_relay and not args.quest_host:
        raise SystemExit("ERROR: --quest-host is required for direct Quest control.")
    if args.vr_timeout <= 0:
        raise SystemExit("ERROR: --vr-timeout must be positive.")
    if args.mode in ("parity", "recovery"):
        if args.start_step is not None:
            raise SystemExit("ERROR: --start-step only applies to --mode steps.")
        if args.task_sequence is not None:
            raise SystemExit("ERROR: --task-sequence only applies to --mode steps.")
        if args.loop_sequence:
            raise SystemExit("ERROR: --loop-sequence only applies to --mode steps.")
        if args.repeat_step:
            raise SystemExit("ERROR: --repeat-step only applies to --mode steps.")
        args.start_index = 0
        args.task_sequence = list(STEPS)
    else:
        args.task_sequence = resolve_task_sequence(args.task_sequence)
        if args.repeat_step and args.loop_sequence:
            raise SystemExit("ERROR: --repeat-step and --loop-sequence are exclusive.")
        args.start_index = resolve_start_step(args.start_step, args.task_sequence)

    collector = MinimalTeleopCollector(args)
    atexit.register(collector.stop)
    try:
        collector.control_loop()
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
