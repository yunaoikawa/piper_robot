#!/usr/bin/env python3
"""
Bimanual teleop collector matching the original rollout/recorder.py format.

Key design decisions:
- MP4 videos written with streaming imageio.get_writer() (low memory)
- 3 separate videos per episode: _head.mp4, _left.mp4, _right.mp4
- RGB frames NOT stored in HDF5 (only depth + poses)
- HDF5 fields match what convert_to_lerobot.py expects

Controller mapping (NORMAL — no swap):
  Left controller (X/Y)  → Left arm
  Right controller (A/B) → Right arm

Episode logic:
  ANY arm start → recording begins
  ALL arms stop → recording ends + save

Output directory logic:
  Even episode numbers → DATA_DIR/type_even/
  Odd  episode numbers → DATA_DIR/type_odd/
"""
import argparse
import atexit
import queue
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
from robot.teleop.oculus_msgs import parse_controller_state

# =========================
# Configuration
# =========================
VR_TCP_HOST = "192.168.1.36"
VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"
CONTROL_FREQ = 30
DATA_DIR = Path("./teleop_demonstrations")

CAMERA_LABELS = ["head", "right", "left"]

RecordingSample = namedtuple("RecordingSample", [
    "timestamp",
    "left_ee_pose",
    "right_ee_pose",
    "left_gripper",
    "right_gripper",
    "head_rgb", "head_depth", "head_rgb_ts",
    "left_wrist_rgb", "left_wrist_depth", "left_wrist_rgb_ts",
    "right_wrist_rgb", "right_wrist_depth", "right_wrist_rgb_ts",
])


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
        self.writers[label] = imageio.get_writer(
            path, fps=self.fps, codec="libx264", quality=8,
        )
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
        self.save_dir = DATA_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Prepare even/odd subdirectories up front
        (self.save_dir / "type_even").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "type_odd").mkdir(parents=True, exist_ok=True)

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
            self.robot.init()
            self.robot.home_left_arm()
            self.robot.home_right_arm()

        self.latest_controller_state = None
        self.controller_state_lock = threading.Lock()
        self.latest_left_ee_pose = self.latest_right_ee_pose = None
        self.robot_state_lock = threading.Lock()

        self.stop_event = threading.Event()

        self.cameras = {}
        self.camera_threads = []
        self._init_cameras()

        self.is_recording = False
        self.episode_count = 0
        self.episode_data = None
        self.video_writers = None
        self.recording_queue = queue.Queue(maxsize=300)
        self.episode_lock = threading.Lock()

        self.oculus_thread = threading.Thread(target=self._oculus_thread, daemon=True)
        self.robot_state_thread = threading.Thread(target=self._robot_state_thread, daemon=True)
        self.recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self.oculus_thread.start()
        self.robot_state_thread.start()
        self.recording_thread.start()

    def _episode_subdir(self, episode_number: int) -> Path:
        """Return type_even or type_odd subdir based on episode number parity."""
        subdir = "type_even" if episode_number % 2 == 0 else "type_odd"
        return self.save_dir / subdir

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
        for i, dev in enumerate(devs):
            if i >= len(CAMERA_LABELS):
                break
            label = CAMERA_LABELS[i]
            stream = CameraStream(dev, i, label, self.stop_event)
            stream.start()
            if stream.connected:
                self.cameras[label] = stream
                t = threading.Thread(target=stream.run, daemon=True)
                t.start()
                self.camera_threads.append(t)
        print(f"[Cameras] Connected: {list(self.cameras.keys())}")

    def _oculus_thread(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVHWM, 2000)
        if self.args.use_relay:
            host, port = self.args.relay_host, self.args.relay_port
            topic = self.args.relay_topic.encode("utf-8")
        else:
            host, port, topic = VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC
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

                left_pos = sample.left_ee_pose.translation()
                left_quat = sample.left_ee_pose.rotation().wxyz
                right_pos = sample.right_ee_pose.translation()
                right_quat = sample.right_ee_pose.rotation().wxyz

                self.episode_data["timestamps"].append(sample.timestamp)
                self.episode_data["left_ee_pos"].append(left_pos)
                self.episode_data["left_ee_quat"].append(left_quat)
                self.episode_data["left_gripper"].append(sample.left_gripper)
                self.episode_data["right_ee_pos"].append(right_pos)
                self.episode_data["right_ee_quat"].append(right_quat)
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

                head_rgb = _rotate(sample.head_rgb)
                head_depth = _rotate_depth(sample.head_depth)
                self.episode_data["depth_frames"].append(head_depth)
                self.episode_data["rgb_frame_timestamps"].append(
                    sample.head_rgb_ts if sample.head_rgb_ts is not None else sample.timestamp
                )
                if self.video_writers:
                    self.video_writers.write_frame("head", head_rgb)
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
        name = f"episode_{self.episode_count:04d}_{ts}"

        # Route to type_even or type_odd based on episode number parity
        episode_dir = self._episode_subdir(self.episode_count)
        base_path = str(episode_dir / name)
        print(f"[SAVE] Output dir: {episode_dir}")

        with self.episode_lock:
            self.episode_data = self._new_episode_data()
            self.video_writers = VideoWriterSet(base_path, fps=CONTROL_FREQ)
            for label in CAMERA_LABELS:
                self.video_writers.open(label)
            self._current_base_path = base_path

        self.is_recording = True
        print(f"\n=== RECORDING STARTED (episode {self.episode_count}) ===\n")

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
                    f.attrs["episode_number"] = self.episode_count
                    f.attrs["control_frequency_hz"] = CONTROL_FREQ
            except Exception as e:
                print(f"[SAVE] HDF5 error: {e}")

            self.episode_data = None

        print(f"\n=== RECORDING STOPPED (episode {self.episode_count}) ===")
        print(f"[SAVE] HDF5 : {h5_path} ({n} samples)")
        for label, path in video_paths.items():
            print(f"[SAVE] MP4  : {path} ({video_counts.get(label, 0)} frames)")
        print()
        self.episode_count += 1

    # === Control loop (NORMAL: left ctrl→left arm, right ctrl→right arm) ===
    def control_loop(self):
        rate = RateLimiter(CONTROL_FREQ)
        prev_any_teleop = False
        prev_all_stopped = True

        while not self.stop_event.is_set():
            with self.controller_state_lock:
                cs = self.latest_controller_state
            if cs is None:
                rate.sleep()
                continue

            with self.robot_state_lock:
                eeL = self.latest_left_ee_pose
                eeR = self.latest_right_ee_pose
            if eeL is None or eeR is None:
                rate.sleep()
                continue

            # --- Button handling (NORMAL) ---
            if cs.left_x:
                self.X_Cinit_left = cs.left_SE3
                self.X_ee_init_left = eeL
                self.start_teleop_left = True
            if cs.left_y:
                self.start_teleop_left = False

            if cs.right_a:
                self.X_Cinit_right = cs.right_SE3
                self.X_ee_init_right = eeR
                self.start_teleop_right = True
            if cs.right_b:
                self.start_teleop_right = False

            any_teleop = self.start_teleop_left or self.start_teleop_right
            all_stopped = (not self.start_teleop_left) and (not self.start_teleop_right)

            if any_teleop and not prev_any_teleop:
                self._start_episode()
            if all_stopped and not prev_all_stopped:
                with self.robot_rpc_lock:
                    self.robot.home_left_arm()
                    self.robot.home_right_arm()
                self._end_episode_and_save()

            prev_any_teleop = any_teleop
            prev_all_stopped = all_stopped

            # --- Compute + send commands (NORMAL) ---
            with self.robot_rpc_lock:
                if self.start_teleop_left and self.X_Cinit_left is not None and self.X_ee_init_left is not None:
                    Xd = self.X_Cinit_left.inverse().multiply(cs.left_SE3)
                    Rd = self.H.inverse() @ Xd @ self.H
                    p = self.X_ee_init_left.translation() + Rd.translation()
                    R = Rd.rotation() @ self.X_ee_init_left.rotation()
                    gr = 1.0 if cs.left_index_trigger < 0.5 else 0.0
                    self.robot.set_left_ee_target(
                        ee_target=mink.SE3(np.concatenate([R.wxyz, p])),
                        gripper_target=gr, preview_time=0.05,
                    )
                if self.start_teleop_right and self.X_Cinit_right is not None and self.X_ee_init_right is not None:
                    Xd = self.X_Cinit_right.inverse().multiply(cs.right_SE3)
                    Rd = self.H.inverse() @ Xd @ self.H
                    p = self.X_ee_init_right.translation() + Rd.translation()
                    R = Rd.rotation() @ self.X_ee_init_right.rotation()
                    gr = 1.0 if cs.right_index_trigger < 0.5 else 0.0
                    self.robot.set_right_ee_target(
                        ee_target=mink.SE3(np.concatenate([R.wxyz, p])),
                        gripper_target=gr, preview_time=0.05,
                    )

            # --- Record sample (NORMAL) ---
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
                    left_ee_pose=ee_left,
                    right_ee_pose=ee_right,
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
        if self.is_recording:
            self._end_episode_and_save()
        self.stop_event.set()
        for t in [self.oculus_thread, self.robot_state_thread, self.recording_thread] + self.camera_threads:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-relay", action="store_true")
    ap.add_argument("--relay-host", default="100.125.255.41")
    ap.add_argument("--relay-port", type=int, default=6006)
    ap.add_argument("--relay-topic", default="oculus_controller")
    args = ap.parse_args()
    if args.use_relay and not args.relay_host:
        raise SystemExit("ERROR: --relay-host is required when --use-relay is set.")
    collector = MinimalTeleopCollector(args)
    atexit.register(collector.stop)
    collector.control_loop()


if __name__ == "__main__":
    main()