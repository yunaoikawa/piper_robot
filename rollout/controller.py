"""Main policy controller orchestrating robot control and data collection."""

import zmq
import time
import mink
import numpy as np
import threading
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from robot.rpc import RPCClient
from loop_rate_limiters import RateLimiter

from .camera import CameraFeedManager, USBWristCameraFeedManager
from .recorder import DataRecorder, RecordingSample
from .episode import EpisodeManager
from .keyboard import KeyboardController
from .manipulability import ManipulabilityCalculator


# Default data directory
DATA_DIR = Path("./your_save_dir_here")

# Default task description (override with --task flag)
DEFAULT_TASK = "put the flask in the incubator"


def quat_to_r6(quat, batched=False):
    """Convert quaternion to 6D rotation representation."""
    rot_mat = R.from_quat(quat, scalar_first=True).as_matrix()
    if batched:
        a1, a2 = rot_mat[:, :, 0], rot_mat[:, :, 1]
        return np.concatenate((a1, a2), axis=-1)
    else:
        a1, a2 = rot_mat[:, 0], rot_mat[:, 1]
        return np.concatenate((a1, a2))


class PolicyController:
    """Main controller for robot policy execution and data collection."""

    def __init__(self, hpc_host="192.168.1.50", obs_port=5555, action_port=5556,
                 enable_recording=False, save_dir=None, autonomous_mode=False,
                 episode_timeout=60.0, manipulability_threshold=0.05,
                 task=DEFAULT_TASK):
        self.stop_event = threading.Event()
        self.policy_active = False
        self.task = task

        # Robot connection
        self.obs_cone_e = RPCClient("localhost", 8081)
        self.obs_cone_e.init()
        self.obs_rpc_lock = threading.Lock()

        self.cone_e = RPCClient("localhost", 8081)
        self.cone_e.init()
        self.cone_e.home_left_arm()
        self.cone_e.home_right_arm()

        # ZeroMQ setup
        self._setup_zmq(hpc_host, obs_port, action_port)

        # Transform (identity for this setup)
        self.H = mink.SE3.from_rotation(
            mink.SO3.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        )

        # Gripper state tracking
        self.last_left_gripper = 1.0
        self.last_right_gripper = 1.0
        self.last_left_gripper_binary = 1.0
        self.last_right_gripper_binary = 1.0

        self.starting_pose_left = None
        self.starting_pose_right = None

        self.stats = {
            'observations_sent': 0,
            'actions_received': 0,
            'errors': 0,
            'buffer_wraps': 0
        }

        self.test_qpos = None
        save_path = Path(save_dir) if save_dir else DATA_DIR

        # Head camera (device 0)
        self.camera = CameraFeedManager(self.stop_event)
        self.camera.autonomous_mode = autonomous_mode
        self.camera.start()

        # Left wrist camera - disabled (no physical camera on left arm)
        self.left_wrist_camera = None

        # Right wrist camera (device 1)
        self.right_wrist_camera = USBWristCameraFeedManager(
            self.stop_event, device_index=1, label="right wrist"
        )
        self.right_wrist_camera.start()

        # Link wrist camera to head camera display (single display thread)
        self.camera.wrist_camera = self.right_wrist_camera

        # Data recorder (optional)
        self.recorder = DataRecorder(save_path, self.stop_event) if enable_recording else None

        # Episode manager
        self.episode_manager = EpisodeManager(
            recorder=self.recorder,
            robot_rpc=self.cone_e,
            control_socket=self.control_socket,
            autonomous_mode=autonomous_mode,
            episode_timeout=episode_timeout,
            manipulability_threshold=manipulability_threshold
        )

        # Keyboard controller
        self.keyboard = KeyboardController(
            self.stop_event, self.episode_manager, enable_recording, autonomous_mode
        )
        self.keyboard.start()

        # Manipulability calculator
        self.manipulability_calc = ManipulabilityCalculator(self.obs_cone_e, self.obs_rpc_lock)

        # Start background threads
        self.obs_thread = threading.Thread(target=self._observation_publishing_loop, daemon=True)
        self.obs_thread.start()

    def _setup_zmq(self, hpc_host, obs_port, action_port):
        self.zmq_context = zmq.Context()

        self.obs_socket = self.zmq_context.socket(zmq.PUB)
        obs_address = f"tcp://{hpc_host}:{obs_port}"
        self.obs_socket.connect(obs_address)
        print(f"Publishing observations to {obs_address}")

        self.action_socket = self.zmq_context.socket(zmq.REQ)
        self.action_socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.action_socket.setsockopt(zmq.SNDTIMEO, 2000)
        self.action_socket.setsockopt(zmq.LINGER, 0)
        action_address = f"tcp://{hpc_host}:{action_port}"
        self.action_socket.connect(action_address)
        print(f"Requesting actions from {action_address}")

        self.control_port = action_port + 1
        self.control_socket = self.zmq_context.socket(zmq.REQ)
        self.control_socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.control_socket.setsockopt(zmq.SNDTIMEO, 2000)
        self.control_socket.setsockopt(zmq.LINGER, 0)
        control_address = f"tcp://{hpc_host}:{self.control_port}"
        self.control_socket.connect(control_address)
        print(f"Sending control commands to {control_address}")

        time.sleep(0.5)

    def get_observation(self):
        timestamp = time.time()

        with self.obs_rpc_lock:
            ee_pose_left = self.obs_cone_e.get_left_ee_pose()
            ee_pose_right = self.obs_cone_e.get_right_ee_pose()
            left_gripper = self.obs_cone_e.get_left_gripper_exact()
            right_gripper = self.obs_cone_e.get_right_gripper_exact()
            left_joint_positions = self.obs_cone_e.get_left_joint_positions()
            right_joint_positions = self.obs_cone_e.get_right_joint_positions()

        left_gripper_binary, right_gripper_binary = self._process_gripper_states(
            left_gripper, right_gripper
        )

        # Head camera
        rgb_frame, rgb_timestamp, depth_frame = self.camera.get_latest_frame()
        if rgb_frame is not None:
            rgb_frame = np.rot90(rgb_frame, k=3)
        if depth_frame is not None:
            depth_frame = np.rot90(depth_frame, k=3)

        # Left wrist camera (disabled)
        left_wrist_frame = None

        # Right wrist camera
        if self.right_wrist_camera is not None:
            right_wrist_frame, _, _ = self.right_wrist_camera.get_latest_frame()
            if right_wrist_frame is not None:
                right_wrist_frame = np.rot90(right_wrist_frame, k=3)
        else:
            right_wrist_frame = None

        # Build state vectors
        left_pose = np.concatenate([
            ee_pose_left.translation(),
            quat_to_r6(ee_pose_left.rotation().wxyz, batched=False)
        ])
        right_pose = np.concatenate([
            ee_pose_right.translation(),
            quat_to_r6(ee_pose_right.rotation().wxyz, batched=False),
        ])

        observation = {
            'qpos': np.concatenate([
                left_pose,
                np.array([left_gripper], dtype=float),
                right_pose,
                np.array([right_gripper], dtype=float),
            ]),
            "images": {
                "cam_high": rgb_frame if rgb_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                "cam_left_wrist": left_wrist_frame if left_wrist_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                "cam_right_wrist": right_wrist_frame if right_wrist_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
            },
            "depth": depth_frame if depth_frame is not None else np.zeros((480, 640), dtype=np.float32),
            "timestamp": timestamp,
            "rgb_timestamp": rgb_timestamp,
            "task": self.task,
        }

        # Record if enabled
        if self.recorder and self.recorder.is_recording:
            sample = RecordingSample(
                timestamp=timestamp,
                left_ee_pose=ee_pose_left,
                right_ee_pose=ee_pose_right,
                left_gripper_exact=left_gripper,
                right_gripper_exact=right_gripper,
                left_gripper=left_gripper_binary,
                right_gripper=right_gripper_binary,
                rgb_frame=self.camera.latest_rgb_frame.copy() if self.camera.latest_rgb_frame is not None else None,
                depth_frame=self.camera.latest_depth_frame.copy() if self.camera.latest_depth_frame is not None else None,
                rgb_timestamp=rgb_timestamp,
                left_joint_positions=left_joint_positions,
                right_joint_positions=right_joint_positions,
                left_wrist_rgb_frame=None,
                right_wrist_rgb_frame=right_wrist_frame,
            )
            self.recorder.record_sample(sample)

        return observation

    def _process_gripper_states(self, left_gripper, right_gripper):
        delta_left_gripper = left_gripper - self.last_left_gripper
        delta_right_gripper = right_gripper - self.last_right_gripper

        if abs(delta_left_gripper) > 0.04:
            self.last_left_gripper = left_gripper
            left_gripper_binary = 1.0 if delta_left_gripper > 0 else 0.0
            self.last_left_gripper_binary = left_gripper_binary
        else:
            left_gripper_binary = self.last_left_gripper_binary

        if abs(delta_right_gripper) > 0.04:
            self.last_right_gripper = right_gripper
            right_gripper_binary = 1.0 if right_gripper > 0.5 else 0.0
            self.last_right_gripper_binary = right_gripper_binary
        else:
            right_gripper_binary = self.last_right_gripper_binary

        return left_gripper_binary, right_gripper_binary

    def _observation_publishing_loop(self):
        print("Observation publishing thread started")
        rate_limiter = RateLimiter(10)

        while not self.stop_event.is_set():
            try:
                observation = self.get_observation()
                self.obs_socket.send_pyobj(observation, flags=zmq.NOBLOCK)
                self.stats['observations_sent'] += 1

                if self.stats['observations_sent'] % 300 == 0:
                    print(f"Published {self.stats['observations_sent']} observations")

            except zmq.Again:
                pass

            self.camera.is_episode_active = self.episode_manager.is_active()
            self.camera.episode_start_time = self.episode_manager.get_start_time()
            self.camera.is_recording = self.recorder.is_recording if self.recorder else False

            rate_limiter.sleep()

        print("Observation publishing thread stopped")

    def request_action(self):
        try:
            self.action_socket.send_pyobj({'request': 'action'})
            action = self.action_socket.recv_pyobj()

            if 'error' in action:
                print(f"Server error: {action['error']}")
                return None

            self.stats['actions_received'] += 1

            if action.get('is_stale', False):
                self.stats['buffer_wraps'] += 1

            return action

        except zmq.error.Again:
            if not hasattr(self, '_last_timeout_warning'):
                self._last_timeout_warning = 0
            now = time.time()
            if now - self._last_timeout_warning > 2.0:
                print("Timeout waiting for action from server")
                self._last_timeout_warning = now
            self.stats['errors'] += 1
            return None
        except Exception as e:
            if not hasattr(self, '_last_comm_error_time'):
                self._last_comm_error_time = 0
            now = time.time()
            if now - self._last_comm_error_time > 1.0:
                print(f"Communication error: {e}")
                self._last_comm_error_time = now
            self.stats['errors'] += 1
            return None

    def apply_action(self, action):
        if not self.episode_manager.is_active():
            self.starting_pose_left = None
            self.starting_pose_right = None
            return

        if self.starting_pose_left is None or self.starting_pose_right is None:
            self.starting_pose_left = self.cone_e.get_left_ee_pose()
            self.starting_pose_right = self.cone_e.get_right_ee_pose()

        # Apply left arm action
        if 'left_delta_pose' in action and action['left_delta_pose'] is not None:
            self.starting_pose_left = self._apply_arm_action(
                action['left_delta_pose'],
                action.get('left_gripper', 0.5),
                self.starting_pose_left,
                self.cone_e.set_left_ee_target
            )
        elif 'left_ee_pose' in action and action['left_ee_pose'] is not None:
            self._apply_arm_action_absolute(
                action['left_ee_pose'],
                action.get('left_gripper', 0.5),
                self.starting_pose_left,
                self.cone_e.set_left_ee_target
            )
            self.starting_pose_left = self.cone_e.get_left_ee_pose()

        # Apply right arm action
        if 'right_delta_pose' in action and action['right_delta_pose'] is not None:
            self.starting_pose_right = self._apply_arm_action(
                action['right_delta_pose'],
                action.get('right_gripper', 0.5),
                self.starting_pose_right,
                self.cone_e.set_right_ee_target
            )
        elif 'right_ee_pose' in action and action['right_ee_pose'] is not None:
            self._apply_arm_action_absolute(
                action['right_ee_pose'],
                action.get('right_gripper', 0.5),
                self.starting_pose_right,
                self.cone_e.set_right_ee_target
            )
            self.starting_pose_right = self.cone_e.get_right_ee_pose()

    def _apply_arm_action(self, delta_pose, gripper, starting_pose, set_target_fn):
        X_delta = mink.SE3(delta_pose)
        X_Rdelta = self.H.inverse() @ X_delta @ self.H

        p_target = starting_pose.translation() + X_Rdelta.translation()
        R_target = X_Rdelta.rotation() @ starting_pose.rotation()

        ee_distance = np.linalg.norm(X_Rdelta.translation())
        preview_time = np.clip(ee_distance / 0.5, 0.01, 0.5)

        target_pose = mink.SE3(np.concatenate([R_target.wxyz, p_target]))

        set_target_fn(
            ee_target=target_pose,
            gripper_target=gripper,
            preview_time=preview_time,
        )

        return target_pose

    def _apply_arm_action_absolute(self, abs_pose, gripper, starting_pose, set_target_fn):
        X_target = mink.SE3(abs_pose)
        X_Rtarget = self.H.inverse() @ X_target @ self.H

        p_target = X_Rtarget.translation()
        R_target = X_Rtarget.rotation()

        ee_distance = np.linalg.norm(p_target - starting_pose.translation())
        preview_time = np.clip(ee_distance / 0.5, 0.01, 0.5)

        set_target_fn(
            ee_target=mink.SE3(np.concatenate([R_target.wxyz, p_target])),
            gripper_target=gripper,
            preview_time=preview_time,
        )

    def control_loop(self, control_rate=30):
        rate_limiter = RateLimiter(control_rate)
        self.policy_active = True

        self._print_startup_info(control_rate)
        self.episode_manager.set_controller_start_time()

        for idx in range(5):
            print(f"Control loop will start in {5 - idx}")
            time.sleep(1)

        iteration = 0
        wait_for_ready_count = 0

        while not self.stop_event.is_set():
            loop_start = time.time()

            self.episode_manager.check_autonomous_conditions(
                self.manipulability_calc, iteration
            )

            # エピソードが非アクティブなら、アクションリクエストをスキップ
            if not self.episode_manager.is_active():
                rate_limiter.sleep()
                continue

            action = self.request_action()

            if action is not None:
                self.apply_action(action)

                if iteration % 30 == 0:
                    self._print_status(action, loop_start)

                wait_for_ready_count = 0
            else:
                wait_for_ready_count += 1
                if wait_for_ready_count % 30 == 1:
                    print("Waiting for HPC server to be ready...")

            iteration += 1
            rate_limiter.sleep()

        print("\nControl loop stopped")
        self._print_stats()
        self.policy_active = False

    def _print_startup_info(self, control_rate):
        print(f"\nStarting policy control loop at {control_rate} Hz")
        print("Observations publishing at 10 Hz in background")
        print(f"Task: '{self.task}'")

        if self.episode_manager.autonomous_mode:
            print("AUTONOMOUS MODE ENABLED")
            print(f"  Auto-start delay: {self.episode_manager.auto_start_delay}s")
            print(f"  Episode timeout: {self.episode_manager.episode_timeout}s")
            print(f"  Manipulability threshold: {self.episode_manager.manipulability_threshold}")
        else:
            print("MANUAL MODE")
            print("  Press 's' to start episode")
            print("  Press 'e' to end episode")

        if self.recorder:
            print("Recording is ENABLED (automatic with episodes)")

        print("Press 'q' or Ctrl+C to stop\n")

    def _print_status(self, action, loop_start):
        loop_time = (time.time() - loop_start) * 1000
        buffer_age = action.get('buffer_age', 0) * 1000
        buffer_remaining = action.get('buffer_remaining', '?')
        is_stale = action.get('is_stale', False)

        stale_marker = " ⚠️STALE" if is_stale else ""
        rec_marker = " 🔴REC" if (self.recorder and self.recorder.is_recording) else ""
        episode_marker = " ▶️ACTIVE" if self.episode_manager.is_active() else " ⏸️PAUSED"

        status_msg = (f"Iter {self.stats['actions_received']}: "
                      f"loop={loop_time:.1f}ms, buffer_age={buffer_age:.1f}ms, "
                      f"remaining={buffer_remaining}{stale_marker}{rec_marker}{episode_marker}")

        if self.episode_manager.is_active() and self.episode_manager.get_start_time():
            episode_elapsed = time.time() - self.episode_manager.get_start_time()
            status_msg += f", episode_time={episode_elapsed:.1f}s"

        print(status_msg)

        if self.stats['buffer_wraps'] > 0 and self.stats['actions_received'] % 90 == 0:
            print(f"  WARNING: {self.stats['buffer_wraps']} stale actions served (inference too slow)")

    def _print_stats(self):
        print(f"\n{'='*50}")
        print("Final Statistics:")
        print(f"  Observations sent: {self.stats['observations_sent']}")
        print(f"  Actions received: {self.stats['actions_received']}")
        print(f"  Stale actions: {self.stats['buffer_wraps']}")
        print(f"  Errors: {self.stats['errors']}")
        if self.recorder:
            print(f"  Episodes recorded: {self.episode_manager.get_count()}")
        print(f"{'='*50}\n")

    def stop(self):
        print("Stopping policy controller...")

        if self.episode_manager.is_active():
            self.episode_manager.end_episode(reason="shutdown")

        self.stop_event.set()

        self.obs_thread.join(timeout=2.0)
        self.camera.stop()
        if self.right_wrist_camera:
            self.right_wrist_camera.stop()
        if self.recorder:
            self.recorder.stop()
        self.keyboard.stop()

        self.obs_socket.close()
        self.action_socket.close()
        self.zmq_context.term()

        print("Policy controller stopped")