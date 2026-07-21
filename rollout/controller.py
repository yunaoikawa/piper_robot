"""Main policy controller orchestrating robot control and data collection."""

import cv2
import zmq
import time
import mink
import numpy as np
import threading
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from robot.rpc import RPCClient
from robot.camera_id import load_camera_map
from loop_rate_limiters import RateLimiter

from .camera import CameraFeedManager, USBWristCameraFeedManager
from .recorder import DataRecorder, RecordingSample
from .episode import EpisodeManager
from .keyboard import KeyboardController
from .manipulability import ManipulabilityCalculator
from .safety import SafetyLayer

DATA_DIR = Path("./your_save_dir_here")
DEFAULT_TASK = "put the flask in the incubator"
TARGET_H, TARGET_W = 480, 640

# Hard cap on any single-axis EE bias. The largest value ever needed by hand is
# 0.03 m (outputs/lab/act/horizon/EVAL_RESULTS.md), so 0.06 leaves room to tune
# while bounding what an automatically-proposed bias can do.
MAX_BIAS_M = 0.06


def quat_to_r6(quat, batched=False):
    rot_mat = R.from_quat(quat, scalar_first=True).as_matrix()
    if batched:
        a1, a2 = rot_mat[:, :, 0], rot_mat[:, :, 1]
        return np.concatenate((a1, a2), axis=-1)
    a1, a2 = rot_mat[:, 0], rot_mat[:, 1]
    return np.concatenate((a1, a2))


def _rotate_and_resize(frame):
    if frame is None:
        return None
    frame = np.rot90(frame, k=3)
    if frame.shape[0] != TARGET_H or frame.shape[1] != TARGET_W:
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    return frame


class PolicyController:
    def __init__(self, hpc_host="192.168.1.50", obs_port=5555, action_port=5556,
                 enable_recording=False, save_dir=None, autonomous_mode=False,
                 episode_timeout=600.0, manipulability_threshold=0.05,
                 task=DEFAULT_TASK, safety_config=None, bias_port=5560):
        self.stop_event = threading.Event()
        self.policy_active = False
        self.task = task

        # Per-arm EE offset in robot frame, applied in apply_action(). This
        # replaces the server-side --z-bias so the bias exists in exactly one
        # place; two independent biases would silently sum.
        self.xyz_bias = {'left': np.zeros(3), 'right': np.zeros(3)}
        self.safety = SafetyLayer.from_config(safety_config)
        self._buffer_gen = None  # server's buffer generation; changes on re-plan
        # NB: not `control_port` -- _setup_zmq() uses that name for the REQ
        # socket to the *server's* control port (action_port + 1).
        self.bias_port = bias_port

        self.obs_cone_e = RPCClient("localhost", 8081)
        self.obs_cone_e.init()
        self.obs_rpc_lock = threading.Lock()

        self.cone_e = RPCClient("localhost", 8081)
        self.cone_e.init()
        self.cone_e.home_left_arm()
        self.cone_e.home_right_arm()

        self._setup_zmq(hpc_host, obs_port, action_port)

        self.H = mink.SE3.from_rotation(mink.SO3.from_matrix(np.eye(3)))

        self.last_left_gripper = 1.0
        self.last_right_gripper = 1.0
        self.last_left_gripper_binary = 1.0
        self.last_right_gripper_binary = 1.0
        self.starting_pose_left = None
        self.starting_pose_right = None

        self.stats = {
            'observations_sent': 0, 'actions_received': 0,
            'errors': 0, 'buffer_wraps': 0
        }
        self.test_qpos = None
        save_path = Path(save_dir) if save_dir else DATA_DIR
        cam_map = load_camera_map()

        self.camera = CameraFeedManager(self.stop_event)
        self.camera.autonomous_mode = autonomous_mode
        self.camera.start()

        self.right_wrist_camera = USBWristCameraFeedManager(
            self.stop_event, device_index=cam_map.get("right", 1), label="right wrist"
        )
        self.right_wrist_camera.start()

        self.left_wrist_camera = USBWristCameraFeedManager(
            self.stop_event, device_index=cam_map.get("left", 2), label="left wrist"
        )
        self.left_wrist_camera.start()

        self.camera.wrist_camera = self.right_wrist_camera
        self.camera.left_wrist_camera = self.left_wrist_camera

        self.recorder = DataRecorder(save_path, self.stop_event) if enable_recording else None

        self.episode_manager = EpisodeManager(
            recorder=self.recorder, robot_rpc=self.cone_e,
            control_socket=self.control_socket, autonomous_mode=autonomous_mode,
            episode_timeout=episode_timeout, manipulability_threshold=manipulability_threshold
        )

        self.keyboard = KeyboardController(
            self.stop_event, self.episode_manager, enable_recording, autonomous_mode
        )
        self.keyboard.start()

        self.manipulability_calc = ManipulabilityCalculator(self.obs_cone_e, self.obs_rpc_lock)

        self.obs_thread = threading.Thread(target=self._observation_publishing_loop, daemon=True)
        self.obs_thread.start()

        self.bias_thread = threading.Thread(target=self._bias_control_loop, daemon=True)
        self.bias_thread.start()

    def _bias_control_loop(self):
        """REP socket for changing the EE bias without restarting anything.

        Mirrors the server's control loop in hpc_inference_act.py. Tuning a bias
        used to mean killing and relaunching the inference server; now it is one
        message. Commands:
            {'command': 'set_bias', 'arm': 'right', 'bias': [x, y, z]}
            {'command': 'get_bias'}
        """
        # Use a private context: terminating the policy sockets' context during
        # shutdown must never wait on a socket owned by this worker thread.
        context = zmq.Context()
        sock = context.socket(zmq.REP)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(f"tcp://*:{self.bias_port}")
        print(f"Bias control listening on port {self.bias_port}")

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        while not self.stop_event.is_set():
            try:
                # Poll so shutdown isn't blocked waiting on a message.
                if not poller.poll(timeout=200):
                    continue
                cmd = sock.recv_pyobj()
                name = cmd.get('command')

                if name == 'set_bias':
                    arm = cmd.get('arm', 'right')
                    if arm not in self.xyz_bias:
                        reply = {'status': 'error', 'message': f"unknown arm '{arm}'"}
                    else:
                        applied = self.set_bias(arm, cmd.get('bias', [0, 0, 0]))
                        reply = {'status': 'ok', 'arm': arm, 'bias': applied.tolist()}
                elif name == 'get_bias':
                    reply = {'status': 'ok',
                             'bias': {k: v.tolist() for k, v in self.xyz_bias.items()},
                             'safety_rejected': self.safety.rejected_count}
                else:
                    reply = {'status': 'error', 'message': f"unknown command '{name}'"}

                sock.send_pyobj(reply)
            except Exception as e:
                if self.stop_event.is_set():
                    break
                print(f"[bias] control thread error: {e}", flush=True)
                # REP sockets must reply or the socket wedges in a bad state.
                try:
                    sock.send_pyobj({'status': 'error', 'message': str(e)})
                except Exception:
                    pass

        sock.close()
        context.term()

    def _setup_zmq(self, hpc_host, obs_port, action_port):
        self.zmq_context = zmq.Context()

        self.obs_socket = self.zmq_context.socket(zmq.PUB)
        self.obs_socket.connect(f"tcp://{hpc_host}:{obs_port}")
        print(f"Publishing observations to tcp://{hpc_host}:{obs_port}")

        self.action_socket = self.zmq_context.socket(zmq.REQ)
        self.action_socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.action_socket.setsockopt(zmq.SNDTIMEO, 2000)
        self.action_socket.setsockopt(zmq.LINGER, 0)
        self.action_socket.connect(f"tcp://{hpc_host}:{action_port}")
        print(f"Requesting actions from tcp://{hpc_host}:{action_port}")

        self.control_port = action_port + 1
        self.control_socket = self.zmq_context.socket(zmq.REQ)
        self.control_socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.control_socket.setsockopt(zmq.SNDTIMEO, 2000)
        self.control_socket.setsockopt(zmq.LINGER, 0)
        self.control_socket.connect(f"tcp://{hpc_host}:{self.control_port}")
        print(f"Sending control commands to tcp://{hpc_host}:{self.control_port}")
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

        rgb_frame, rgb_timestamp, depth_frame = self.camera.get_latest_frame()
        rgb_frame = _rotate_and_resize(rgb_frame)

        left_wrist_frame = None
        if self.left_wrist_camera is not None:
            left_wrist_frame, _, _ = self.left_wrist_camera.get_latest_frame()
            left_wrist_frame = _rotate_and_resize(left_wrist_frame)

        right_wrist_frame = None
        if self.right_wrist_camera is not None:
            right_wrist_frame, _, _ = self.right_wrist_camera.get_latest_frame()
            right_wrist_frame = _rotate_and_resize(right_wrist_frame)

        blank = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)

        left_pose = np.concatenate([
            ee_pose_left.translation(),
            quat_to_r6(ee_pose_left.rotation().wxyz)
        ])
        right_pose = np.concatenate([
            ee_pose_right.translation(),
            quat_to_r6(ee_pose_right.rotation().wxyz)
        ])

        observation = {
            'qpos': np.concatenate([
                left_pose, np.array([left_gripper], dtype=float),
                right_pose, np.array([right_gripper], dtype=float),
            ]),
            "images": {
                "cam_high": rgb_frame if rgb_frame is not None else blank.copy(),
                "cam_left_wrist": left_wrist_frame if left_wrist_frame is not None else blank.copy(),
                "cam_right_wrist": right_wrist_frame if right_wrist_frame is not None else blank.copy(),
            },
            "depth": None,
            "timestamp": timestamp,
            "rgb_timestamp": rgb_timestamp,
            "task": self.task,
        }

        if self.recorder and self.recorder.is_recording:
            sample = RecordingSample(
                timestamp=timestamp,
                left_ee_pose=ee_pose_left, right_ee_pose=ee_pose_right,
                left_gripper_exact=left_gripper, right_gripper_exact=right_gripper,
                left_gripper=left_gripper_binary, right_gripper=right_gripper_binary,
                rgb_frame=self.camera.latest_rgb_frame.copy() if self.camera.latest_rgb_frame is not None else None,
                depth_frame=None, rgb_timestamp=rgb_timestamp,
                left_joint_positions=left_joint_positions,
                right_joint_positions=right_joint_positions,
                left_wrist_rgb_frame=self.left_wrist_camera.latest_rgb.copy() if (self.left_wrist_camera and self.left_wrist_camera.latest_rgb is not None) else None,
                right_wrist_rgb_frame=self.right_wrist_camera.latest_rgb.copy() if (self.right_wrist_camera and self.right_wrist_camera.latest_rgb is not None) else None,
            )
            self.recorder.record_sample(sample)

        return observation

    def _process_gripper_states(self, left_gripper, right_gripper):
        delta_left = left_gripper - self.last_left_gripper
        delta_right = right_gripper - self.last_right_gripper

        if abs(delta_left) > 0.04:
            self.last_left_gripper = left_gripper
            self.last_left_gripper_binary = 1.0 if delta_left > 0 else 0.0
        if abs(delta_right) > 0.04:
            self.last_right_gripper = right_gripper
            self.last_right_gripper_binary = 1.0 if right_gripper > 0.5 else 0.0

        return self.last_left_gripper_binary, self.last_right_gripper_binary

    def _observation_publishing_loop(self):
        print("Observation publishing thread started")
        rate_limiter = RateLimiter(2)

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
            now = time.time()
            if not hasattr(self, '_last_timeout_warning') or now - self._last_timeout_warning > 2.0:
                print("Timeout waiting for action from server")
                self._last_timeout_warning = now
            self.stats['errors'] += 1
            return None
        except Exception as e:
            now = time.time()
            if not hasattr(self, '_last_comm_error_time') or now - self._last_comm_error_time > 1.0:
                print(f"Communication error: {e}")
                self._last_comm_error_time = now
            self.stats['errors'] += 1
            return None

    def apply_action(self, action):
        # Debug: log action values every 10 steps
        if self.stats['actions_received'] % 10 == 1:
            lp = action.get('left_ee_pose')
            rp = action.get('right_ee_pose')
            ld = action.get('left_delta_pose')
            rd = action.get('right_delta_pose')
            parts = []
            if lp is not None:
                parts.append(f"L_abs={lp[:3]}")
            if rp is not None:
                parts.append(f"R_abs={rp[:3]}")
            if ld is not None:
                parts.append(f"L_delta={ld[:3]}")
            if rd is not None:
                parts.append(f"R_delta={rd[:3]}")
            if parts:
                print(f"  ACTION: {', '.join(parts)}")

        if not self.episode_manager.is_active():
            self.starting_pose_left = None
            self.starting_pose_right = None
            self.safety.reset()  # next episode's first target has no predecessor
            self._buffer_gen = None
            return

        # A re-plan overwrites the buffer, so the next target is computed from
        # the arm's *current* pose while the previous one was however far ahead
        # the tracking lag had put it. That legitimate discontinuity is not a
        # runaway, and at transport speed it can exceed the step limit -- which
        # would reject exactly the fast motion we need. The server stamps every
        # action with the buffer generation, so drop the step reference when it
        # changes and let the first target of each chunk through.
        gen = action.get('total_buffer_updates')
        if gen is not None and gen != self._buffer_gen:
            self.safety.reset()
            self._buffer_gen = gen

        if self.starting_pose_left is None or self.starting_pose_right is None:
            # Latch the reference pose. In delta mode this is the ONLY place the
            # bias enters -- applying it per step would integrate (see
            # _apply_arm_action). In absolute mode the latched pose is just a
            # distance reference and the bias is re-applied per target, so
            # offsetting it here is harmless.
            self.starting_pose_left = self._biased_pose(
                self.cone_e.get_left_ee_pose(), 'left')
            self.starting_pose_right = self._biased_pose(
                self.cone_e.get_right_ee_pose(), 'right')
            self.safety.reset()

        if 'left_delta_pose' in action and action['left_delta_pose'] is not None:
            self.starting_pose_left = self._apply_arm_action(
                'left', action['left_delta_pose'], action.get('left_gripper', 0.5),
                self.starting_pose_left, self.cone_e.set_left_ee_target
            )
        elif 'left_ee_pose' in action and action['left_ee_pose'] is not None:
            self._apply_arm_action_absolute(
                'left', action['left_ee_pose'], action.get('left_gripper', 0.5),
                self.starting_pose_left, self.cone_e.set_left_ee_target
            )
            self.starting_pose_left = self.cone_e.get_left_ee_pose()

        if 'right_delta_pose' in action and action['right_delta_pose'] is not None:
            self.starting_pose_right = self._apply_arm_action(
                'right', action['right_delta_pose'], action.get('right_gripper', 0.5),
                self.starting_pose_right, self.cone_e.set_right_ee_target
            )
        elif 'right_ee_pose' in action and action['right_ee_pose'] is not None:
            self._apply_arm_action_absolute(
                'right', action['right_ee_pose'], action.get('right_gripper', 0.5),
                self.starting_pose_right, self.cone_e.set_right_ee_target
            )
            self.starting_pose_right = self.cone_e.get_right_ee_pose()

    def _biased_pose(self, pose, arm):
        """Return `pose` translated by that arm's bias, rotation untouched."""
        bias = self.xyz_bias[arm]
        if pose is None or not np.any(bias):
            return pose
        return mink.SE3(np.concatenate([
            pose.rotation().wxyz, pose.translation() + bias
        ]))

    def set_bias(self, arm, bias):
        """Set an arm's xyz bias (metres, robot frame), clamped to MAX_BIAS_M.

        The clamp is the guard against a bad automatic value: a VLM-proposed
        offset can nudge the arm, never fling it.
        """
        b = np.asarray(bias, dtype=float).reshape(3)
        clamped = np.clip(b, -MAX_BIAS_M, MAX_BIAS_M)
        if not np.allclose(b, clamped):
            print(f"[bias] {arm} request {np.round(b, 4)} clamped to "
                  f"{np.round(clamped, 4)} (limit ±{MAX_BIAS_M} m)", flush=True)
        self.xyz_bias[arm] = clamped
        # Changing the bias jumps the next target by the delta -- a legitimate
        # discontinuity, not a runaway. Drop the step reference so the safety
        # layer doesn't reject the frame right after a live bias change.
        self.safety.reset(arm)
        print(f"[bias] {arm} = {np.round(clamped, 4)} m", flush=True)
        return clamped

    def _apply_arm_action(self, arm, delta_pose, gripper, starting_pose, set_target_fn):
        X_delta = mink.SE3(delta_pose)
        X_Rdelta = self.H.inverse() @ X_delta @ self.H
        p_target = starting_pose.translation() + X_Rdelta.translation()
        R_target = X_Rdelta.rotation() @ starting_pose.rotation()

        # NOTE: no bias is added here. In delta mode `starting_pose` is replaced
        # by the target we return, so adding a constant offset every step would
        # integrate into a drift. The bias is applied once, to the latched
        # starting pose, in apply_action().
        p_safe = self.safety.check(arm, p_target)
        if p_safe is None:
            return starting_pose  # hold: keep the previous target as reference

        target_pose = mink.SE3(np.concatenate([R_target.wxyz, p_safe]))
        set_target_fn(ee_target=target_pose, gripper_target=gripper, preview_time=0.5)
        return target_pose

    def _apply_arm_action_absolute(self, arm, abs_pose, gripper, starting_pose, set_target_fn):
        X_target = mink.SE3(abs_pose)
        X_Rtarget = self.H.inverse() @ X_target @ self.H

        # Absolute mode: the target is recomputed from scratch each step, so a
        # constant offset stays constant -- safe to add every time. Applied in
        # the robot frame (post-H) so the numbers mean what the workspace bounds
        # and the EVAL_RESULTS z-bias figures mean.
        p_target = X_Rtarget.translation() + self.xyz_bias[arm]
        R_target = X_Rtarget.rotation()

        p_safe = self.safety.check(arm, p_target)
        if p_safe is None:
            return

        set_target_fn(
            ee_target=mink.SE3(np.concatenate([R_target.wxyz, p_safe])),
            gripper_target=gripper, preview_time=0.5,
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
            self.episode_manager.check_autonomous_conditions(self.manipulability_calc, iteration)

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
        print(f"Observations publishing at 2 Hz in background")
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

        stale = " ⚠️STALE" if is_stale else ""
        rec = " 🔴REC" if (self.recorder and self.recorder.is_recording) else ""
        ep = " ▶️ACTIVE" if self.episode_manager.is_active() else " ⏸️PAUSED"

        msg = (f"Iter {self.stats['actions_received']}: "
               f"loop={loop_time:.1f}ms, buffer_age={buffer_age:.1f}ms, "
               f"remaining={buffer_remaining}{stale}{rec}{ep}")

        if self.episode_manager.is_active() and self.episode_manager.get_start_time():
            msg += f", episode_time={time.time() - self.episode_manager.get_start_time():.1f}s"
        print(msg)

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
        # The bias thread owns a socket on self.zmq_context. Let it observe the
        # stop event and close that socket before terminating the shared
        # context; otherwise Context.term() can block indefinitely at shutdown.
        self.bias_thread.join(timeout=2.0)
        self.camera.stop()
        if self.left_wrist_camera:
            self.left_wrist_camera.stop()
        if self.right_wrist_camera:
            self.right_wrist_camera.stop()
        if self.recorder:
            self.recorder.stop()
        self.keyboard.stop()
        self.obs_socket.close()
        self.action_socket.close()
        self.zmq_context.term()
        print("Policy controller stopped")
