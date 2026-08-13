"""Main policy controller orchestrating robot control and data collection."""

import cv2
import zmq
import time
import mink
import numpy as np
import threading
import hashlib
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from robot.rpc import RPCClient
from robot.camera_id import configure_camera_map_by_udid, load_camera_map
from loop_rate_limiters import RateLimiter

from .camera import CameraFeedManager, USBWristCameraFeedManager
from .recorder import DataRecorder, RecordingSample
from .episode import EpisodeManager
from .keyboard import KeyboardController
from .manipulability import ManipulabilityCalculator
from .safety import SafetyLayer
from .agent_collection import (
    AgentEpisodeRecorder, AgentRecordingSample, ControllerClaim,
    FAILURE_REASONS, GripperCloseLatch, InterventionState,
    gripper_latch_config,
)
from .agent_collection_ui import AgentCollectionUI
from .agent_cycle import AlternatingAgentCycle
from .agent_home import run_agent_auto_home

DATA_DIR = Path("./your_save_dir_here")
DEFAULT_TASK = "put the flask in the incubator"
TARGET_H, TARGET_W = 480, 640


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
                 task=DEFAULT_TASK, safety_config=None, bias_port=5560,
                 agent_collection=False, agent_task=None,
                 agent_ui_host="0.0.0.0", agent_ui_port=8780,
                 agent_ui_token="", controller_lock="/tmp/piper_robot_right_arm_controller.lock",
                 agent_auto_home=True, agent_config="src/configs/lid_agent_collection.json",
                 agent_cycle=False):
        self.stop_event = threading.Event()
        self.policy_active = False
        self.task = task
        self.agent_collection = bool(agent_collection)
        self.agent_task = agent_task
        if self.agent_collection and agent_task not in {"lid_open", "lid_close"}:
            raise ValueError("--agent-task must be lid_open or lid_close")
        self.controller_claim = ControllerClaim(controller_lock) if self.agent_collection else None
        if self.controller_claim:
            self.controller_claim.acquire()
        self.intervention = InterventionState() if self.agent_collection else None
        self.agent_profile = (
            json.loads(Path(agent_config).read_text())
            if self.agent_collection else {}
        )
        self.agent_cycle_enabled = bool(agent_cycle)
        if self.agent_cycle_enabled and not self.agent_collection:
            raise ValueError("agent_cycle requires agent_collection")
        self.agent_cycle = None
        if self.agent_cycle_enabled:
            cycle = self.agent_profile.get("cycle", {})
            self.agent_cycle = AlternatingAgentCycle(
                cycle.get("tasks", ("lid_close", "lid_open")),
                initial_task=agent_task,
                required_cameras=cycle.get("required_cameras", ("head", "right")),
                camera_failure_grace_s=cycle.get("camera_failure_grace_s", 1.0),
                inference_failure_grace_s=cycle.get(
                    "inference_failure_grace_s", 10.0
                ),
                maximum_episode_s=cycle.get("maximum_episode_s", 60.0),
                bias_retry_offsets_m=cycle.get("bias_retry_offsets_m"),
            )
        self._agent_task_bias = {
            name: np.asarray(settings["initial_right_xyz_bias_m"], dtype=float)
            for name, settings in self.agent_profile.get("tasks", {}).items()
        }
        if self.agent_cycle is not None:
            maximum_bias = float(
                self.agent_profile.get("intervention", {}).get(
                    "maximum_absolute_bias_m", 0.06
                )
            )
            for task_name, offsets in self.agent_cycle.bias_retry_offsets_m.items():
                scheduled = (
                    self._agent_task_bias[task_name][None, :]
                    + np.asarray(offsets, dtype=float)
                )
                if np.any(np.abs(scheduled) > maximum_bias + 1e-12):
                    raise ValueError(
                        f"{task_name} bias retry schedule exceeds configured "
                        f"±{maximum_bias:.3f} m limit"
                    )
        self._agent_cycle_transition_active = False
        self._agent_camera_last_id = {}
        self._agent_camera_last_change_s = {}
        self._resume_after_obs_timestamp = None
        self._agent_sample_index = 0
        self._last_action_record = {
            "policy": np.full(16, np.nan), "commanded": np.full(16, np.nan),
            "chunk_index": -1, "generation": -1, "obs_timestamp": np.nan,
        }
        self._action_record_lock = threading.Lock()
        self._latest_ui_frames = {}
        self._resume_anchor_xyz = None
        self._resume_anchor_offset = np.zeros(3, dtype=float)
        self.agent_auto_home = bool(agent_auto_home)
        # Time-agnostic ACT can otherwise re-enter an earlier open/close phase
        # after a successful grasp.  Both pilot tasks transport before their
        # demonstrated final release, so the same measured-displacement gate
        # protects their grasp without suppressing that release.
        self._right_close_latch = GripperCloseLatch(
            **gripper_latch_config(self.agent_profile, agent_task)
        )
        self._agent_post_release_offset = self._agent_release_offset(agent_task)
        self._right_close_latch_enabled = self.agent_collection

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
        if self.agent_collection:
            self.agent_start_home_report = None
            if self.agent_auto_home:
                self.agent_start_home_report = self._agent_home_both_arms()
        else:
            self.agent_start_home_report = None
            self.cone_e.home_left_arm()
            self.cone_e.home_right_arm()

        self._setup_zmq(hpc_host, obs_port, action_port)

        self.H = mink.SE3.from_rotation(mink.SO3.from_matrix(np.eye(3)))

        self.last_left_gripper = 1.0
        self.last_right_gripper = 1.0
        self.latest_right_gripper_exact = 1.0
        self.latest_right_ee_xyz = None
        self._policy_image_cache = {
            name: {"timestamp": None, "frame": None}
            for name in ("head", "left", "right")
        }
        self._observation_state_rpc_available = None
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
        if self.agent_collection:
            cam_map, live_udids = configure_camera_map_by_udid(
                self.agent_profile["camera_udids"], preserve_unknown=False
            )
            print(f"[agent] camera map by UDID: {cam_map}; live={live_udids}")
        else:
            cam_map = load_camera_map()

        # The agent phone UI is the display surface.  Creating Qt/OpenCV
        # windows from a headless lab daemon aborts the whole process before
        # the operator can press START.
        self.camera = CameraFeedManager(
            self.stop_event, display=not self.agent_collection
        )
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

        if self.agent_collection:
            self.recorder = AgentEpisodeRecorder(save_path, self.stop_event, fps=30)
        else:
            self.recorder = DataRecorder(save_path, self.stop_event) if enable_recording else None

        self.episode_manager = EpisodeManager(
            recorder=self.recorder, robot_rpc=self.cone_e,
            control_socket=self.control_socket, autonomous_mode=autonomous_mode,
            episode_timeout=episode_timeout, manipulability_threshold=manipulability_threshold
        )
        if self.agent_cycle_enabled:
            self._activate_agent_inference(agent_task)

        self.keyboard = None
        if not self.agent_collection:
            self.keyboard = KeyboardController(
                self.stop_event, self.episode_manager, enable_recording, autonomous_mode
            )
            self.keyboard.start()

        self.agent_ui = None
        if self.agent_collection:
            self.set_bias(
                "right", self._agent_task_bias.get(
                    agent_task, np.array([0.0, 0.0, -0.03])
                )
            )
            self.intervention.set_bias("right", self.xyz_bias["right"])
            self.intervention.set_bias("left", self.xyz_bias["left"])
            self.intervention.latest_metrics["task"] = agent_task
            self.intervention.latest_metrics["cycle_enabled"] = self.agent_cycle_enabled
            self.intervention.latest_metrics["auto_home"] = self.agent_start_home_report
            self.intervention.latest_metrics["pilot_success_target"] = 10
            self.intervention.latest_metrics["success_count"] = self.recorder.success_count(agent_task)
            self.agent_ui = AgentCollectionUI(
                agent_ui_host, agent_ui_port, agent_ui_token, self.intervention,
                self._submit_agent_command, self._agent_frame_provider,
            )
            self.agent_ui.start()
            print(f"Agent collection UI: http://{agent_ui_host}:{self.agent_ui.port}/")

        self.manipulability_calc = ManipulabilityCalculator(self.obs_cone_e, self.obs_rpc_lock)

        self.obs_thread = threading.Thread(target=self._observation_publishing_loop, daemon=True)
        self.obs_thread.start()

        self.bias_thread = threading.Thread(target=self._bias_control_loop, daemon=True)
        self.bias_thread.start()

    def _agent_home_both_arms(self):
        root = Path(__file__).resolve().parents[1]
        report = run_agent_auto_home(
            self.cone_e,
            torque_config_path=root / "src/configs/pasteur_lid_torque.json",
            audit_path="/var/tmp/piper-agent-collection/auto_home_torque.jsonl",
            control_hz=30.0,
        )
        print("[agent] auto-home " + json.dumps(report, sort_keys=True), flush=True)
        return report

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
                    if self.agent_collection:
                        reply = {'status': 'error', 'message': 'use the audited agent phone UI'}
                    elif arm not in self.xyz_bias:
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
        self._agent_inference_sockets = {}

        endpoints = None
        if self.agent_cycle_enabled:
            endpoints = self.agent_profile.get("inference_endpoints")
            expected = set(self.agent_cycle.tasks)
            if not isinstance(endpoints, dict) or set(endpoints) != expected:
                raise ValueError(
                    "agent cycle requires exactly one inference endpoint for "
                    f"each task {sorted(expected)}"
                )
        if endpoints is None:
            endpoints = {
                self.agent_task or "default": {
                    "obs_port": int(obs_port), "action_port": int(action_port)
                }
            }

        for task_name, endpoint in endpoints.items():
            task_obs_port = int(endpoint["obs_port"])
            task_action_port = int(endpoint["action_port"])
            task_control_port = int(endpoint.get(
                "control_port", task_action_port + 1
            ))
            obs_socket = self.zmq_context.socket(zmq.PUB)
            obs_socket.connect(f"tcp://{hpc_host}:{task_obs_port}")

            action_socket = self.zmq_context.socket(zmq.REQ)
            action_socket.setsockopt(zmq.RCVTIMEO, 2000)
            action_socket.setsockopt(zmq.SNDTIMEO, 2000)
            action_socket.setsockopt(zmq.LINGER, 0)
            action_socket.connect(f"tcp://{hpc_host}:{task_action_port}")

            control_socket = self.zmq_context.socket(zmq.REQ)
            control_socket.setsockopt(zmq.RCVTIMEO, 2000)
            control_socket.setsockopt(zmq.SNDTIMEO, 2000)
            control_socket.setsockopt(zmq.LINGER, 0)
            control_socket.connect(f"tcp://{hpc_host}:{task_control_port}")
            self._agent_inference_sockets[task_name] = {
                "obs": obs_socket,
                "action": action_socket,
                "control": control_socket,
                "ports": (task_obs_port, task_action_port, task_control_port),
            }
            print(
                f"Inference endpoint {task_name}: {hpc_host}:"
                f"{task_obs_port}/{task_action_port}/{task_control_port}"
            )

        initial = self.agent_task if self.agent_task in self._agent_inference_sockets else next(
            iter(self._agent_inference_sockets)
        )
        sockets = self._agent_inference_sockets[initial]
        self.obs_socket = sockets["obs"]
        self.action_socket = sockets["action"]
        self.control_socket = sockets["control"]
        self.control_port = sockets["ports"][2]
        time.sleep(0.5)

    def _activate_agent_inference(self, task):
        if task not in self._agent_inference_sockets:
            raise RuntimeError(f"no inference endpoint configured for {task}")
        sockets = self._agent_inference_sockets[task]
        self.obs_socket = sockets["obs"]
        self.action_socket = sockets["action"]
        self.control_socket = sockets["control"]
        self.control_port = sockets["ports"][2]
        self.episode_manager.control_socket = self.control_socket
        self.agent_task = task
        # Task switches can change the demo-derived terminal transport gate.
        # Construct a fresh latch before the next episode; never carry grasp
        # evidence across tasks.
        self._right_close_latch = GripperCloseLatch(
            **gripper_latch_config(self.agent_profile, task)
        )
        self._agent_post_release_offset = self._agent_release_offset(task)
        self.task = {
            "lid_open": "open the petri dish lid",
            "lid_close": "close the petri dish lid",
        }[task]
        if self.intervention is not None:
            with self.intervention.lock:
                self.intervention.latest_metrics.update({
                    "task": task,
                    "inference_ports": list(sockets["ports"]),
                })
                self.intervention.revision += 1

    def _agent_release_offset(self, task):
        settings = self.agent_profile.get("tasks", {}).get(task, {})
        release = settings.get("gripper_transport_release", {})
        offset = np.asarray(
            release.get("post_release_request_xyz_offset_m", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError(f"invalid post-release offset for {task}")
        return offset

    def get_observation(self):
        timestamp = time.time()

        with self.obs_rpc_lock:
            if self._observation_state_rpc_available is not False:
                try:
                    state = self.obs_cone_e.get_observation_state(
                        active_arm="right" if self.agent_collection else None
                    )
                    self._observation_state_rpc_available = True
                except Exception as error:
                    # Rolling deployment: the collector can be restarted first
                    # while the physical ConeE daemon still serves the older
                    # API.  Fall back safely until its next planned restart.
                    self._observation_state_rpc_available = False
                    print(
                        "[observation] state snapshot RPC unavailable; "
                        f"using legacy calls until restart: {error}",
                        flush=True,
                    )
            if self._observation_state_rpc_available is False:
                state = {
                    "left_ee_pose": self.obs_cone_e.get_left_ee_pose(),
                    "right_ee_pose": self.obs_cone_e.get_right_ee_pose(),
                    "left_gripper_exact": self.obs_cone_e.get_left_gripper_exact(),
                    "right_gripper_exact": self.obs_cone_e.get_right_gripper_exact(),
                    "left_joint_positions": self.obs_cone_e.get_left_joint_positions(),
                    "right_joint_positions": self.obs_cone_e.get_right_joint_positions(),
                }
        ee_pose_left = state["left_ee_pose"]
        ee_pose_right = state["right_ee_pose"]
        left_joint_positions = state["left_joint_positions"]
        right_joint_positions = state["right_joint_positions"]
        # A single-arm agent does not need to block its 30 Hz loop on the
        # inactive Dynamixel.  Keep the last exact inactive value instead.
        left_gripper = state["left_gripper_exact"]
        right_gripper = state["right_gripper_exact"]
        if left_gripper is None:
            left_gripper = self.last_left_gripper
        if right_gripper is None:
            right_gripper = self.last_right_gripper
        self.latest_right_gripper_exact = float(right_gripper)
        self.latest_right_ee_xyz = np.asarray(
            ee_pose_right.translation(), dtype=float
        ).copy()

        left_gripper_binary, right_gripper_binary = self._process_gripper_states(
            left_gripper, right_gripper
        )

        rgb_frame, rgb_timestamp, depth_frame = self.camera.get_latest_frame()
        rgb_frame = self._policy_camera_frame("head", rgb_frame, rgb_timestamp)

        left_wrist_frame = None
        left_wrist_timestamp = np.nan
        if self.left_wrist_camera is not None:
            left_wrist_frame, left_wrist_timestamp, _ = self.left_wrist_camera.get_latest_frame()
            left_wrist_frame = self._policy_camera_frame(
                "left", left_wrist_frame, left_wrist_timestamp
            )

        right_wrist_frame = None
        right_wrist_timestamp = np.nan
        if self.right_wrist_camera is not None:
            right_wrist_frame, right_wrist_timestamp, _ = self.right_wrist_camera.get_latest_frame()
            right_wrist_frame = self._policy_camera_frame(
                "right", right_wrist_frame, right_wrist_timestamp
            )

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

        def frame_id(camera_timestamp):
            try:
                value = float(camera_timestamp)
            except (TypeError, ValueError):
                value = timestamp
            if not np.isfinite(value):
                value = timestamp
            return int(value * 1e9)

        def finite_timestamp(camera_timestamp):
            try:
                return float(camera_timestamp)
            except (TypeError, ValueError):
                return np.nan

        self._latest_ui_frames = {
            "head": (observation["images"]["cam_high"], frame_id(rgb_timestamp)),
            "left": (observation["images"]["cam_left_wrist"], frame_id(left_wrist_timestamp)),
            "right": (observation["images"]["cam_right_wrist"], frame_id(right_wrist_timestamp)),
        }

        sample_monotonic = time.monotonic()
        if (self.agent_collection and self.recorder.is_recording
                and self.intervention.mode == "running"):
            head_recording, left_recording, right_recording = (
                self._copy_agent_recording_frames()
            )
            with self._action_record_lock:
                action_record = {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in self._last_action_record.items()
                }
            sample = AgentRecordingSample(
                wall_timestamp=timestamp,
                active_timestamp=self.recorder.active_timestamp(
                    now=sample_monotonic
                ),
                left_ee_pose=ee_pose_left, right_ee_pose=ee_pose_right,
                left_gripper_exact=left_gripper, right_gripper_exact=right_gripper,
                left_gripper=left_gripper_binary, right_gripper=right_gripper_binary,
                left_joint_positions=np.asarray(left_joint_positions),
                right_joint_positions=np.asarray(right_joint_positions),
                head_rgb=head_recording,
                left_rgb=left_recording,
                right_rgb=right_recording,
                camera_timestamps=(finite_timestamp(rgb_timestamp), finite_timestamp(left_wrist_timestamp), finite_timestamp(right_wrist_timestamp)),
                camera_frame_ids=tuple(self._latest_ui_frames[name][1] for name in ("head", "left", "right")),
                policy_action_quat16=action_record["policy"],
                commanded_target_quat16=action_record["commanded"],
                xyz_bias_left_right=np.concatenate([self.xyz_bias["left"], self.xyz_bias["right"]]),
                chunk_index=int(action_record["chunk_index"]),
                action_generation=int(action_record["generation"]),
                action_observation_timestamp=float(action_record["obs_timestamp"]),
                intervention_revision=self.intervention.correction_revision,
                safety_rejected_count=self.safety.rejected_count,
            )
            self.recorder.record_sample(sample)
            self._agent_sample_index += 1
        elif (not self.agent_collection) and self.recorder and self.recorder.is_recording:
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

    def _policy_camera_frame(self, name, frame, camera_timestamp):
        """Transform a camera image only when its source timestamp changes.

        The head source is slower than the policy loop and its 1920x1440
        rotate/resize costs about 23 ms.  Repeating the last transformed image
        at 30 Hz is correct; the unchanged source timestamp remains attached
        to the observation and recording metadata.
        """
        cache = self._policy_image_cache[name]
        if frame is None:
            return cache["frame"]
        if cache["frame"] is None or camera_timestamp != cache["timestamp"]:
            cache["timestamp"] = camera_timestamp
            cache["frame"] = _rotate_and_resize(frame)
        return cache["frame"]

    def _copy_agent_recording_frames(self):
        """Copy raw source frames under their camera locks.

        Recording remains independent from the resized policy observation, so
        repeated 30 Hz policy frames never pretend that a slower camera
        produced a new image.  The original camera timestamp/frame id records
        that repetition honestly.
        """
        with self.camera.rgb_frame_lock:
            head = (
                None if self.camera.latest_rgb_frame is None
                else self.camera.latest_rgb_frame.copy()
            )
        with self.left_wrist_camera.lock:
            left = (
                None if self.left_wrist_camera.latest_rgb is None
                else self.left_wrist_camera.latest_rgb.copy()
            )
        with self.right_wrist_camera.lock:
            right = (
                None if self.right_wrist_camera.latest_rgb is None
                else self.right_wrist_camera.latest_rgb.copy()
            )
        return head, left, right

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
        observation_rate = 30 if self.agent_collection else 2
        rate_limiter = RateLimiter(
            observation_rate,
            name="agent observation" if self.agent_collection else "observation",
            warn=not self.agent_collection,
        )

        while not self.stop_event.is_set():
            started = time.monotonic()
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
            if (self.agent_collection and self.recorder.is_recording
                    and self.intervention.mode == "running"
                    and time.monotonic() - started > 1.5 / observation_rate):
                self.recorder.note_deadline_miss()
            rate_limiter.sleep()

        print("Observation publishing thread stopped")

    def _agent_frame_provider(self, camera):
        return self._latest_ui_frames.get(camera, (None, None))

    def _agent_camera_health(self, now=None):
        """Return freshness/visibility gates for cycle-required cameras."""
        now = time.monotonic() if now is None else float(now)
        result = {}
        required = (
            self.agent_cycle.required_cameras
            if self.agent_cycle is not None else ("head", "right")
        )
        stale_after_s = float(
            self.agent_profile.get("cycle", {}).get("camera_stale_after_s", 1.0)
        )
        for camera in required:
            frame, frame_id = self._agent_frame_provider(camera)
            visible = (
                frame is not None and frame_id is not None
                and float(np.std(frame)) >= 2.0
            )
            if visible and self._agent_camera_last_id.get(camera) != frame_id:
                self._agent_camera_last_id[camera] = frame_id
                self._agent_camera_last_change_s[camera] = now
            changed_at = self._agent_camera_last_change_s.get(camera)
            result[camera] = bool(
                visible and changed_at is not None
                and now - changed_at <= stale_after_s
            )
        return result

    def _refresh_agent_target_selection(self):
        """Carry normalized metadata to the next task with fresh provenance."""
        with self.intervention.lock:
            previous = dict(self.intervention.target_selection or {})
        if not previous:
            raise RuntimeError("cycle has no target-selection metadata")
        frame, frame_id = self._agent_frame_provider("head")
        if frame is None or frame_id is None or np.std(frame) < 2.0:
            raise RuntimeError("required head camera is missing/blank")
        previous.update({
            "frame_id": int(frame_id),
            "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "selected_at": time.time(),
            "cycle_reused_normalized_selection": True,
        })
        with self.intervention.lock:
            self.intervention.target_selection = previous
        return previous

    def _reset_agent_episode_runtime(self):
        self._agent_sample_index = 0
        self._resume_after_obs_timestamp = None
        self._resume_anchor_xyz = None
        self._resume_anchor_offset[:] = 0
        self._right_close_latch.reset()
        self.starting_pose_left = self.starting_pose_right = None
        self._buffer_gen = None
        self.safety.reset()
        self.intervention.correction_revision = 0
        with self._action_record_lock:
            self._last_action_record = {
                "policy": np.full(16, np.nan),
                "commanded": np.full(16, np.nan),
                "chunk_index": -1, "generation": -1,
                "obs_timestamp": np.nan,
            }

    def _start_cycle_episode(self, task):
        camera_health = self._agent_camera_health()
        missing = [name for name, ready in camera_health.items() if not ready]
        if missing:
            raise RuntimeError("required camera is missing/stale: " + ",".join(missing))
        selection = self._refresh_agent_target_selection()
        self.recorder.configure_episode({
            "task": task,
            "target_selection": selection,
            "initial_bias_m": {
                arm: bias.tolist() for arm, bias in self.xyz_bias.items()
            },
            "policy_mode": "absolute", "control_frequency_hz": 30,
            "automatic_cycle": True,
            "bias_attempt_index": self.agent_cycle.current_bias_attempt_index,
            "bias_offset_m": list(self.agent_cycle.current_bias_offset_m),
        })
        self._reset_agent_episode_runtime()
        self.episode_manager.start_episode()
        if not self.episode_manager.is_active():
            raise RuntimeError("episode manager refused automatic cycle start")
        self.recorder.set_sampling_active(True)
        self.agent_cycle.begin_episode(
            safety_rejected_count=self.safety.rejected_count
        )
        with self.intervention.lock:
            self.intervention.mode = "running"
            self.intervention.latest_metrics.update({
                "task": task,
                "gripper_close_latched": False,
                "gripper_release_enabled": False,
                "gripper_released_after_transport": False,
                "gripper_transport_drop_detected": False,
                "cycle_state": "running",
                "bias_attempt_index": self.agent_cycle.current_bias_attempt_index,
                "bias_offset_m": list(self.agent_cycle.current_bias_offset_m),
            })
            self.intervention.revision += 1

    def _hold_agent_measured_state(self):
        for arm in ("left", "right"):
            measured = np.asarray(
                getattr(self.cone_e, f"get_{arm}_joint_positions")(), dtype=float
            )
            getattr(self.cone_e, f"set_{arm}_joint_target")(
                measured, gripper_target=None, preview_time=0.10
            )

    def _stop_agent_cycle(self, reason):
        """Fail closed: save failure, hold measured state, never auto-retry."""
        if self.episode_manager.is_active():
            self.recorder.log_event("cycle_stopped", reason=str(reason))
            self.episode_manager.end_episode(
                reason=f"cycle_stop:{reason}", home_after=False
            )
            if self.recorder.episode_dir is not None:
                failure_reason = (
                    str(reason) if str(reason) in FAILURE_REASONS else "abort"
                )
                destination = self.recorder.finalize(
                    "failure", reason=failure_reason
                )
            else:
                destination = None
        else:
            destination = None
        try:
            self.episode_manager.clear_action_queue()
        finally:
            self._hold_agent_measured_state()
        if self.agent_cycle is not None and self.agent_cycle.enabled:
            self.agent_cycle.stop(str(reason))
        with self.intervention.lock:
            self.intervention.mode = "cycle_stopped"
            self.intervention.latest_metrics.update({
                "cycle_state": "stopped",
                "cycle_stop_reason": str(reason),
                "last_episode": None if destination is None else str(destination),
            })
            self.intervention.last_error = str(reason)
            self.intervention.revision += 1

    @staticmethod
    def _cycle_failure_manifest_reason(reason):
        reason = str(reason)
        return reason if reason in FAILURE_REASONS else "abort"

    def _retry_or_exhaust_agent_cycle(self, decision):
        """Save a failed attempt, guarded-home, then retry the same task.

        Bias is applied after homing and before the next recorder/episode is
        started.  If the finite configured schedule is exhausted, the final
        failure is still saved and pressure-homed before the cycle is disabled.
        """
        if decision.action not in {"retry_failure", "stop_retry_exhausted"}:
            raise ValueError("decision is not a bias retry transition")
        self._agent_cycle_transition_active = True
        task = decision.task
        try:
            self.recorder.log_event(
                "cycle_attempt_failed",
                failure_reason=decision.reason,
                task=task,
                bias_attempt_index=self.agent_cycle.current_bias_attempt_index,
                bias_m=self.xyz_bias["right"].tolist(),
            )
            self.episode_manager.end_episode(
                reason=f"cycle_retry:{decision.reason}", home_after=False
            )
            destination = self.recorder.finalize(
                "failure",
                reason=self._cycle_failure_manifest_reason(decision.reason),
            )
            self.episode_manager.clear_action_queue()
            with self.intervention.lock:
                self.intervention.mode = "cycle_homing"
                self.intervention.latest_metrics.update({
                    "cycle_state": "homing_after_failure",
                    "last_episode": str(destination),
                    "last_failure_reason": decision.reason,
                })
                self.intervention.revision += 1

            report = self._agent_home_both_arms()
            if decision.action == "stop_retry_exhausted":
                stopped = self.agent_cycle.stop_after_retry_exhausted(decision)
                with self.intervention.lock:
                    self.intervention.mode = "cycle_stopped"
                    self.intervention.latest_metrics.update({
                        "cycle_state": "stopped",
                        "cycle_stop_reason": stopped.reason,
                        "auto_home": report,
                    })
                    self.intervention.last_error = stopped.reason
                    self.intervention.revision += 1
                return

            self.agent_cycle.advance_after_retry(decision)
            offset = np.asarray(decision.bias_offset_m, dtype=float)
            next_bias = self._agent_task_bias[task] + offset
            self.set_bias("right", next_bias)
            self.intervention.set_bias("right", next_bias)
            self.episode_manager.clear_action_queue()
            with self.intervention.lock:
                self.intervention.latest_metrics.update({
                    "auto_home": report,
                    "retry_task": task,
                    "bias_attempt_index": decision.bias_attempt_index,
                    "bias_offset_m": offset.tolist(),
                    "retry_bias_m": next_bias.tolist(),
                })
            self._start_cycle_episode(task)
        except Exception as error:
            self._stop_agent_cycle(
                f"cycle_retry_transition:{type(error).__name__}:{error}"
            )
        finally:
            self._agent_cycle_transition_active = False

    def _complete_and_advance_agent_cycle(self, decision):
        """Promote success, guarded-home, route the next ACT, and restart."""
        self._agent_cycle_transition_active = True
        current_task = decision.task
        next_task = decision.next_task
        try:
            self.recorder.log_event(
                "automatic_terminal_release",
                next_task=next_task,
                transport_displacement_m=(
                    self._right_close_latch.transport_displacement_m
                ),
            )
            self.episode_manager.end_episode(
                reason="released_after_transport", home_after=False
            )
            destination = self.recorder.finalize("success")
            self.episode_manager.clear_action_queue()
            self._agent_task_bias[current_task] = self.xyz_bias["right"].copy()
            with self.intervention.lock:
                self.intervention.mode = "cycle_homing"
                self.intervention.latest_metrics.update({
                    "cycle_state": "homing",
                    "last_episode": str(destination),
                    "next_task": next_task,
                })
                self.intervention.revision += 1

            report = self._agent_home_both_arms()
            self.agent_cycle.advance_after_success()
            self._activate_agent_inference(next_task)
            self.set_bias("right", self._agent_task_bias[next_task])
            self.intervention.set_bias("right", self.xyz_bias["right"])
            self.episode_manager.clear_action_queue()
            with self.intervention.lock:
                self.intervention.latest_metrics.update({
                    "auto_home": report,
                    "success_count": self.recorder.success_count(next_task),
                })
            self._start_cycle_episode(next_task)
        except Exception as error:
            self._stop_agent_cycle(f"cycle_transition:{type(error).__name__}:{error}")
        finally:
            self._agent_cycle_transition_active = False

    def _evaluate_agent_cycle(self):
        if (not self.agent_cycle_enabled or self.agent_cycle is None
                or not self.episode_manager.is_active()
                or self._agent_cycle_transition_active):
            return False
        explicit_failure = None
        grasp_deadline = float(
            self.agent_profile.get("cycle", {}).get(
                "grasp_confirmation_deadline_s", 0.0
            )
        )
        if (grasp_deadline > 0 and self.agent_cycle.episode_started_at is not None
                and time.monotonic() - self.agent_cycle.episode_started_at
                >= grasp_deadline
                and not self._right_close_latch.latched):
            explicit_failure = "grasp_miss"
        if self._right_close_latch.dropped:
            explicit_failure = "drop"
        decision = self.agent_cycle.evaluate(
            terminal_release=self._right_close_latch.released,
            camera_ready=self._agent_camera_health(),
            safety_rejected_count=self.safety.rejected_count,
            explicit_failure=explicit_failure,
        )
        if decision is None:
            return False
        if decision.action == "complete_success":
            self._complete_and_advance_agent_cycle(decision)
        elif decision.action in {"retry_failure", "stop_retry_exhausted"}:
            self._retry_or_exhaust_agent_cycle(decision)
        else:
            self._stop_agent_cycle(decision.reason)
        return True

    def _submit_agent_command(self, command, payload):
        allowed = {"home", "select_target", "start", "pause", "nudge", "resume", "success", "failure"}
        if command not in allowed:
            raise ValueError(f"unknown command {command!r}")
        request_id = self.intervention.submit(command, payload)
        return self.intervention.wait(
            request_id, timeout_s=45.0 if command == "home" else (
                12.0 if command in {"success", "failure"} else 4.0
            )
        )

    def _clear_actions_after(self, minimum_obs_timestamp):
        self.control_socket.send_pyobj({
            "command": "clear_queue",
            "minimum_obs_timestamp": float(minimum_obs_timestamp),
        })
        response = self.control_socket.recv_pyobj()
        if response.get("status") != "ok":
            raise RuntimeError(response.get("message", "failed to clear action queue"))

    @staticmethod
    def _action_quat16(action):
        output = np.full(16, np.nan, dtype=float)
        if action.get("left_ee_pose") is not None:
            output[:7] = np.asarray(action["left_ee_pose"], dtype=float)
        if action.get("right_ee_pose") is not None:
            output[7:14] = np.asarray(action["right_ee_pose"], dtype=float)
        if action.get("left_gripper") is not None:
            output[14] = float(action["left_gripper"])
        if action.get("right_gripper") is not None:
            output[15] = float(action["right_gripper"])
        return output

    def _process_agent_command(self):
        if not self.agent_collection:
            return
        with self.intervention.lock:
            command = self.intervention.requested
            payload = dict(self.intervention.request_payload)
            request_id = self.intervention.request_id
        if command is None:
            return
        try:
            mode = self.intervention.mode
            metrics = {}
            if command == "home":
                if mode != "idle" or self.episode_manager.is_active():
                    raise RuntimeError("HOME is accepted only while idle")
                self.episode_manager.clear_action_queue()
                report = self._agent_home_both_arms()
                self.starting_pose_left = self.starting_pose_right = None
                self.safety.reset()
                metrics["auto_home"] = report
            elif command == "select_target":
                if mode != "idle":
                    raise RuntimeError("target can only be selected while idle")
                u, v = float(payload["u"]), float(payload["v"])
                if not (0 <= u <= 1 and 0 <= v <= 1):
                    raise ValueError("target u/v must be normalized to [0,1]")
                frame, frame_id = self._agent_frame_provider("head")
                selection = {
                    "camera": "head", "u": u, "v": v,
                    "frame_id": int(frame_id) if frame_id is not None else None,
                    "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest() if frame is not None else None,
                    "selected_at": time.time(),
                }
                with self.intervention.lock:
                    self.intervention.target_selection = selection
                metrics["target_selected"] = True
            elif command == "start":
                if mode != "idle" or self.episode_manager.is_active():
                    raise RuntimeError("episode is not idle")
                if self.intervention.target_selection is None:
                    raise RuntimeError("tap the target in the head image first")
                for camera in ("head", "right"):
                    frame, frame_id = self._agent_frame_provider(camera)
                    if frame is None or frame_id is None or np.std(frame) < 2.0:
                        raise RuntimeError(f"required {camera} camera is missing/blank")
                task = payload.get("task", self.agent_task)
                if task != self.agent_task:
                    raise RuntimeError(f"controller is configured for {self.agent_task}, not {task}")
                self.recorder.configure_episode({
                    "task": task,
                    "target_selection": dict(self.intervention.target_selection),
                    "initial_bias_m": {arm: bias.tolist() for arm, bias in self.xyz_bias.items()},
                    "policy_mode": "absolute", "control_frequency_hz": 30,
                })
                self._agent_sample_index = 0
                self._resume_anchor_xyz = None
                self._resume_anchor_offset[:] = 0
                self._right_close_latch.reset()
                metrics["gripper_close_latched"] = False
                metrics["gripper_release_enabled"] = False
                metrics["gripper_release_commanded"] = False
                metrics["gripper_released_after_transport"] = False
                metrics["gripper_transport_drop_detected"] = False
                self.intervention.correction_revision = 0
                with self._action_record_lock:
                    self._last_action_record = {
                        "policy": np.full(16, np.nan),
                        "commanded": np.full(16, np.nan),
                        "chunk_index": -1, "generation": -1,
                        "obs_timestamp": np.nan,
                    }
                self.episode_manager.start_episode()
                self.recorder.set_sampling_active(True)
                if self.agent_cycle_enabled:
                    self.agent_cycle.begin_episode(
                        safety_rejected_count=self.safety.rejected_count
                    )
                    metrics["cycle_state"] = "running"
                mode = "running"
            elif command == "pause":
                if mode not in {"running", "awaiting_fresh_action"}:
                    raise RuntimeError("pause requires a running episode")
                with self.intervention.lock:
                    self.intervention.mode = "paused"
                self.recorder.set_sampling_active(False)
                pose = self.cone_e.get_right_ee_pose()
                gripper = self.cone_e.get_right_gripper_exact()
                self.cone_e.set_right_ee_target(ee_target=pose, gripper_target=gripper, preview_time=0.05)
                self.episode_manager.clear_action_queue()
                self.recorder.log_event("paused", bias=self.xyz_bias["right"].tolist())
                mode = "paused"
            elif command == "nudge":
                if mode not in {"idle", "paused"}:
                    raise RuntimeError("nudge is accepted only while idle or paused")
                arm = "right"
                axis = payload.get("axis")
                if axis not in {"x", "y", "z"}:
                    raise ValueError("axis must be x, y, or z")
                step_mm = int(payload.get("step_mm", 2))
                direction = int(payload.get("direction", 0))
                if step_mm not in {1, 2, 5} or direction not in {-1, 1}:
                    raise ValueError("step must be 1/2/5 mm and direction ±1")
                bias = self.xyz_bias[arm].copy()
                bias[{"x": 0, "y": 1, "z": 2}[axis]] += direction * step_mm / 1000.0
                self.intervention.set_bias(arm, bias)
                self.set_bias(arm, bias)
                # A phone direction button is a real Cartesian jog, not merely
                # a future absolute-policy bias. Keep the measured gripper
                # opening, including a successfully grasped object.
                pose = self.cone_e.get_right_ee_pose()
                delta = np.zeros(3, dtype=float)
                delta[{"x": 0, "y": 1, "z": 2}[axis]] = direction * step_mm / 1000.0
                target_xyz = pose.translation() + delta
                safe_xyz = self.safety.check("right", target_xyz)
                if safe_xyz is None:
                    raise RuntimeError("Cartesian nudge rejected by pressure/motion safety")
                target = mink.SE3(np.r_[pose.rotation().wxyz, safe_xyz])
                gripper = float(self.cone_e.get_right_gripper_exact())
                self.cone_e.set_right_ee_target(
                    ee_target=target, gripper_target=gripper, preview_time=0.20
                )
                self.intervention.correction_revision += 1
                if self.recorder.is_recording:
                    self.recorder.log_event(
                        "bias_nudge", arm=arm, axis=axis, direction=direction,
                        step_mm=step_mm, bias_m=bias.tolist(),
                        direct_cartesian_target_xyz=safe_xyz.tolist(),
                        correction_revision=self.intervention.correction_revision,
                    )
                metrics["last_nudge"] = {
                    "arm": arm, "axis": axis,
                    "direction": direction, "step_mm": step_mm,
                }
            elif command == "resume":
                if mode != "paused":
                    raise RuntimeError("resume requires paused mode")
                self._resume_after_obs_timestamp = time.time()
                self._resume_anchor_xyz = self.cone_e.get_right_ee_pose().translation().copy()
                self._resume_anchor_offset[:] = 0
                self._clear_actions_after(self._resume_after_obs_timestamp)
                self.starting_pose_left = self.starting_pose_right = None
                self.safety.reset()
                self.recorder.log_event("resumed", fresh_after=self._resume_after_obs_timestamp)
                mode = "awaiting_fresh_action"
            elif command in {"success", "failure"}:
                if not self.episode_manager.is_active():
                    raise RuntimeError("no active episode")
                if command == "failure" and payload.get("reason") not in FAILURE_REASONS:
                    raise ValueError("select a supported failure reason")
                if self.agent_cycle_enabled:
                    if command == "success":
                        raise RuntimeError(
                            "cycle success is automatic after measured terminal release"
                        )
                    failure_reason = str(payload.get("reason"))
                    decision = self.agent_cycle.evaluate(
                        terminal_release=False,
                        camera_ready=self._agent_camera_health(),
                        safety_rejected_count=self.safety.rejected_count,
                        explicit_failure=failure_reason,
                    )
                    if decision is None:
                        raise RuntimeError("cycle did not accept explicit failure")
                    if decision.action in {
                        "retry_failure", "stop_retry_exhausted"
                    }:
                        self._retry_or_exhaust_agent_cycle(decision)
                    else:
                        self._stop_agent_cycle(
                            f"operator_failure:{failure_reason}"
                        )
                    mode = self.intervention.mode
                    metrics["cycle_state"] = self.intervention.latest_metrics.get(
                        "cycle_state", mode
                    )
                    self.intervention.complete(
                        request_id, mode=mode, **metrics
                    )
                    return
                self.episode_manager.end_episode(reason=command)
                destination = self.recorder.finalize(command, reason=payload.get("reason"))
                self.episode_manager.clear_action_queue()
                mode = "idle"
                metrics["last_episode"] = str(destination)
                metrics["success_count"] = self.recorder.success_count(self.agent_task)
                with self.intervention.lock:
                    self.intervention.target_selection = None
                self._right_close_latch.reset()
                self._resume_anchor_xyz = None
                self._resume_anchor_offset[:] = 0
            self.intervention.complete(request_id, mode=mode, **metrics)
        except Exception as error:
            self.intervention.complete(request_id, error=str(error))

    def request_action(self):
        try:
            self.action_socket.send_pyobj({'request': 'action'})
            action = self.action_socket.recv_pyobj()
            if 'error' in action:
                print(f"Server error: {action['error']}")
                if self.agent_cycle is not None:
                    self.agent_cycle.note_inference(False)
                return None
            if self.agent_cycle is not None:
                self.agent_cycle.note_inference(True)
            self.stats['actions_received'] += 1
            if action.get('is_stale', False):
                self.stats['buffer_wraps'] += 1
            return action
        except zmq.error.Again:
            if self.agent_cycle is not None:
                self.agent_cycle.note_inference(False)
            now = time.time()
            if not hasattr(self, '_last_timeout_warning') or now - self._last_timeout_warning > 2.0:
                print("Timeout waiting for action from server")
                self._last_timeout_warning = now
            self.stats['errors'] += 1
            return None
        except Exception as e:
            if self.agent_cycle is not None:
                self.agent_cycle.note_inference(False)
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

        if self.agent_collection:
            if action.get("left_delta_pose") is not None or action.get("right_delta_pose") is not None:
                raise RuntimeError("agent collection refuses delta ACT; use absolute actions")
            obs_timestamp = action.get("obs_timestamp")
            if self._resume_after_obs_timestamp is not None:
                if obs_timestamp is None or float(obs_timestamp) < self._resume_after_obs_timestamp:
                    return
                self._resume_after_obs_timestamp = None
                self.recorder.set_sampling_active(True)
                with self.intervention.lock:
                    self.intervention.mode = "running"
                    self.intervention.revision += 1
            with self._action_record_lock:
                self._last_action_record.update({
                    "policy": self._action_quat16(action),
                    "chunk_index": int(action.get("chunk_index", -1)),
                    "generation": int(action.get("total_buffer_updates", -1)),
                    "obs_timestamp": float(obs_timestamp) if obs_timestamp is not None else np.nan,
                })

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

        if self.agent_collection and self.starting_pose_right is None:
            self.starting_pose_right = self._biased_pose(
                self.cone_e.get_right_ee_pose(), 'right')
            self.safety.reset()
        elif (not self.agent_collection) and (self.starting_pose_left is None or self.starting_pose_right is None):
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

        if (not self.agent_collection) and 'left_delta_pose' in action and action['left_delta_pose'] is not None:
            self.starting_pose_left = self._apply_arm_action(
                'left', action['left_delta_pose'], action.get('left_gripper', 0.5),
                self.starting_pose_left, self.cone_e.set_left_ee_target
            )
        elif (not self.agent_collection) and 'left_ee_pose' in action and action['left_ee_pose'] is not None:
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
        """Set an arm's finite xyz bias in metres in the robot frame."""
        b = np.asarray(bias, dtype=float).reshape(3)
        if not np.all(np.isfinite(b)):
            raise ValueError(f"bias must contain three finite values, got {bias!r}")
        if np.any(np.abs(b) > 0.06 + 1e-12):
            raise ValueError("bias is limited to ±0.06 m")
        self.xyz_bias[arm] = b
        # Changing the bias jumps the next target by the delta -- a legitimate
        # discontinuity, not a runaway. Drop the step reference so the safety
        # layer doesn't reject the frame right after a live bias change.
        self.safety.reset(arm)
        print(f"[bias] {arm} = {np.round(b, 4)} m", flush=True)
        return b.copy()

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
        if arm == "right" and self._resume_anchor_xyz is not None:
            if not np.any(self._resume_anchor_offset):
                self._resume_anchor_offset = self._resume_anchor_xyz - p_target
                if self.recorder.is_recording:
                    self.recorder.log_event(
                        "resume_anchor",
                        offset_m=self._resume_anchor_offset.tolist(),
                    )
            p_target = p_target + self._resume_anchor_offset
        R_target = X_Rtarget.rotation()

        # Preserve the successful grasp pose.  A placement correction is
        # introduced only after ACT asks for its demonstrated final release;
        # the latch keeps the jaws closed for the configured settling time.
        if (arm == "right" and self._right_close_latch_enabled
                and self._right_close_latch.release_requested_at is not None):
            p_target = p_target + self._agent_post_release_offset

        p_safe = self.safety.check(arm, p_target)
        if p_safe is None:
            return

        target = mink.SE3(np.concatenate([R_target.wxyz, p_safe]))
        gripper_command = float(gripper)
        if arm == "right" and self._right_close_latch_enabled:
            drop_was_detected = self._right_close_latch.dropped
            release_was_enabled = self._right_close_latch.release_enabled
            release_was_commanded = self._right_close_latch.release_commanded
            release_was_completed = self._right_close_latch.released
            gripper_command, newly_latched = self._right_close_latch.apply(
                gripper_command, self.latest_right_gripper_exact,
                measured_xyz=self.latest_right_ee_xyz,
            )
            if newly_latched:
                grasp_xyz = self._right_close_latch.grasp_xyz_m
                self.recorder.log_event(
                    "gripper_close_latched",
                    policy_gripper=float(gripper),
                    held_gripper_target=gripper_command,
                    grasp_xyz_m=None if grasp_xyz is None else grasp_xyz.tolist(),
                    grasp_measured_opening=(
                        self._right_close_latch.grasp_measured_opening
                    ),
                )
                with self.intervention.lock:
                    self.intervention.latest_metrics["gripper_close_latched"] = True
                    self.intervention.revision += 1
            if not drop_was_detected and self._right_close_latch.dropped:
                self.recorder.log_event(
                    "gripper_transport_drop_detected",
                    grasp_measured_opening=(
                        self._right_close_latch.grasp_measured_opening
                    ),
                    measured_opening=float(self.latest_right_gripper_exact),
                    opening_decrease=(
                        self._right_close_latch.drop_opening_decrease
                    ),
                    minimum_opening_decrease=(
                        self._right_close_latch.drop_opening_decrease_threshold
                    ),
                    confirmation_elapsed_s=(
                        self._right_close_latch.drop_confirmation_elapsed_s
                    ),
                    required_confirmation_s=(
                        self._right_close_latch.drop_confirmation_time_s
                    ),
                    transport_displacement_m=(
                        self._right_close_latch.transport_displacement_m
                    ),
                )
                with self.intervention.lock:
                    self.intervention.latest_metrics.update({
                        "gripper_transport_drop_detected": True,
                        "gripper_drop_opening_decrease": (
                            self._right_close_latch.drop_opening_decrease
                        ),
                    })
                    self.intervention.revision += 1
            if (not self._right_close_latch.dropped
                    and not release_was_enabled
                    and self._right_close_latch.release_enabled):
                grasp_xyz = self._right_close_latch.grasp_xyz_m
                measured_xyz = self._right_close_latch.latest_xyz_m
                self.recorder.log_event(
                    "gripper_release_enabled",
                    grasp_xyz_m=None if grasp_xyz is None else grasp_xyz.tolist(),
                    measured_xyz_m=(
                        None if measured_xyz is None else measured_xyz.tolist()
                    ),
                    transport_displacement_m=(
                        self._right_close_latch.transport_displacement_m
                    ),
                    minimum_transport_distance_m=(
                        self._right_close_latch.minimum_transport_distance_m
                    ),
                )
                with self.intervention.lock:
                    self.intervention.latest_metrics.update({
                        "gripper_release_enabled": True,
                        "gripper_transport_displacement_m": (
                            self._right_close_latch.transport_displacement_m
                        ),
                    })
                    self.intervention.revision += 1
            if not release_was_commanded and self._right_close_latch.release_commanded:
                self.recorder.log_event(
                    "gripper_release_commanded",
                    policy_gripper=float(gripper),
                    commanded_gripper=gripper_command,
                    transport_displacement_m=(
                        self._right_close_latch.transport_displacement_m
                    ),
                )
                with self.intervention.lock:
                    self.intervention.latest_metrics[
                        "gripper_release_commanded"
                    ] = True
                    self.intervention.revision += 1
            if not release_was_completed and self._right_close_latch.released:
                self.recorder.log_event(
                    "gripper_released_after_transport",
                    policy_gripper=float(gripper),
                    commanded_gripper=gripper_command,
                    transport_displacement_m=(
                        self._right_close_latch.transport_displacement_m
                    ),
                )
                with self.intervention.lock:
                    self.intervention.latest_metrics.update({
                        "gripper_close_latched": False,
                        "gripper_released_after_transport": True,
                    })
                    self.intervention.revision += 1
        set_target_fn(
            ee_target=target,
            gripper_target=gripper_command, preview_time=0.5,
        )
        if self.agent_collection:
            commanded = np.full(16, np.nan, dtype=float)
            section = slice(0, 7) if arm == "left" else slice(7, 14)
            commanded[section] = np.concatenate([target.rotation().wxyz, target.translation()])
            commanded[14 if arm == "left" else 15] = gripper_command
            with self._action_record_lock:
                self._last_action_record["commanded"] = commanded

    def control_loop(self, control_rate=30):
        if self.agent_collection and control_rate != 30:
            raise ValueError("agent collection must run at exactly 30 Hz")
        rate_limiter = RateLimiter(
            control_rate,
            name="agent control" if self.agent_collection else "control",
            warn=not self.agent_collection,
        )
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
            self._process_agent_command()
            self.episode_manager.check_autonomous_conditions(self.manipulability_calc, iteration)

            if not self.episode_manager.is_active():
                rate_limiter.sleep()
                continue

            if self.agent_collection and self.intervention.mode == "paused":
                rate_limiter.sleep()
                continue

            if self._evaluate_agent_cycle():
                rate_limiter.sleep()
                continue

            action = self.request_action()
            if action is not None:
                self.apply_action(action)
                if self._evaluate_agent_cycle():
                    rate_limiter.sleep()
                    continue
                if iteration % 30 == 0:
                    self._print_status(action, loop_start)
                wait_for_ready_count = 0
            else:
                if self._evaluate_agent_cycle():
                    rate_limiter.sleep()
                    continue
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
        print(f"Observations publishing at {30 if self.agent_collection else 2} Hz in background")
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
            self.episode_manager.end_episode(
                reason="shutdown", home_after=not self.agent_cycle_enabled
            )
            if self.agent_collection and self.recorder.episode_dir is not None:
                self.recorder.finalize("failure", reason="abort")
        self.stop_event.set()
        if self.agent_ui:
            self.agent_ui.stop()
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
        if self.keyboard:
            self.keyboard.stop()
        closed = set()
        for endpoint in self._agent_inference_sockets.values():
            for socket in (endpoint["obs"], endpoint["action"], endpoint["control"]):
                if id(socket) not in closed:
                    socket.close()
                    closed.add(id(socket))
        self.zmq_context.term()
        if self.controller_claim:
            self.controller_claim.release()
        print("Policy controller stopped")
