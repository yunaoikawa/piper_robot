import zmq
import numpy as np
import time
from typing import Tuple
from dataclasses import dataclass
from loop_rate_limiters import RateLimiter
import threading

import mujoco
import mujoco.viewer
import mink
from mink.lie import SE3, SO3

from scipy.spatial.transform.rotation import Rotation as R

from robot.arm.fps_counter import FPSCounter
from robot.network import VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC


@dataclass
class ControllerState:
    created_timestamp: float
    left_x: bool
    left_y: bool
    left_menu: bool
    left_thumbstick: bool
    left_index_trigger: float
    left_hand_trigger: float
    left_thumbstick_axes: np.ndarray
    left_local_position: np.ndarray
    left_local_rotation: np.ndarray

    right_a: bool
    right_b: bool
    right_menu: bool
    right_thumbstick: bool
    right_index_trigger: float
    right_hand_trigger: float
    right_thumbstick_axes: np.ndarray
    right_local_position: np.ndarray
    right_local_rotation: np.ndarray

    @property
    def left_SE3(self) -> SE3:
        # convert left-handed to right-handed
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        rotation_mat = M @ R.from_quat(self.left_local_rotation).as_matrix() @ M.T
        translation = self.left_local_position * np.array([1, 1, -1])
        return SE3.from_rotation_and_translation(rotation=SO3.from_matrix(rotation_mat), translation=translation)

    @property
    def right_SE3(self) -> SE3:
        # convert left-handed to right-handed
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        rotation_mat = M @ R.from_quat(self.right_local_rotation).as_matrix() @ M.T
        translation = self.right_local_position * np.array([1, 1, -1])
        return SE3.from_rotation_and_translation(rotation=SO3.from_matrix(rotation_mat), translation=translation)


def parse_controller_state(controller_state_string: str) -> ControllerState:
    left_data, right_data = controller_state_string.split("|")

    left_data = left_data.split(";")[1:-1]
    right_data = right_data.split(";")[1:-1]

    def parse_bool(val: str) -> bool:
        return val.split(":")[1].lower().strip() == "true"

    def parse_float(val: str) -> float:
        return float(val.split(":")[1])

    def parse_list_float(val: str) -> np.ndarray:
        return np.array(list(map(float, val.split(":")[1].split(","))))

    def parse_section(data: list) -> Tuple:
        return (
            # Buttons
            parse_bool(data[0]),
            parse_bool(data[1]),
            parse_bool(data[2]),
            parse_bool(data[3]),
            # Triggers
            parse_float(data[4]),
            parse_float(data[5]),
            # Thumbstick
            parse_list_float(data[6]),
            # Pose
            parse_list_float(data[7]),
            parse_list_float(data[8]),
        )

    left_parsed = parse_section(left_data)
    right_parsed = parse_section(right_data)

    return ControllerState(time.time(), *left_parsed, *right_parsed)


def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


def create_subscriber_socket(host, port, topic):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    # CANNOT CONFLATE, because messages come multi-part
    socket.connect("tcp://{}:{}".format(host, port))
    socket.subscribe(topic)
    return socket


# This class is used to detect the hand keypoints from the VR and publish them.
class OculusReader:
    def __init__(self):
        # Create a subscriber socket
        self.controller_state_lock_ = threading.Lock()
        self.listen_stop_event_ = threading.Event()
        self.listen_thread = threading.Thread(target=self.listen_for_controller, daemon=True)
        self.controller_state = None

        self.listen_thread.start()

    def listen_for_controller(self):
        stick_socket = create_subscriber_socket(VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC)
        while not self.listen_stop_event_.is_set():
            _, message = stick_socket.recv_multipart()
            # print(message)
            with self.controller_state_lock_:
                self.controller_state = parse_controller_state(message.decode())

    # Function to publish the left/right hand keypoints and button Feedback
    def stream(self):
        print("oculus stick stream")
        model = mujoco.MjModel.from_xml_path("../arm/mujoco/scene_piper.xml")
        data = mujoco.MjData(model)

        model.body_gravcomp[:] = float(True)

        joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        velocity_limits = {k: np.pi/2 if "joint" in k else 0.05 for k in joint_names}

        dof_ids = np.array([model.joint(name).id for name in joint_names])
        actuator_ids = np.array([model.actuator(name + "_pos").id for name in joint_names])

        configuration = mink.Configuration(model)

        end_effector_task = mink.FrameTask(
            frame_name="ee",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.1,
            lm_damping=1.0,
        )

        posture_task = mink.PostureTask(
            model, cost=np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
        )

        tasks = [end_effector_task, posture_task]

        limits = [mink.ConfigurationLimit(model), mink.VelocityLimit(model, velocity_limits)]

        solver = "quadprog"
        pos_threshold = 1e-2
        ori_threshold = 1e-2
        max_iters = 20

        X_ee_init = SE3.from_mocap_name(model, data, "pinch_site_target")
        H = SE3.from_rotation(SO3.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])))

        start_teleop = False
        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            configuration.update(data.qpos)
            mujoco.mj_forward(model, data)
            posture_task.set_target_from_configuration(configuration)

            # Initialize the mocap target at the end-effector site.
            mink.move_mocap_to_frame(model, data, "pinch_site_target", "ee", "site")

            rate = RateLimiter(frequency=300.0, warn=True, name="oculus teleop")
            fps_counter = FPSCounter()
            dt = rate.dt
            t = 0.0

            while viewer.is_running():
                with self.controller_state_lock_:
                    state = self.controller_state

                if state is None: continue

                if state.right_a:
                    # print("start teleop")
                    X_Cinit = state.right_SE3
                    start_teleop = True

                if state.right_b:
                    # print("pause teleop")
                    X_ee_init = SE3.from_mocap_name(model, data, "pinch_site_target")
                    start_teleop = False

                if start_teleop:
                    X_Ctarget = state.right_SE3
                    X_Cdelta = X_Cinit.inverse().multiply(X_Ctarget)
                    X_Rdelta = H.inverse() @ X_Cdelta @ H

                    # translation
                    p_REt = X_ee_init.translation() + X_Rdelta.translation()

                    # rotation
                    R_REt = X_ee_init.rotation() @ X_Rdelta.rotation()
                    data.mocap_pos[model.body("pinch_site_target").mocapid[0]] = p_REt
                    data.mocap_quat[model.body("pinch_site_target").mocapid[0]] = R_REt.wxyz

                T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target") # TODO: might be coming one step late
                end_effector_task.set_target(T_wt)

                for i in range(max_iters):
                    vel = mink.solve_ik(configuration, tasks, rate.dt, solver, damping=1e-12, limits=limits)
                    configuration.integrate_inplace(vel, rate.dt)

                    # Exit condition
                    pos_achieved = True
                    ori_achieved = True
                    err = end_effector_task.compute_error(configuration)
                    pos_achieved &= bool(np.linalg.norm(err[:3]) <= pos_threshold)
                    ori_achieved &= bool(np.linalg.norm(err[3:]) <= ori_threshold)
                    if pos_achieved and ori_achieved:
                        break

                # data.ctrl[actuator_ids] = configuration.q[dof_ids]
                data.qpos[actuator_ids] = configuration.q[dof_ids]
                mujoco.mj_forward(model, data)

                fps = fps_counter.tick()
                if fps is not None: print(f"{fps:.2f}")
                viewer.sync()
                rate.sleep()
                t += dt


def main():
    oculus_reader = OculusReader()
    oculus_reader.stream()


if __name__ == "__main__":
    main()
