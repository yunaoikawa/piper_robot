import zmq
import numpy as np
import time
from typing import Tuple
from dataclasses import dataclass
from loop_rate_limiters import RateLimiter

import mujoco
import mujoco.viewer
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
        self.stick_socket = create_subscriber_socket(VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC)

    # Function to publish the left/right hand keypoints and button Feedback
    def stream(self):
        print("oculus stick stream")
        model = mujoco.MjModel.from_xml_path("scene.xml")
        data = mujoco.MjData(model)

        X_ee_init = SE3.from_mocap_name(model, data, "pinch_site_target")
        H = SE3.from_rotation(SO3.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])))

        start_teleop = False

        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            rate = RateLimiter(frequency=200.0, warn=False)
            fps_counter = FPSCounter()
            while viewer.is_running():
                try:
                    _, message = self.stick_socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.ZMQError:
                    continue
                controller_state = parse_controller_state(message.decode())

                if controller_state.right_a:
                    # print("start teleop")
                    X_Cinit = controller_state.right_SE3
                    start_teleop = True

                if controller_state.right_b:
                    # print("pause teleop")
                    X_ee_init = SE3.from_mocap_name(model, data, "pinch_site_target")
                    start_teleop = False

                if start_teleop:
                    X_Ctarget = controller_state.right_SE3
                    X_Cdelta = X_Cinit.inverse().multiply(X_Ctarget)
                    X_Rdelta = H.inverse() @ X_Cdelta @ H

                    # translation
                    p_REt = X_ee_init.translation() + X_Rdelta.translation()

                    # rotation
                    R_REt = X_ee_init.rotation() @ X_Rdelta.rotation()
                    data.mocap_pos[model.body("pinch_site_target").mocapid[0]] = p_REt
                    data.mocap_quat[model.body("pinch_site_target").mocapid[0]] = R_REt.wxyz

                mujoco.mj_forward(model, data)
                fps_counter.getAndPrintFPS()
                viewer.sync()
                rate.sleep()


def main():
    oculus_reader = OculusReader()
    oculus_reader.stream()


if __name__ == "__main__":
    main()
