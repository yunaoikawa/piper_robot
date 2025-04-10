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

    # @property
    # def right_position(self) -> np.ndarray:
    #     return self.right_affine[:3, 3]

    # @property
    # def left_position(self) -> np.ndarray:
    #     return self.left_affine[:3, 3]

    # @property
    # def right_rotation_matrix(self) -> np.ndarray:
    #     return self.right_affine[:3, :3]

    # @property
    # def left_rotation_matrix(self) -> np.ndarray:
    #     return self.left_affine[:3, :3]

    @property
    def left_SE3(self) -> SE3:
        # convert left-handed to right-handed
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        rotation_mat = M @ R.from_quat(self.left_local_rotation).as_matrix() @ M.T
        translation = self.left_local_position * np.array([1, 1, -1])
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(rotation_mat), translation=translation
        )

    @property
    def right_SE3(self) -> SE3:
        # convert left-handed to right-handed
        M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        rotation_mat = M @ R.from_quat(self.right_local_rotation).as_matrix() @ M.T
        translation = self.right_local_position * np.array([1, 1, -1])
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(rotation_mat), translation=translation
        )

    # def get_affine(self, controller_position: np.ndarray, controller_rotation: np.ndarray):
    #     """Returns a 4x4 affine matrix from the controller's position and rotation.
    #     Args:
    #         controller_position: 3D position of the controller.
    #         controller_rotation: 4D quaternion of the controller's rotation.

    #         All in headset space.
    #     """

    #     return np.block([
    #         [R.as_matrix(R.from_quat(controller_rotation)), controller_position[:, np.newaxis]],
    #         [np.zeros((1, 3)), 1.0],
    #     ])


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

        p_OCi_O = np.zeros(3)
        p_OCt_O = np.zeros(3)
        p_REi = np.zeros(3)
        p_REt = np.zeros(3)
        R_RO = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
           ])
        start_teleop = False

        with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            # rate = RateLimiter(frequency=200.0, warn=False)
            while viewer.is_running():
                _, message = self.stick_socket.recv_multipart()
                controller_state = parse_controller_state(message.decode())

                if controller_state.right_a:
                    print("start teleop")
                    p_OCi_O = controller_state.right_SE3.translation()
                    p_REi = SE3.from_mocap_name(model, data, "pinch_site_target").translation()
                    start_teleop = True

                if controller_state.right_b:
                    print("pause teleop")
                    start_teleop = False

                if start_teleop:
                    p_OCt_O = controller_state.right_SE3.translation()
                    print(f"{p_OCt_O=}")
                    p_CiCt_O = (p_OCt_O - p_OCi_O)
                    p_REt = R_RO @ p_CiCt_O + p_REi
                    data.mocap_pos[model.body("pinch_site_target").mocapid[0]] = p_REt

                mujoco.mj_forward(model, data)
                viewer.sync()
                # rate.sleep()


def main():
    oculus_reader = OculusReader()
    oculus_reader.stream()


if __name__ == "__main__":
    main()
