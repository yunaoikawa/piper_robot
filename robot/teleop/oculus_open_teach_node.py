import zmq
import numpy as np
import time
from typing import Tuple
from dataclasses import dataclass
from loop_rate_limiters import RateLimiter

from scipy.spatial.transform import Rotation as R

from robot.network import Publisher, ARM_COMMAND_PORT, BASE_PORT, VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC
from robot.network.msgs import ArmCommand, Command, CommandType


@dataclass
class ControllerState:
    created_timestamp: float
    left_x: bool
    left_y: bool
    left_menu: bool
    left_thumbstick: bool
    left_index_trigger: float
    left_hand_trigger: float
    left_thumbstick_axes: np.ndarray[Tuple[float, float]]
    left_local_position: np.ndarray[Tuple[float, float, float]]
    left_local_rotation: np.ndarray[Tuple[float, float, float, float]]

    right_a: bool
    right_b: bool
    right_menu: bool
    right_thumbstick: bool
    right_index_trigger: float
    right_hand_trigger: float
    right_thumbstick_axes: np.ndarray[Tuple[float, float]]
    right_local_position: np.ndarray[Tuple[float, float, float]]
    right_local_rotation: np.ndarray[Tuple[float, float, float, float]]

    @property
    def right_position(self) -> np.ndarray:
        return self.right_affine[:3, 3]

    @property
    def left_position(self) -> np.ndarray:
        return self.left_affine[:3, 3]

    @property
    def right_rotation_matrix(self) -> np.ndarray:
        return self.right_affine[:3, :3]

    @property
    def left_rotation_matrix(self) -> np.ndarray:
        return self.left_affine[:3, :3]

    @property
    def left_affine(self) -> np.ndarray:
        return self.get_affine(self.left_local_position, self.left_local_rotation)

    @property
    def right_affine(self) -> np.ndarray:
        return self.get_affine(self.right_local_position, self.right_local_rotation)

    def get_affine(self, controller_position: np.ndarray, controller_rotation: np.ndarray):
        """ Returns a 4x4 affine matrix from the controller's position and rotation.
        Args:
            controller_position: 3D position of the controller.
            controller_rotation: 4D quaternion of the controller's rotation.

            All in headset space.
        """

        return np.block([[R.as_matrix(R.from_quat(controller_rotation)), controller_position[:, np.newaxis]],
                         [np.zeros((1, 3)), 1.]])


def parse_controller_state(controller_state_string: str) -> ControllerState:

    left_data, right_data = controller_state_string.split('|')

    left_data = left_data.split(';')[1:-1]
    right_data = right_data.split(';')[1:-1]

    def parse_bool(val: str) -> bool:
        return val.split(':')[1].lower().strip() == "true"

    def parse_float(val: str) -> float:
        return float(val.split(':')[1])

    def parse_list_float(val: str) -> np.ndarray:
        return np.array(list(map(float, val.split(':')[1].split(','))))

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
            parse_list_float(data[8])
        )

    left_parsed = parse_section(left_data)
    right_parsed = parse_section(right_data)

    return ControllerState(time.time(), *left_parsed, *right_parsed)

def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))

def create_subscriber_socket(host, port, topic):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    # socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect('tcp://{}:{}'.format(host, port))
    socket.subscribe(topic)
    return socket

# This class is used to detect the hand keypoints from the VR and publish them.
class OculusReader:
    def __init__(self):
        # Create a subscriber socket
        self.stick_socket = create_subscriber_socket(VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC)

        # Create a publisher for the controller state
        # self.arm_pub = Publisher(ctx, ARM_COMMAND_PORT)
        # self.base_pub = Publisher(ctx, BASE_PORT)

    # Function to publish the left/right hand keypoints and button Feedback
    def stream(self):
        print("oculus stick stream")
        rate_limiter = RateLimiter(frequency=60, name="oculus")
        max_velocity = np.array([0.5, 0.5, 0.78])

        try:
            while True:
                message = self.stick_socket.recv_string()
                print(message)
                if message == "oculus_controller":
                    continue
                controller_state = parse_controller_state(message)
                # arm_msg = ArmCommand(
                #     timestamp=time.perf_counter_ns(),
                #     left_target=controller_state.left_affine,
                #     left_gripper_value=controller_state.left_index_trigger,
                #     left_start_teleop=controller_state.left_x,
                #     left_home=False,
                #     left_pause_teleop=controller_state.left_y,
                #     right_target=controller_state.right_affine,
                #     right_gripper_value=controller_state.right_index_trigger,
                #     right_start_teleop=controller_state.right_a,
                #     right_pause_teleop=controller_state.right_b,
                #     right_home=False,
                # )
                # print(arm_msg)
                rate_limiter.sleep()
                # self.arm_pub.publish("/arm_command", arm_msg)

                # vy, vx = controller_state.right_thumbstick_axes
                # w = controller_state.left_thumbstick_axes[0]

                # target_velocity = apply_deadzone(np.array([vx, -vy, -w])) * max_velocity

                # base_msg = Command(
                #     timestamp=time.perf_counter_ns(),
                #     type=CommandType.BASE_VELOCITY,
                #     target=target_velocity
                # )
                # self.base_pub.publish("/command", base_msg)

        except KeyboardInterrupt:
            pass

        print('Stopping the oculus reader...')

def main():
    oculus_reader = OculusReader()
    oculus_reader.stream()

if __name__ == "__main__":
    main()
