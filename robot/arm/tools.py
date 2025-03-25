import math
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple
import zmq
import pickle
from scipy.spatial.transform import Rotation as R
import threading


# Pub/Sub classes for Keypoints
class ZMQKeypointPublisher(object):
    def __init__(self, host, port):
        self._host, self._port = host, port
        self._init_publisher()

    def _init_publisher(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://{}:{}".format(self._host, self._port))

    def pub_keypoints(self, keypoint_array, topic_name):
        """
        Process the keypoints into a byte stream and input them in this function
        """
        buffer = pickle.dumps(keypoint_array, protocol=-1)
        self.socket.send(bytes("{} ".format(topic_name), "utf-8") + buffer)

    def stop(self):
        print("Closing the publisher socket in {}:{}.".format(self._host, self._port))
        self.socket.close()
        self.context.term()


class ZMQKeypointSubscriber(threading.Thread):
    def __init__(self, host, port, topic):
        self._host, self._port, self._topic = host, port, topic
        self._init_subscriber()

        # Topic chars to remove
        self.strip_value = bytes("{} ".format(self._topic), "utf-8")

    def _init_subscriber(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect("tcp://{}:{}".format(self._host, self._port))
        self.socket.setsockopt(zmq.SUBSCRIBE, bytes(self._topic, "utf-8"))

    def recv_keypoints(self, flags=None):
        if flags is None:
            raw_data = self.socket.recv()
            raw_array = raw_data.lstrip(self.strip_value)
            return pickle.loads(raw_array)
        else:  # For possible usage of no blocking zmq subscriber
            try:
                raw_data = self.socket.recv(flags)
                raw_array = raw_data.lstrip(self.strip_value)
                return pickle.loads(raw_array)
            except zmq.Again:
                # print('zmq again error')
                return None

    def stop(self):
        print("Closing the subscriber socket in {}:{}.".format(self._host, self._port))
        self.socket.close()
        self.context.term()


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
        """Returns a 4x4 affine matrix from the controller's position and rotation.
        Args:
            controller_position: 3D position of the controller.
            controller_rotation: 4D quaternion of the controller's rotation.

            All in headset space.
        """

        return np.block([
            [
                R.as_matrix(R.from_quat(controller_rotation)),
                controller_position[:, np.newaxis],
            ],
            [np.zeros((1, 3)), 1.0],
        ])


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


def notify_component_start(component_name):
    print("***************************************************************")
    print("     Starting {} component".format(component_name))
    print("***************************************************************")


def create_subscriber_socket(host, port, topic, conflate=False):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, int(conflate))
    socket.connect("tcp://{}:{}".format(host, port))
    socket.subscribe(topic)
    flush_socket(socket)
    return socket


def flush_socket(socket):
    """Flush all messages currently in the socket."""
    while True:
        try:
            # Check if a message is waiting in the queue
            message = socket.recv(zmq.NOBLOCK)
            print(message)
        except zmq.Again:
            # No more messages to flush
            break


class FrequencyTimer(object):
    def __init__(self, frequency_rate):
        self.time_available = 1e9 / frequency_rate

    def start_loop(self):
        self.start_time = time.time_ns()

    def check_time(self, frequency_rate):
        # if prev_check_time variable doesn't exist, create it
        if not hasattr(self, "prev_check_time"):
            self.prev_check_time = self.start_time

        curr_time = time.time_ns()
        if (curr_time - self.prev_check_time) > 1e9 / frequency_rate:
            self.prev_check_time = curr_time
            return True
        return False

    def end_loop(self):
        current_time = time.time_ns()
        wait_time = self.start_time + self.time_available - current_time
        if wait_time > 0:
            time.sleep(wait_time / 1e9)
        elif wait_time < -self.time_available / 2:
            print(f"Warning: Overran the time available by {wait_time / 1e9} seconds")


def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    transformation_matrix[3, 0] = 0
    transformation_matrix[3, 1] = 0
    transformation_matrix[3, 2] = 0
    transformation_matrix[3, 3] = 1
    return transformation_matrix


def quaternion_from_matrix(matrix):
    """Convert a rotation matrix to quaternion (x,y,z,w order).

    Args:
        matrix: A 4x4 or 3x3 rotation matrix

    Returns:
        quaternion in x,y,z,w order to match the expected usage with Pinocchio
    """
    if matrix.shape == (4, 4):
        R = matrix[:3, :3]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise ValueError("Matrix must be 3x3 or 4x4")

    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    # Return in x,y,z,w order to match the expected usage
    return np.array([qx, qy, qz, qw])


def quaternion_from_euler(roll, pitch, yaw):
    """Convert euler angles to quaternion (x,y,z,w order).

    Args:
        roll: rotation around x in radians
        pitch: rotation around y in radians
        yaw: rotation around z in radians

    Returns:
        quaternion in x,y,z,w order to match the expected usage with Pinocchio
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Quaternion in x,y,z,w order
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    return np.array([qx, qy, qz, qw])
