import zmq
import numpy as np
import atexit
import time
import threading
from loop_rate_limiters import RateLimiter

import mink

from robot.teleop.oculus_msgs import parse_controller_state

from robot.rpc import RPCClient


def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


# VR Constants
VR_TCP_HOST = "10.19.165.216"
# VR_TCP_HOST = "10.19.189.139"
VR_TCP_PORT = 5555
VR_CONTROLLER_TOPIC = b"oculus_controller"
GRIPPER_ANGLE_MAX = -22.0


class OculusReader:
    def __init__(self):
        # teleop state
        self.ee_pose = None
        self.start_teleop = False
        self.H = mink.SE3.from_rotation(mink.SO3.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])))
        self.X_Cinit = None
        self.X_ee_init = None

        self.cone_e = RPCClient("localhost", 8081)
        self.cone_e.init()
        self.cone_e.home_right_arm()

        # Oculus thread
        self.controller_state_lock_ = threading.Lock()
        self.controller_state = None
        self.stop_event = threading.Event()

        self.oculus_thread = threading.Thread(target=self.oculus_handler)
        self.oculus_thread.start()

    def oculus_handler(self):
        context = zmq.Context()
        # poller = zmq.Poller()

        # Connect to the VR controller
        stick_socket = context.socket(zmq.SUB)
        stick_socket.connect("tcp://{}:{}".format(VR_TCP_HOST, VR_TCP_PORT))
        stick_socket.subscribe(VR_CONTROLLER_TOPIC)

        # initialize the polling set
        # poller.register(stick_socket, zmq.POLLIN)
        last_command_timestamp = None
        interval_history = []

        while not self.stop_event.is_set():
            # events = dict(poller.poll(1000))
            # if stick_socket in events:
            _, message = stick_socket.recv_multipart()
            with self.controller_state_lock_:
                self.controller_state = parse_controller_state(message.decode())
                if last_command_timestamp is not None:
                    interval_now = time.time() - last_command_timestamp
                    interval_history.append(interval_now)
                last_command_timestamp = time.time()

        np.array(interval_history).tofile("interval_history.bin")
        stick_socket.close()
        context.destroy()

    def control_loop(self):
        rate = RateLimiter(20)
        while not self.stop_event.is_set():
            with self.controller_state_lock_:
                controller_state = self.controller_state
            if controller_state is None:
                print("WARN: no controller state yet")
                rate.sleep()
                continue

            ee_pose = self.cone_e.get_right_ee_pose()

            if controller_state.right_a:
                print("start teleop")
                self.X_Cinit = controller_state.right_SE3
                self.X_ee_init = ee_pose
                self.start_teleop = True

            if controller_state.right_b:
                self.start_teleop = False

            if self.start_teleop:
                if self.X_Cinit is None or self.X_ee_init is None:
                    print("WARN: no initial pose yet")
                    rate.sleep()
                    continue
                X_Ctarget = controller_state.right_SE3
                X_Cdelta = self.X_Cinit.inverse().multiply(X_Ctarget)
                X_Rdelta = self.H.inverse() @ X_Cdelta @ self.H
                # translation
                p_REt = self.X_ee_init.translation() + X_Rdelta.translation()
                # rotation
                R_REt = self.X_ee_init.rotation() @ X_Rdelta.rotation()

                # publish the target pose
                gripper = GRIPPER_ANGLE_MAX if controller_state.right_index_trigger < 0.5 else 0.0
                self.cone_e.set_right_ee_target(
                    target=mink.SE3(np.concatenate([R_REt.wxyz, p_REt])), gripper_target=gripper, preview_time=0.05
                )

            rate.sleep()

    def stop(self):
        self.stop_event.set()
        self.oculus_thread.join()


def main():
    oculus_reader = OculusReader()
    atexit.register(oculus_reader.stop)
    oculus_reader.control_loop()


if __name__ == "__main__":
    main()
