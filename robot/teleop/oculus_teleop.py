import zmq
import numpy as np
# import time
import atexit
import threading
from loop_rate_limiters import RateLimiter
# from typing import Any

import mink

# from dora import Node

from robot.teleop.oculus_msgs import parse_controller_state

# from robot.network import VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC
# from robot.msgs.pose import Pose, ArmCommand
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

        # communication
        # self.node = Node()
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
        poller = zmq.Poller()

        # Connect to the VR controller
        stick_socket = context.socket(zmq.SUB)
        stick_socket.connect("tcp://{}:{}".format(VR_TCP_HOST, VR_TCP_PORT))
        stick_socket.subscribe(VR_CONTROLLER_TOPIC)

        # initialize the polling set
        poller.register(stick_socket, zmq.POLLIN)

        while not self.stop_event.is_set():
            events = dict(poller.poll(1000))
            if stick_socket in events:
                _, message = stick_socket.recv_multipart()
                with self.controller_state_lock_:
                    self.controller_state = parse_controller_state(message.decode())

        # close the context
        poller.unregister(stick_socket)
        stick_socket.close()
        context.destroy()

    # def arm_ee_pose_handler(self, event: dict[str, Any]):
    #     ee_pose = Pose.decode(event["value"], event["metadata"])
    #     self.ee_pose = ee_pose

    # def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
    #     current_time = time.perf_counter_ns()
    #     delay = (current_time - timestamp) / 1e9
    #     if delay > max_delay or delay < 0:
    #         print(f"Skipping message because of delay: {delay}s")
    #         return False
    #     return True

    def control_loop(self):
        rate = RateLimiter(20)
        while not self.stop_event.is_set():
            with self.controller_state_lock_:
                controller_state = self.controller_state
            if controller_state is None:
                print("WARN: no controller state yet")
                rate.sleep()
                # time.sleep(0.1)
                continue

            # if self.ee_pose is None:
            #     print("WARN: no ee pose yet")
            #     return

            # if not self.check_timestamp(self.ee_pose.timestamp, 0.05):
            #     print("WARN: ee pose timestamp is too old")
            #     return

            ee_pose = mink.SE3(wxyz_xyz=self.cone_e.get_right_ee_pose())

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
                    rate.sleep()  # time.sleep(0.1)
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
                    ee_target_wxyz_xyz=np.concatenate([R_REt.wxyz, p_REt]), gripper_target=gripper, preview_time=0.1
                )
                # arm_command = ArmCommand(time.perf_counter_ns(), np.concatenate([R_REt.wxyz, p_REt]), gripper)
                # self.node.send_output("arm_command", *arm_command.encode())

            rate.sleep()

    def stop(self):
        self.stop_event.set()
        self.oculus_thread.join()

    # def spin(self):
    #     for event in self.node:
    #         event_type = event["type"]
    #         if event_type == "INPUT":
    #             event_id = event["id"]

    #             if event_id == "ee_pose":
    #                 self.arm_ee_pose_handler(event)

    #             elif event_id == "tick":
    #                 self.step()

    #         elif event_type == "STOP":
    #             self.stop_event.set()
    #             self.oculus_thread.join()


def main():
    oculus_reader = OculusReader()
    atexit.register(oculus_reader.stop)
    oculus_reader.control_loop()


if __name__ == "__main__":
    main()
