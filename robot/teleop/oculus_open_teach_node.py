import zmq
import numpy as np
import time
import threading
from loop_rate_limiters import RateLimiter

import mink

from robot.teleop.oculus_msgs import parse_controller_state
from robot.network.node import Node
from robot.network import VR_TCP_HOST, VR_TCP_PORT, VR_CONTROLLER_TOPIC
from robot.msgs.pose import Pose


def apply_deadzone(arr, deadzone_size=0.05):
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))

class OculusReader(Node):
    def __init__(self):
        super().__init__()
        # Shared between threads
        self.controller_state_lock_ = threading.Lock()
        self.controller_state = None
        self.ee_pose_lock_ = threading.Lock()
        self.ee_pose = None

        self.subscribe("EE_POSE", self.arm_ee_pose_handler)
        self.create_thread(self.oculus_handler)
        self.create_thread(self.teleop_loop)

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

    def arm_ee_pose_handler(self, channel, data):
        ee_pose = Pose.decode(data)

        with self.ee_pose_lock_:
            self.ee_pose = ee_pose #             # print(f"ee_pose: {self.ee_pose}")

    def teleop_loop(self):
        print("teleop loop started")
        rate_limiter = RateLimiter(frequency=60, name="oculus")

        start_teleop = False
        H = mink.SE3.from_rotation(mink.SO3.from_matrix(np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])))

        while not self.stop_event.is_set():
            with self.controller_state_lock_:
                controller_state = self.controller_state
            if controller_state is None:
                rate_limiter.sleep()
                continue

            with self.ee_pose_lock_:
                ee_pose = self.ee_pose
            if ee_pose is None:
                print("WARN: no ee pose yet")
                rate_limiter.sleep()
                continue
            if not self.check_timestamp(ee_pose.timestamp, 0.05):
                print("WARN: ee pose timestamp is too old")
                rate_limiter.sleep()
                continue

            ee_pose = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(np.array(ee_pose.rotation)), translation=np.array(ee_pose.translation)
            )

            if controller_state.right_a:
                print("start teleop")
                X_Cinit = controller_state.right_SE3
                X_ee_init = ee_pose
                start_teleop = True

            if controller_state.right_b:
                # X_ee_init = ee_pose
                start_teleop = False

            if start_teleop:
                X_Ctarget = controller_state.right_SE3
                X_Cdelta = X_Cinit.inverse().multiply(X_Ctarget)
                X_Rdelta = H.inverse() @ X_Cdelta @ H
                # translation
                p_REt = X_ee_init.translation() + X_Rdelta.translation()
                # rotation
                R_REt = X_ee_init.rotation() @ X_Rdelta.rotation()

                # publish the target pose
                target_pose = Pose()
                target_pose.timestamp = time.perf_counter_ns()
                target_pose.translation = p_REt.tolist()
                target_pose.rotation = R_REt.wxyz.tolist()
                self.publish("ARM_COMMAND", target_pose)

            rate_limiter.sleep()


def main():
    oculus_reader = OculusReader()
    oculus_reader.spin()


if __name__ == "__main__":
    main()
