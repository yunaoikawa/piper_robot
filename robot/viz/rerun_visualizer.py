import zmq
import numpy as np
import rerun as rr
import time

from robot.timer import FrequencyTimer
from robot.constants import LENGTH, WIDTH
from robot.communications import ROBOT_IP, STATE_PORT

def rr_visualize_state():
    rr.init("robot_visualization", spawn=True)

    topic = bytes("state ", 'utf-8')
    state_subscriber = zmq.Context().socket(zmq.SUB)
    state_subscriber.connect(f"tcp://{ROBOT_IP}:{STATE_PORT}")
    state_subscriber.setsockopt(zmq.CONFLATE, 1)
    state_subscriber.setsockopt(zmq.SUBSCRIBE, topic)
    timer = FrequencyTimer(frequency=10)

    # Create coordinate frame arrows
    frame_length = 1.0  # 1 meter arrows
    x_axis = np.array([[0, 0], [frame_length, 0]])
    y_axis = np.array([[0, 0], [0, -frame_length]])


    while True:
        with timer:
            buffer = state_subscriber.recv().lstrip(topic)
            if len(buffer) % np.dtype(float).itemsize != 0: continue
            state = np.frombuffer(buffer, dtype=float)

            rr.log("world/x_axis", rr.LineStrips2D([x_axis], colors=(255, 0, 0)))
            rr.log("world/y_axis", rr.LineStrips2D([y_axis], colors=[0, 0, 255]))

            x, y, theta = state
            y = -y  # Invert y-axis
            theta = -theta  # Invert theta
            corners = np.array([[LENGTH, WIDTH], [LENGTH, -WIDTH], [-LENGTH, -WIDTH], [-LENGTH, WIDTH], [LENGTH, WIDTH]])
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            corners = (R @ corners.T).T + np.array([x, y])

            rr.set_time_seconds("real_time", time.time())
            rr.log('robot/base', rr.LineStrips2D([corners]))

            arrow_length = 0.7 * LENGTH * 2
            arrow_tip = np.array([x, y]) + arrow_length * np.array([np.cos(theta), np.sin(theta)])
            arrow_line = np.array([[x, y], arrow_tip])
            rr.log("robot/arrow", rr.LineStrips2D([arrow_line]))


if __name__ == "__main__":
    rr_visualize_state()

