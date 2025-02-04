import zmq
import numpy as np
import rerun as rr
import matplotlib.pyplot as plt
import time
import math
from collections import deque

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
    y_axis = np.array([[0, 0], [0, frame_length]])

    rr.log("world/x_axis", rr.LineStrips2D([x_axis], colors=(255, 0, 0)))
    rr.log("world/y_axis", rr.LineStrips2D([y_axis], colors=[0, 0, 255]))

    while True:
        with timer:
            buffer = state_subscriber.recv().lstrip(topic)
            if len(buffer) % np.dtype(float).itemsize != 0: continue
            state = np.frombuffer(buffer, dtype=float)

            x, y, theta = state
            # print(f"x: {state[0]} | y: {state[1]} | θ: {(state[2] + np.pi) % (2 * np.pi) - np.pi}")
            corners = np.array([[LENGTH, WIDTH], [LENGTH, -WIDTH], [-LENGTH, -WIDTH], [-LENGTH, WIDTH], [LENGTH, WIDTH]])
            R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            corners = (R @ corners.T).T + np.array([x, y])

            rr.set_time_seconds("real_time", time.time())
            rr.log('robot/base', rr.LineStrips2D([corners]))

            arrow_length = 0.7 * LENGTH * 2
            arrow_tip = np.array([x, y]) + arrow_length * np.array([np.cos(theta), np.sin(theta)])
            arrow_line = np.array([[x, y], arrow_tip])
            rr.log("robot/arrow", rr.LineStrips2D([arrow_line]))


def print_state():
    topic = bytes("state ", 'utf-8')
    state_subscriber = zmq.Context().socket(zmq.SUB)
    state_subscriber.connect(f"tcp://{ROBOT_IP}:{STATE_PORT}")
    state_subscriber.setsockopt(zmq.CONFLATE, 1)
    state_subscriber.setsockopt(zmq.SUBSCRIBE, topic)
    timer = FrequencyTimer(frequency=10)

    try:
        while True:
            with timer:
                buffer = state_subscriber.recv().lstrip(topic)
                if len(buffer) % np.dtype(float).itemsize != 0: continue
                state = np.frombuffer(buffer, dtype=float)

                print(f"x: {state[0]} | y: {state[1]} | θ: {(state[2] + np.pi) % (2 * np.pi) - np.pi}")
    except KeyboardInterrupt:
        pass

def record_state():
    topic = bytes("state ", 'utf-8')
    state_subscriber = zmq.Context().socket(zmq.SUB)
    state_subscriber.connect(f"tcp://{ROBOT_IP}:{STATE_PORT}")
    state_subscriber.setsockopt(zmq.CONFLATE, 1)
    state_subscriber.setsockopt(zmq.SUBSCRIBE, topic)
    timer = FrequencyTimer(frequency=10)

    states = []
    start_time = time.time()

    try:
        while True:
            with timer:
                buffer = state_subscriber.recv().lstrip(topic)
                if len(buffer) % np.dtype(float).itemsize != 0: continue
                state = np.frombuffer(buffer, dtype=float)

                states.append(state)
    except KeyboardInterrupt:
        print(f"Recording stopped. Duration: {time.time() - start_time} seconds")
        print(f"Number of states recorded: {len(states)}")
        np.save("states.npy", np.array(states))

class Visualizer:
    def __init__(self):
        # Odometry figure
        plt.figure('Odometry')
        plt.axis([-1, 1, -1, 1])
        self.fig_axis = plt.gca()
        self.fig_axis.set_aspect('equal')
        plt.grid(which='both')
        self.robot_line, = plt.plot([], color='tab:gray')
        self.robot_arrow = plt.arrow(0, 0, 0, 0, head_width=0)
        self.odom_line, = plt.plot([], '--', color='tab:blue')

        # Velocity figure
        # plt.figure('Velocity')
        # self.vel_fig_axis = plt.gca()
        # self.vel_x_line, = plt.plot([], label='x')
        # self.vel_y_line, = plt.plot([], label='y')
        # self.vel_th_line, = plt.plot([], label='θ')
        # plt.grid(True)
        # plt.legend()

        # Bring to foreground
        plt.figure('Odometry')

    def draw(self, x, x_data):
        # Robot outline
        th = x[2]
        angles = th + np.radians([135, 45, -45, -135], dtype=np.float32)
        corners = (math.sqrt(2) / 2) * np.stack((np.cos(angles, dtype=np.float32), np.sin(angles, dtype=np.float32)), axis=1)
        corners = np.array([x[0], x[1]], dtype=np.float32) + WIDTH * corners
        corners = np.append(corners, corners[:1], axis=0)
        self.robot_line.set_data(*corners.T)

        # Robot heading
        arrow_dx = 0.25 * LENGTH * math.cos(th)
        arrow_dy = 0.25 * WIDTH * math.sin(th)
        self.robot_arrow.set_data(x=x[0], y=x[1], dx=arrow_dx, dy=arrow_dy, head_width=WIDTH / 8)

        # Trajectory (odometry)
        x_data = np.array(x_data, dtype=np.float32)
        self.odom_line.set_data(x_data[:, 0], x_data[:, 1])

        # Robot velocity
        # t_data = np.array(t_data, dtype=np.float64)  # Do not cast timestamps to np.float32, major loss of precision
        # dx_data = np.array(dx_data, dtype=np.float32)
        # self.vel_x_line.set_data(t_data, dx_data[:, 0])
        # self.vel_y_line.set_data(t_data, dx_data[:, 1])
        # self.vel_th_line.set_data(t_data, dx_data[:, 2])

        # Update plots
        self.fig_axis.relim()
        self.fig_axis.autoscale()
        # self.vel_fig_axis.relim()
        # self.vel_fig_axis.autoscale()
        plt.pause(0.05)

if __name__ == "__main__":
    # visualize_state()
    # record_state(); exit()

    # print_state()

    viz = Visualizer()
    states = np.load("states.npy")
    x_replay = deque(maxlen=500)

    i = 0
    for x in states:
        print(x)
        x_replay.append(x)
        viz.draw(x, x_replay)

