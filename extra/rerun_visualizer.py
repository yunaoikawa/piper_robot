import zmq
import numpy as np
import rerun as rr

from robot.timer import FrequencyTimer
from robot.communications import ROBOT_IP, STATE_PORT

def visualize_state():
    topic = bytes("state", 'utf-8')
    state_subscriber = zmq.Context().socket(zmq.SUB)
    state_subscriber.connect(f"tcp://{ROBOT_IP}:{STATE_PORT}")
    state_subscriber.setsockopt(zmq.SUBSCRIBE, topic)
    timer = FrequencyTimer(frequency=100)

    while True:
        with timer:
            state = np.frombuffer(state_subscriber.recv().lstrip(topic))
            print(f"x: {state[0]} | y: {state[1]} | θ: {(state[2] + np.pi) % (2 * np.pi) - np.pi}")




