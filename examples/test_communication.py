from robot.communications import CommandType, send_command
from robot.constants import COMMAND_PORT
import numpy as np
import zmq
import time

if __name__ == "__main__":
    socket = zmq.Context().socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{COMMAND_PORT}")
    try:
        while True:
            send_command(socket, CommandType.SET_TARGET_VELOCITY, np.array([0.0, 0.2, 0.0], dtype=float).tobytes())
            response = socket.recv()
            time.sleep(0.1)
    finally:
        socket.close()
        exit(0)
