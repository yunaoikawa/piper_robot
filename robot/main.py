import zmq
import threading
import signal

from robot.base.base_controller import Base
from robot.communications import CommandType, receive_command, COMMAND_PORT, STATE_PORT
from robot.timer import FrequencyTimer


class RobotMain:
  def __init__(self):
    self.base = Base()
    self.listener = zmq.Context().socket(zmq.REP)
    self.listener.bind(f"tcp://*:{COMMAND_PORT}")

    self.state_publisher_thread = threading.Thread(target=self.state_publisher)
    self.running = False

  def run(self):
    self.running = True
    self.state_publisher_thread.start()

    command_handlers = {
      CommandType.SET_TARGET_VELOCITY: self.base.set_target_velocity,
      CommandType.SET_TARGET_POSITION: self.base.set_target_position,
    }
    while self.running:
      command_type, data = receive_command(self.listener)
      try:
        result: bytes = command_handlers[command_type](data)
        self.listener.send(result)
      except Exception as e:
        print(f"Server Error: {e}")
        self.handle_shutdown()
        break

  def state_publisher(self):
    topic = bytes("state ", 'utf-8')
    state_publisher: zmq.Socket = zmq.Context().socket(zmq.PUB)
    state_publisher.setsockopt(zmq.CONFLATE, 1)
    state_publisher.bind(f"tcp://*:{STATE_PORT}")
    state_publisher_timer = FrequencyTimer(frequency=10)
    while self.running:
      with state_publisher_timer:
        state_publisher.send(topic + self.base.x.tobytes())

  def handle_shutdown(self):
    self.running = False
    self.state_publisher_thread.join()
    if self.base.control_loop_running: self.base.stop_control()
    self.listener.close()

def sigint_handler(signum, frame):
    r.handle_shutdown()
    print("Robot server shutdown gracefully.")
    exit(0)

signal.signal(signal.SIGINT, sigint_handler)

if __name__ == "__main__":
  r = RobotMain()
  r.run()
