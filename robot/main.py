import zmq
import threading

from robot.base.base_controller import Base
from robot.communications import CommandType, receive_command, COMMAND_PORT, STATE_PORT
from robot.timer import FrequencyTimer


class RobotMain:
  def __init__(self):
    self.base = Base()
    context = zmq.Context()
    self.listener = context.socket(zmq.REP)
    self.listener.bind(f"tcp://*:{COMMAND_PORT}")

    self.state_publisher: zmq.Socket = context.socket(zmq.PUB)
    self.state_publisher.bind(f"tcp://*:{STATE_PORT}")
    self.state_publisher_timer = FrequencyTimer(frequency=100)
    self.state_publisher_thread = threading.Thread(target=self.state_publisher)

    self.running = False

  def run(self):
    self.running = True
    self.state_publisher_thread.start()

    command_handlers = {
      CommandType.SET_TARGET_VELOCITY: self.base.set_target_velocity,
      CommandType.SET_TARGET_POSITION: self.base.set_target_position,
    }
    while True:
      command_type, data = receive_command(self.listener)
      try:
        result: bytes = command_handlers[command_type](data)
        self.listener.send(result)
      except Exception as e:
        print(f"Server Error: {e}")
        self.listener.send(b"Error: " + str(e).encode())
        self.handle_shutdown()
        break

  def state_publisher(self):
    topic = bytes("state ", 'utf-8')
    while self.running:
      with self.state_publisher_timer:
        self.state_publisher.send(topic + self.base.x.tobytes())

  def handle_shutdown(self):
    self.running = False
    self.state_publisher_thread.join()
    if self.base.control_loop_running: self.base.stop_control()
    self.listener.close()


if __name__ == "__main__":
  r = RobotMain()
  r.run()
