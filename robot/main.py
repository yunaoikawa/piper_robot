import zmq

from robot.base.base_controller import Base
from robot.constants import COMMAND_PORT
from robot.communications import CommandType, receive_command


class RobotMain:
  def __init__(self):
    self.base = Base()
    self.listener = zmq.Context().socket(zmq.REP)
    self.listener.bind(f"tcp://*:{COMMAND_PORT}")

  def run(self):
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

  def handle_shutdown(self):
    if self.base.control_loop_running:
      self.base.stop_control()
    self.listener.close()


if __name__ == "__main__":
  r = RobotMain()
  r.run()
