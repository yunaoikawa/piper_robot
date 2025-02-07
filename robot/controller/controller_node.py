from dora import Node
import threading

from robot.msgs import CommandType
from robot.controller.base import Base

def main():
  node = Node("robot")
  stop_event = threading.Event()

  base = Base()
  base_control_loop = threading.Thread(target=base.control_loop, args=(stop_event,))
  base_control_loop.start()

  for event in node:
    if event["type"] == "INPUT":
      if event["id"] == "command":
        metadata = event["metadata"]
        target = event["value"].to_numpy()
        match metadata["command_type"]:
          case CommandType.BASE_VELOCITY.value:
            base.set_target_velocity(target)
          case CommandType.BASE_POSITION.value:
            base.set_target_position(target)
      if event["id"] == "tick":
        base.step()

    # if event["type"] == "STOP" or event["type"] == "ERROR":
    #   stop_event.set()
    #   base_control_loop.join()

if __name__ == "__main__":
  main()