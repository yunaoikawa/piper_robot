from dora import Node

from robot.controller import CommandType
from robot.controller.base import Base


def main():
  node = Node("robot")

  base = Base()
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


if __name__ == "__main__":
  main()
