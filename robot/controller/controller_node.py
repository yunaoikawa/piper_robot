import zmq
from robot.msgs import Command
from robot.communications import Subscriber, FrequencyTimer, COMMAND_PORT
from robot.controller.base import Base


def main():
    ctx = zmq.Context()
    command_sub = Subscriber(ctx, COMMAND_PORT, ["/command"], [Command.deserialize])
    timer = FrequencyTimer("Controller", 250)

    base = Base()

    while True:
        with timer:
            _, command = command_sub.receive()
            if command is not None:
                base.set_target(command)
            base.step()
    # for event in node:
    #     if event["type"] == "INPUT":
    #         if event["id"] == "command":
    #             metadata = event["metadata"]
    #             target = event["value"].to_numpy()
    #             match metadata["command_type"]:
    #                 case CommandType.BASE_VELOCITY.value:
    #                     base.set_target_velocity(target)
    #                 case CommandType.BASE_POSITION.value:
    #                     base.set_target_position(target)
    #         if event["id"] == "tick":
    #             base.step()


if __name__ == "__main__":
    main()
