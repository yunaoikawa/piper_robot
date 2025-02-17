from typing import cast
import zmq

from robot.network import Subscriber, COMMAND_PORT
from robot.network.msgs import Command
from robot.network.timer import FrequencyTimer
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
                base.set_target(cast(Command, command))
            base.step()


if __name__ == "__main__":
    main()
