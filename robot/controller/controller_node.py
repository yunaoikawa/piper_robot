from typing import cast
import zmq
import signal

from robot.network import Subscriber, COMMAND_PORT
from robot.network.msgs import Command
from robot.network.timer import FrequencyTimer
from robot.controller.base import Base


def main():
    ctx = zmq.Context()
    command_sub = Subscriber(ctx, COMMAND_PORT, ["/command"], [Command.deserialize], no_block=True)
    timer = FrequencyTimer("Controller", 250)
    running = True

    def signal_handler(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    base = Base()

    while running:
        with timer:
            _, command = command_sub.receive()
            if command is not None:
                base.set_target(cast(Command, command))
            base.step()

    command_sub.stop()


if __name__ == "__main__":
    main()
