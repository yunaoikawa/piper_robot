import asyncio
import zmq
import zmq.asyncio

from robot.msgs import Command, CommandType
from robot.controller.base import Base
from robot.communications import COMMAND_PORT, AsyncSubscriber
from robot.timer import RateKeeper
import threading

class ControllerNode:
    def __init__(self, rate: float = 250):
        self.ctx = zmq.asyncio.Context()
        self.base = Base()
        # self.left_arm = Arm()
        # self.right_arm = Arm()
        # self.lift = Lift()
        self.subscriber = AsyncSubscriber(self.ctx, COMMAND_PORT, ["/robot/command"], Command.deserialize)
        self.rate = rate
        self.command_queue = asyncio.Queue()

    async def listen_task(self):
        while True:
            command: Command = await self.subscriber.receive()
            await self.command_queue.put(command)

    async def control_loop(self):
        asyncio.create_task(self.listen_task())
        while True:
            start_time = asyncio.get_event_loop().time()
            if not self.command_queue.empty():
                command: Command = await self.command_queue.get()
                match command.type:
                    case CommandType.BASE_VELOCITY:
                        self.base.set_target_velocity(command.payload)
                    case CommandType.BASE_POSITION:
                        self.base.set_target_position(command.payload)
            self.command_queue.task_done()

            self.base.step()
            elapsed = asyncio.get_event_loop().time() - start_time
            await asyncio.sleep(max(0, (1 / self.rate) - elapsed))

if __name__ == "__main__":
    node = ControllerNode()
    asyncio.run(node.control_loop())
