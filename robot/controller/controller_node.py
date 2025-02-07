import zmq
import threading

from robot.msgs import Command, CommandType, RobotState
from robot.controller.base import Base
from robot.communications import COMMAND_PORT, ROBOT_STATE_PORT, Subscriber, Publisher
from robot.timer import RateKeeper


class ControllerNode:
  def __init__(self):
    self.ctx = zmq.Context()
    self.base = Base()
    # self.left_arm = Arm()
    # self.right_arm = Arm()
    # self.lift = Lift()
    self.rk = RateKeeper(name="ControllerNode", rate=250, print_delay_threshold=0.001)

  def command_listener(self, event):
    command_sub = Subscriber(self.ctx, COMMAND_PORT, ["/robot/command"], Command.deserialize)
    while not event.is_set():
      _, command = command_sub.receive()
      match command.type:
        case CommandType.BASE_VELOCITY:
          self.base.set_target_velocity(command.payload)
        case CommandType.BASE_POSITION:
          self.base.set_target_position(command.payload)
    command_sub.stop()

  def run(self):
    e = threading.Event()
    t = threading.Thread(target=self.command_listener, args=(e,))

    state_pub = Publisher(self.ctx, ROBOT_STATE_PORT, RobotState.serialize)
    try:
      while True:
        self.base.step()
        state_pub.publish("/robot/state", RobotState(timestamp=self.rk.last_monitor_time, base_pose=self.base.x, base_velocity=self.base.dx))
        self.rk.keep_time()

    finally:
      e.set()
      t.join()


def launch_controller_node():
    node = ControllerNode()
    node.run()