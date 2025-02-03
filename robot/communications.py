from enum import Enum
from typing import Optional
import zmq
import numpy as np


class CommandType(Enum):
  SET_TARGET_VELOCITY = 1
  SET_TARGET_POSITION = 2


def send_command(socket: zmq.Socket, command_type: CommandType, data: Optional[bytes] = None, flags: int = 0, copy: bool = True):
  if data is None:
    data = b""
  socket.send(bytes([command_type.value]) + data, flags, copy)


def receive_command(socket: zmq.Socket):
  buffer = socket.recv()
  ct = CommandType(buffer[0])
  match ct:
    case CommandType.SET_TARGET_VELOCITY | CommandType.SET_TARGET_POSITION:
      return ct, np.frombuffer(buffer[1:])
    case _:
      raise ValueError(f"Unknown command type: {ct}")
