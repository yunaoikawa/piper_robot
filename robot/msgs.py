from dataclasses import dataclass
from typing import Dict, Tuple
from enum import Enum
import numpy as np
import json


@dataclass
class Image:
  timestamp: float
  image: np.ndarray

  @classmethod
  def serialize(cls, image: "Image") -> Tuple[str, bytes]:
    metadata = {
      "timestamp": image.timestamp,
      "h": image.image.shape[0],
      "w": image.image.shape[1],
      "c": image.image.shape[2],
      "dtype": str(image.image.dtype),
    }
    return json.dumps(metadata), image.image.tobytes()

  @classmethod
  def deserialize(cls, metadata: str, data: bytes) -> "Image":
    metadata = json.loads(metadata)
    return cls(metadata["timestamp"], np.frombuffer(data, dtype=metadata["dtype"]).reshape(metadata["h"], metadata["w"], metadata["c"]))


class CommandType(Enum):
  BASE_VELOCITY = 1
  BASE_POSITION = 2
  ARM_EEF_CARTESIAN = 3
  ARM_JOINT = 4
  ARM_VELOCITY = 5
  WB_EEF_POSITION = 6


@dataclass
class Command:
  timestamp: float
  type: CommandType
  payload: np.ndarray  # 1-d

  @classmethod
  def serialize(cls, command: "Command") -> Tuple[str, bytes]:
    metadata = {"timestamp": command.timestamp, "type": command.type}
    return json.dumps(metadata), command.payload.tobytes()

  @classmethod
  def deserialize(cls, metadata: str, data: bytes) -> "Command":
    metadata: Dict = json.loads(metadata)
    return cls(metadata["timestamp"], metadata["type"], np.frombuffer(data))


@dataclass
class RobotState:
  timestamp: float
  base_pose: np.ndarray
  base_velocity: np.ndarray

  @classmethod
  def serialize(cls, state: "RobotState") -> Tuple[str, bytes]:
    metadata = {
      "timestamp": state.timestamp,
      "base_pose_len": len(state.base_pose),
      "base_velocity_len": len(state.base_velocity),
    }
    return json.dumps(metadata), np.concatenate((state.base_pose, state.base_velocity)).tobytes()

  @classmethod
  def deserialize(cls, metadata: str, data: bytes) -> "RobotState":
    metadata: Dict = json.loads(metadata)
    base_pose = np.frombuffer(data[:metadata["base_pose_len"]], dtype=np.float64)
    base_velocity = np.frombuffer(data[metadata["base_pose_len"]:], dtype=np.float64)
    return cls(metadata["timestamp"], base_pose, base_velocity)

