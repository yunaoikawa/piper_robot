from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
from typing import Protocol, TypeVar

import msgpack
import msgpack_numpy as m
m.patch()

T = TypeVar('T')

class Serializable(Protocol):
    def serialize(self) -> bytes:
        ...

    @classmethod
    def deserialize(cls: T, data: bytes) -> T:
        ...

@dataclass
class Image:
    timestamp: int
    image: np.ndarray


@dataclass
class EncodedImage:
    timestamp: int
    image: np.ndarray
    encoding: str

    def serialize(self) -> bytes:
        return msgpack.packb(self.__dict__)

    @classmethod
    def deserialize(cls, data: bytes) -> "EncodedImage":
        return cls(**msgpack.unpackb(data))


@dataclass
class EncodedDepth:
    timestamp: int
    depth: np.ndarray
    confidence: np.ndarray
    focal: List[int]
    resolution: List[int]

    def serialize(self) -> bytes:
        return msgpack.packb(self.__dict__)

    @classmethod
    def deserialize(cls, data: bytes) -> "EncodedDepth":
        return cls(**msgpack.unpackb(data))


@dataclass
class Pose:
    timestamp: int
    pose: np.ndarray

    def serialize(self) -> bytes:
        return msgpack.packb(self.__dict__)

    @classmethod
    def deserialize(cls, data: bytes) -> "Pose":
        return cls(**msgpack.unpackb(data))


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
    target: np.ndarray

    def serialize(self) -> bytes:
        data = {
            "timestamp": self.timestamp,
            "type": self.type.value,
            "target": self.target
        }
        return msgpack.packb(data)

    @classmethod
    def deserialize(cls, data: bytes) -> "Command":
        data_unpacked = msgpack.unpackb(data)
        return cls(
            timestamp=data_unpacked["timestamp"],
            type=CommandType(data_unpacked["type"]),
            target=data_unpacked["target"]
        )


@dataclass
class RobotState:
    timestamp: float
    base_pose: np.ndarray
    base_velocity: np.ndarray

    def serialize(self) -> bytes:
        return msgpack.packb(self.__dict__)

    @classmethod
    def deserialize(cls, data: bytes) -> "RobotState":
        return cls(**msgpack.unpackb(data))
