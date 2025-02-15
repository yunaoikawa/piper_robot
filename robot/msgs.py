"""
All the messages that are exchanged.
All timestamps are in nanoseconds.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np
import numpy.typing as npt
from typing import Protocol, TypeVar

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

T = TypeVar('T')

Buffer = npt.NDArray
U8Buffer = npt.NDArray[np.uint8]
F32Buffer = npt.NDArray[np.float32]
F64Buffer = npt.NDArray[np.float64]

class Message(Protocol):
    timestamp: int

    def serialize(self) -> bytes:
        ...

    @classmethod
    def deserialize(cls: T, data: bytes) -> T:
        ...

@dataclass
class Image:
    timestamp: int
    image: U8Buffer


@dataclass
class EncodedImage:
    timestamp: int
    image: U8Buffer
    encoding: str

    def serialize(self) -> bytes:
        return msgpack.packb(self.__dict__)

    @classmethod
    def deserialize(cls, data: bytes) -> "EncodedImage":
        return cls(**msgpack.unpackb(data))


@dataclass
class EncodedDepth:
    timestamp: int
    depth: U8Buffer
    confidence: U8Buffer
    focal: List[int]
    resolution: List[int]
    width: int
    height: int

    def serialize(self) -> bytes:
        return msgpack.packb(self.__dict__)

    @classmethod
    def deserialize(cls, data: bytes) -> "EncodedDepth":
        return cls(**msgpack.unpackb(data))


@dataclass
class Pose:
    timestamp: int
    pose: F64Buffer

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
    timestamp: int
    type: CommandType
    target: F64Buffer

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
    timestamp: int
    base_pose: F64Buffer
    base_velocity: F64Buffer

    def serialize(self) -> bytes:
        return msgpack.packb(self.__dict__)

    @classmethod
    def deserialize(cls, data: bytes) -> "RobotState":
        return cls(**msgpack.unpackb(data))
