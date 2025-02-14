from dataclasses import dataclass
from enum import Enum
import numpy as np


@dataclass
class Image:
    timestamp: int
    image: np.ndarray


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


@dataclass
class RobotState:
    timestamp: float
    base_pose: np.ndarray
    base_velocity: np.ndarray
