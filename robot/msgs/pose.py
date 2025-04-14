from dataclasses import dataclass
import numpy.typing as npt
import pyarrow as pa

@dataclass
class Pose:
    timestamp: int
    wxyz_xyz: npt.NDArray

    def encode(self) -> tuple[pa.Array, dict]:
        return pa.array(self.wxyz_xyz.ravel()), {"timestamp": self.timestamp}

    @classmethod
    def decode(cls, data: pa.Array, metadata: dict) -> "Pose":
        return cls(metadata["timestamp"], data.to_numpy().reshape(7,))

