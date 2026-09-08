"""Pinhole camera model (OpenCV convention: camera looks along +Z, y down).

The one object shared by calibration (`calibrate.py`) and pose fitting (`fit.py`): it
turns 3D world points into pixels and back-projects pixels onto a known world plane. All
metric reasoning about the photo goes through here so the frame/scale is consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def K_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """Intrinsics from a vertical FoV guess (fallback until a real calibration exists)."""
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


@dataclass
class Camera:
    K: np.ndarray              # 3x3 intrinsics
    R_wc: np.ndarray           # 3x3 world->camera rotation (x_cam = R_wc @ x_world + t_wc)
    t_wc: np.ndarray           # 3, world->camera translation
    width: int = 0
    height: int = 0

    @property
    def center(self) -> np.ndarray:
        """Camera center in world coordinates."""
        return -self.R_wc.T @ self.t_wc

    def project(self, pts_world: np.ndarray) -> np.ndarray:
        """(N,3) world -> (N,2) pixels."""
        P = np.atleast_2d(pts_world).astype(float)
        cam = (self.R_wc @ P.T + self.t_wc[:, None]).T      # (N,3) camera frame
        uv = (self.K @ cam.T).T
        return uv[:, :2] / uv[:, 2:3]

    def backproject_to_plane(self, uv, plane_z: float) -> np.ndarray:
        """Pixel (u,v) -> world point on the horizontal plane world_z == plane_z."""
        u, v = float(uv[0]), float(uv[1])
        d_cam = np.linalg.inv(self.K) @ np.array([u, v, 1.0])
        C = self.center
        d_world = self.R_wc.T @ d_cam
        if abs(d_world[2]) < 1e-9:
            raise ValueError("ray parallel to plane")
        t = (plane_z - C[2]) / d_world[2]
        return C + t * d_world
