"""
Piper single-arm lab environment for the flask-in-fridge RL task.

Observation (25-dim):
  qpos[0:6]   arm joint angles (joint1-5 + upper_jaw)
  qvel[0:6]   arm joint velocities
  flask_pos   (3,) world-frame position of flask body
  flask_quat  (4,) world-frame quaternion of flask body
  ee_pos      (3,) world-frame position of end-effector site
  flask_to_target (3,) vector from flask to fridge interior centre

Action (6-dim, continuous [-1, 1]):
  Delta control targets for [joint1, joint2, joint3, joint4, joint5, gripper],
  scaled by ACTION_SCALE before being added to current ctrl.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RL_DIR = Path(__file__).parent.parent          # rl/
_PIPER_MUJOCO_DIR = _RL_DIR.parent             # piper-mujoco/
XML_PATH = _PIPER_MUJOCO_DIR / "xml" / "lab-scene.xml"

# ---------------------------------------------------------------------------
# Physical constants derived from lab-scene.xml
#
# Fridge body: pos=(0.70, 0.0, 0.175), euler=(0, 0, π/2)
# Rotation R_z(π/2): local_x → world_y, local_y → world_(-x)
#
# Interior half-extents (local frame, accounting for wall thickness 0.004 m):
#   lx ∈ (-0.092, 0.092)   → world_y ∈ (-0.092, 0.092)
#   ly ∈ (-0.142, 0.142)   → world_x ∈ (0.558, 0.842)
#   lz ∈ (-0.167, 0.167)   → world_z ∈ (0.008, 0.342)
# ---------------------------------------------------------------------------
FRIDGE_BOUNDS = {
    "x": (0.558, 0.842),
    "y": (-0.092, 0.092),
    "z": (0.008, 0.342),
}
# Target: fridge shelf centre (local lz=0.02 → world_z=0.195)
FRIDGE_TARGET = np.array([0.70, 0.0, 0.195], dtype=np.float32)

# Actuator ctrl limits (from piper.xml joint ranges, inheritrange="1")
CTRL_LIMITS = np.array(
    [
        [-2.618, 2.618],   # joint1
        [0.0,    3.14],    # joint2
        [-2.697, 0.0],     # joint3
        [-1.832, 1.832],   # joint4
        [-1.22,  1.22],    # joint5
        [0.0,    0.99],    # gripper (upper_jaw_joint)
    ],
    dtype=np.float32,
)

# Per-joint delta scale applied to [-1, 1] actions
ACTION_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.02], dtype=np.float32)

# Flask random placement on table surface (world frame, z=0 is table top)
FLASK_X_RANGE = (0.10, 0.50)
FLASK_Y_RANGE = (-0.20, 0.30)


class PiperLabEnv:
    """
    Single-arm Piper lab environment (flask-in-fridge task).

    Compatible with the standard reset() / step() interface:
      obs, info = env.reset()
      obs, reward, terminated, truncated, info = env.step(action)
    """

    #: Number of observation dimensions
    obs_dim: int = 25
    #: Number of action dimensions
    act_dim: int = 6

    def __init__(
        self,
        max_episode_steps: int = 500,
        n_substeps: int = 5,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.n_substeps = n_substeps
        self._step_count: int = 0

        # Cache IDs for fast lookup
        self._flask_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "flask"
        )
        self._ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee"
        )

        assert self._flask_id >= 0, "Body 'flask' not found in model"
        assert self._ee_site_id >= 0, "Site 'ee' not found in model"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)

        # Reset arm to home keyframe (index 0 = lab_home)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # Randomise flask position on the table surface
        self.data.qpos[7] = np.random.uniform(*FLASK_X_RANGE)
        self.data.qpos[8] = np.random.uniform(*FLASK_Y_RANGE)
        self.data.qpos[9] = 0.0            # z: table surface
        self.data.qpos[10:14] = [1, 0, 0, 0]  # identity quaternion
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Δctrl: scale and clip to joint limits
        delta = np.clip(action, -1.0, 1.0).astype(np.float32) * ACTION_SCALE
        self.data.ctrl[:] = np.clip(
            self.data.ctrl[:] + delta,
            CTRL_LIMITS[:, 0],
            CTRL_LIMITS[:, 1],
        )

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, info = self._compute_reward()

        flask_z = self.data.xpos[self._flask_id, 2]
        terminated = info["flask_in_fridge"] or flask_z < -0.1
        truncated = self._step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        flask_pos = self.data.xpos[self._flask_id].copy()
        flask_quat = self.data.xquat[self._flask_id].copy()
        ee_pos = self.data.site_xpos[self._ee_site_id].copy()
        flask_to_target = FRIDGE_TARGET - flask_pos

        obs = np.concatenate(
            [
                self.data.qpos[0:6].copy(),   # arm joints (6)
                self.data.qvel[0:6].copy(),   # arm joint velocities (6)
                flask_pos,                     # (3,)
                flask_quat,                    # (4,)
                ee_pos,                        # (3,)
                flask_to_target,               # (3,)
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(self) -> tuple[float, dict]:
        flask_pos = self.data.xpos[self._flask_id]
        ee_pos = self.data.site_xpos[self._ee_site_id]

        flask_to_target_dist = float(np.linalg.norm(flask_pos - FRIDGE_TARGET))
        ee_to_flask_dist = float(np.linalg.norm(ee_pos - flask_pos))

        # Dense: minimise distance to target + approach flask
        reward = -flask_to_target_dist - 0.3 * ee_to_flask_dist

        # Check containment
        bx, by, bz = FRIDGE_BOUNDS["x"], FRIDGE_BOUNDS["y"], FRIDGE_BOUNDS["z"]
        flask_in_fridge = (
            bx[0] < flask_pos[0] < bx[1]
            and by[0] < flask_pos[1] < by[1]
            and bz[0] < flask_pos[2] < bz[1]
        )
        if flask_in_fridge:
            reward += 10.0

        # Penalty: flask fell off table
        if flask_pos[2] < -0.1:
            reward -= 5.0

        info = {
            "flask_in_fridge": flask_in_fridge,
            "flask_to_target": flask_to_target_dist,
            "ee_to_flask": ee_to_flask_dist,
        }
        return reward, info
