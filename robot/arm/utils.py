import signal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import constants
import mujoco


class SignalBlocker:
    def __enter__(self):
        """Block SIGINT (CTRL+C) and SIGTERM, but store if any were received."""
        self.received_signal = None  # Store interrupted signal

        def handler(signum, frame):
            """Custom handler to store received signal instead of acting immediately."""
            print(
                f"\nSignal {signum} received, delaying execution until after the block."
            )
            self.received_signal = (signum, frame)

        # Save original handlers
        self.original_sigint = signal.signal(signal.SIGINT, handler)
        self.original_sigterm = signal.signal(signal.SIGTERM, handler)

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore original signal handlers and re-raise signal if one was captured."""
        signal.signal(signal.SIGINT, self.original_sigint)
        signal.signal(signal.SIGTERM, self.original_sigterm)

        if self.received_signal:
            signum, frame = self.received_signal
            print(f"\nReplaying signal {signum} now...")
            self.original_sigint(signum, frame)  # Re-execute the original handler


def scale_and_clip_control(q: np.ndarray) -> np.ndarray:
    min_val = np.array(constants.ARM_JOINT_LIMITS_MIN)
    max_val = np.array(constants.ARM_JOINT_LIMITS_MAX)
    N = len(constants.ARM_JOINT_LIMITS_MAX)

    clipped_q = np.zeros(N + 1)
    clipped_q[:N] = np.round(constants.ARM_JOINT_SCALING_CONSTANT * q[:N])
    clipped_q[:N] = np.clip(clipped_q[:N], a_min=min_val, a_max=max_val)

    clipped_q[N] = np.round(constants.WRIST_SCALING_CONSTANT * q[N])

    return clipped_q.astype(int)


DATASET_PATH = Path("~/projects/local-code/robot-utility-model-data/train").expanduser()

TASKS = {
    "door": "Door_Opening",
    "drawer": "Drawer_Opening",
    "tissue": "Tissue_Pick_Up",
    "bag": "Bag_Pick_Up",
    "bottle": "Reorientation",
}
WORLD_TO_SITE = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
])


def add_3color_axes_to_viewer(pos, rotation, scene, length=0.25, opacity=0.1):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 3  # increment ngeom
    # X-axis (Red)
    R_local_to_world = np.column_stack([rotation[:, 1], rotation[:, 2], rotation[:, 0]])
    mujoco.mjv_initGeom(
        geom=scene.geoms[scene.ngeom - 3],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.01, 0.01, length]),
        rgba=np.array([1, 0, 0, opacity]),
        pos=pos,
        mat=R_local_to_world.flatten(),  # Rotate to align with X
    )

    # Y-axis (Green)
    R_local_to_world_y = np.column_stack([
        -rotation[:, 0],
        -rotation[:, 2],
        rotation[:, 1],
    ])
    mujoco.mjv_initGeom(
        geom=scene.geoms[scene.ngeom - 2],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.01, 0.01, length]),
        rgba=np.array([0, 1, 0, opacity]),
        pos=pos,
        mat=R_local_to_world_y.flatten(),  # Rotate 90° to align with Y
    )

    # Z-axis (Blue)
    mujoco.mjv_initGeom(
        geom=scene.geoms[scene.ngeom - 1],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.01, 0.01, length]),
        rgba=np.array([0, 0, 1, opacity]),
        pos=pos,
        mat=rotation.flatten(),  # Identity matrix for alignment
    )


def transform_gripper_values(gripper_values: np.ndarray) -> np.ndarray:
    HIGH = -0.04
    LOW = 0
    return gripper_values * (HIGH - LOW) + LOW


# This transform should take the data from the dataset and transform it to the
# coordinate frame of the robot end-effector.
DOBBE_DATA_TRANSFORM = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1],
])


def get_random_target_trajectory(task, seed=424242):
    np.random.seed(seed)
    target_task = TASKS[task]
    parquet_path_glob = (DATASET_PATH / target_task).glob("**/*.parquet")
    parquet_path_list = list(parquet_path_glob)
    parquet_path = np.random.choice(parquet_path_list)
    trajectory_df = pd.read_parquet(parquet_path)
    gripper_values = trajectory_df["gripper"].values
    xyz = trajectory_df[["x", "y", "z"]].values
    quat = trajectory_df[["qx", "qy", "qz", "qw"]].values
    rotation_matrices = R.from_quat(quat, scalar_first=False).as_matrix()

    transformed_xyz = np.dot(xyz, DOBBE_DATA_TRANSFORM.T)
    transformed_rot = np.matmul(
        DOBBE_DATA_TRANSFORM[None],
        np.matmul(rotation_matrices, DOBBE_DATA_TRANSFORM.T[None]),
    )

    SE3_matrices = np.zeros((len(xyz), 4, 4))
    SE3_matrices[:, :3, :3] = transformed_rot
    SE3_matrices[:, :3, 3] = transformed_xyz
    SE3_matrices[:, 3, 3] = 1
    return SE3_matrices, transform_gripper_values(gripper_values)
