import argparse
from dataclasses import dataclass
from pathlib import Path

import mink
import numpy as np
import pandas as pd
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

import mujoco
import mujoco.viewer

_HERE = Path(__file__).parent
_XML = _HERE / "mujoco_visual" / "scene_gripper.xml"
DATASET_PATH = Path("~/projects/local-code/robot-utility-model-data/train").expanduser()

TASKS = {
    "door": "Door_Opening",
    "drawer": "Drawer_Opening",
    "tissue": "Tissue_Pick_Up",
    "bag": "Bag_Pick_Up",
    "bottle": "Reorientation",
}


@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause


def add_3color_axes_to_viewer(pos, rotation, viewer, length=0.25):
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 3  # increment ngeom
    # X-axis (Red)
    # rotation = np.eye(3)
    R_local_to_world = np.column_stack([rotation[:, 1], rotation[:, 2], rotation[:, 0]])
    mujoco.mjv_initGeom(
        geom=scene.geoms[scene.ngeom - 3],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.01, 0.01, length]),
        rgba=np.array([1, 0, 0, 0.5]),
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
        rgba=np.array([0, 1, 0, 0.5]),
        pos=pos,
        mat=R_local_to_world_y.flatten(),  # Rotate 90° to align with Y
    )

    # Z-axis (Blue)
    mujoco.mjv_initGeom(
        geom=scene.geoms[scene.ngeom - 1],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.array([0.01, 0.01, length]),
        rgba=np.array([0, 0, 1, 0.5]),
        pos=pos,
        mat=rotation.flatten(),  # Identity matrix for alignment
    )


def transform_gripper_values(gripper_values: np.ndarray) -> np.ndarray:
    HIGH = -0.04
    LOW = 0
    return gripper_values * (HIGH - LOW) + LOW


# This transform should take the data from the dataset and transform it to the
# coordinate frame of the robot end-effector.
# The data in our datasets follows the OpenCV convention:
# x -> right, y -> down, z -> forward
# The robot end-effector has the following convention:
# x -> down, y -> left, z -> forward.
DATA_TRANSFORM = np.array([
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

    transformed_xyz = np.dot(xyz, DATA_TRANSFORM.T)
    transformed_rot = np.matmul(
        DATA_TRANSFORM[None], np.matmul(rotation_matrices, DATA_TRANSFORM.T[None])
    )

    SE3_matrices = np.zeros((len(xyz), 4, 4))
    SE3_matrices[:, :3, :3] = transformed_rot
    SE3_matrices[:, :3, 3] = transformed_xyz
    SE3_matrices[:, 3, 3] = 1
    return SE3_matrices, transform_gripper_values(gripper_values)


parser = argparse.ArgumentParser(
    prog="Mink IK test",
)
parser.add_argument(
    "--task",
    type=str,
    choices=TASKS.keys(),
    help="Task to perform open-loop.",
    required=True,
    default="door",
)
parser.add_argument(
    "--seed",
    type=int,
    help="Seed for random trajectory selection.",
    default=424242,
)
parser.add_argument(
    "--fps",
    type=float,
    help="Frames per second in the input data",
    default=30.0,
)

if __name__ == "__main__":
    args = parser.parse_args()
    random_traj = get_random_target_trajectory(task=args.task, seed=args.seed)
    fps = args.fps

    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=100.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]
    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    max_velocities = {
        "joint1": np.pi,
        "joint2": np.pi,
        "joint3": np.pi,
        "joint4": np.pi,
        "joint5": np.pi,
        "joint6": np.pi,
    }

    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    ## =================== ##

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-3
    ori_threshold = 1e-3
    max_iters = 10
    key_callback = KeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("tasks").id)
        mujoco.mj_step(model, data)

        configuration.update(data.qpos)
        # posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        task_trajectory, gripper_values = random_traj
        current_target_idx = 0
        current_time = 0
        init_T_wt = mink.SE3.from_mocap_name(model, data, "target")
        done = False
        target_T_wt = init_T_wt.copy()
        draw_every = 10

        while viewer.is_running():
            current_time += rate.dt
            if (current_time * fps) >= current_target_idx and not done:
                current_target_idx += 1
                if current_target_idx >= len(task_trajectory):
                    done = True
            # Update task target.
            if not done:
                target_SE3 = task_trajectory[current_target_idx]
                target_T_wt = init_T_wt.copy().multiply(
                    mink.SE3.from_matrix(target_SE3)
                )
                target_gripper = gripper_values[current_target_idx]
                if current_target_idx % draw_every == 0:
                    add_3color_axes_to_viewer(
                        target_T_wt.copy().translation(),
                        target_T_wt.copy().rotation().as_matrix(),
                        viewer,
                    )
            # Update task target.
            end_effector_task.set_target(target_T_wt)

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(
                    configuration, tasks, rate.dt, solver, 1e-3, limits=limits
                )
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.ctrl[:6] = configuration.q[:6]
            data.ctrl[6] = target_gripper
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
