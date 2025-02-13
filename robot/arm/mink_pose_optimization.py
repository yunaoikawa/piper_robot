import argparse
from functools import partial
from pathlib import Path

import mink
import nevergrad as ng
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import mujoco

_HERE = Path(__file__).parent
_XML = _HERE / "mujoco_visual" / "shoulder_mounted_pipers.xml"
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


def transform_gripper_values(gripper_values: np.ndarray) -> np.ndarray:
    HIGH = -0.04
    LOW = 0
    return gripper_values * (HIGH - LOW) + LOW


# This transform should take the data from the dataset and transform it to the
# coordinate frame of the robot end-effector.
DATA_TRANSFORM = np.array([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1],
])


def get_random_target_trajectory(task, seed=424242, index=0):
    np.random.seed(seed)
    target_task = TASKS[task]
    parquet_path_glob = (DATASET_PATH / target_task).glob("**/*.parquet")
    parquet_path_list = list(parquet_path_glob)
    np.random.shuffle(parquet_path_list)
    assert 0 <= index < len(parquet_path_list), f"Not enough data found for {task}"
    parquet_path = parquet_path_list[index]
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
parser.add_argument(
    "--video-out",
    type=str,
    default=None,
    help="If provided, run offscreen and save video to this file path.",
)
parser.add_argument(
    "--visualize-every",
    type=int,
    default=10,
    help="How often to visualize the target pose.",
)
parser.add_argument(
    "--left",
    action="store_true",
    help="Use left arm instead of right arm.",
)


def main(
    model,
    data,
    random_traj,
    fps: int = 30,
    use_left: bool = False,
):
    # =================== #
    # Setup IK.
    # =================== #
    configuration = mink.Configuration(model)
    collision_pairs = [
        (
            mink.get_subtree_geom_ids(model, model.body("right_link5").id),
            mink.get_subtree_geom_ids(model, model.body("left_link5").id),
        ),
    ]
    tasks = [
        right_end_effector_task := mink.FrameTask(
            frame_name="right_attachment_site",
            frame_type="site",
            position_cost=100.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        left_end_effector_task := mink.FrameTask(
            frame_name="left_attachment_site",
            frame_type="site",
            position_cost=100.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=0.1,
            collision_detection_distance=0.2,
        ),
    ]

    max_velocities = {
        "right_joint1": np.pi,
        "right_joint2": np.pi,
        "right_joint3": np.pi,
        "right_joint4": np.pi,
        "right_joint5": np.pi,
        "right_joint6": np.pi,
        "left_joint1": np.pi,
        "left_joint2": np.pi,
        "left_joint3": np.pi,
        "left_joint4": np.pi,
        "left_joint5": np.pi,
        "left_joint6": np.pi,
    }

    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    solver = "quadprog"
    pos_threshold = 1e-3
    ori_threshold = 1e-3
    max_iters = 10

    # For playing back data at real-time, we typically step at ~200 Hz in simulation
    # and fetch a new target from the trajectory at 'fps'.
    # rate = RateLimiter(frequency=200.0, warn=False)
    dt = 1.0 / 200.0
    task_trajectory, gripper_values = random_traj
    current_target_idx = 0
    current_time = 0
    # mujoco.mj_resetDataKeyframe(model, data, model.key("tasks").id)
    mujoco.mj_step(model, data)

    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)

    # Initialize the mocap target at the end-effector sites.
    mink.move_mocap_to_frame(
        model, data, "right_target", "right_attachment_site", "site"
    )
    mink.move_mocap_to_frame(model, data, "left_target", "left_attachment_site", "site")
    # Make the IK solver arm-agnostic. We should be able to do the tasks with either arm
    target_name = "left_target" if use_left else "right_target"
    other_target_name = "right_target" if use_left else "left_target"
    end_effector_task = left_end_effector_task if use_left else right_end_effector_task
    other_eef_task = right_end_effector_task if use_left else left_end_effector_task
    gripper_idx = 6 + 7 if use_left else 6
    # Now set the static target for the other arm.
    other_arm_init_T_wt_with_rot = mink.SE3.from_mocap_name(
        model, data, other_target_name
    )
    other_arm_init_T_wt = mink.SE3.from_rotation_and_translation(
        translation=other_arm_init_T_wt_with_rot.translation(),
        rotation=mink.SO3.from_matrix(WORLD_TO_SITE),
    )
    other_eef_task.set_target(other_arm_init_T_wt)
    other_eef_task.set_orientation_cost(0.0)
    other_eef_task.set_position_cost(0.0)  # Let it move freely

    # We'll reuse this in both viewer or offscreen mode:
    def run_simulation_step(current_time, current_target_idx):
        cost = 0.0
        current_time += dt
        if (current_time * fps) >= current_target_idx:
            current_target_idx += 1
            if current_target_idx >= len(task_trajectory):
                return (
                    False,
                    current_time,
                    current_target_idx,
                    cost,
                )  # signals we are done

        # Update task target.
        target_SE3 = task_trajectory[current_target_idx]
        target_gripper = gripper_values[current_target_idx]
        # Build transform from stored SE3
        target_T_wt = init_T_wt.multiply(mink.SE3.from_matrix(target_SE3))
        # Send the new target to the IK solver
        end_effector_task.set_target(target_T_wt)

        # Solve IK in a few iterations
        for _ in range(max_iters):
            vel = mink.solve_ik(configuration, tasks, dt, solver, 1e-3, limits=limits)
            configuration.integrate_inplace(vel, dt)
            err = end_effector_task.compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            cost = np.linalg.norm(err)
            if pos_achieved and ori_achieved:
                break

        data.ctrl[:6] = configuration.q[:6]
        data.ctrl[7:13] = configuration.q[8:14]
        data.ctrl[gripper_idx] = target_gripper
        mujoco.mj_step(model, data)

        return True, current_time, current_target_idx, cost

    # Get initial target pose for reference:
    init_T_wt_with_rot = mink.SE3.from_mocap_name(model, data, target_name)
    init_T_wt = mink.SE3.from_rotation_and_translation(
        translation=init_T_wt_with_rot.translation(),
        rotation=mink.SO3.from_matrix(WORLD_TO_SITE),
    )
    total_cost = 0.0

    while True:
        is_ok, current_time, current_target_idx, cost = run_simulation_step(
            current_time, current_target_idx
        )
        total_cost += cost
        if not is_ok:
            break
    return total_cost


def rollout_with_random_init(initial_position, model, data, random_traj, fps, use_left):
    # Set initial position
    for _ in range(100):
        data.ctrl = initial_position
        mujoco.mj_step(model, data)
    return main(model, data, random_traj, fps, use_left)


def rollout_with_random_tasks_and_fixed_init(
    initial_position,
    seed,
    n_rollouts_per_task,
    fps,
    use_left,
):
    total_cost = 0.0
    for task in TASKS.keys():
        for i in range(n_rollouts_per_task):
            random_traj = get_random_target_trajectory(task=task, seed=seed, index=i)
            model = mujoco.MjModel.from_xml_path(_XML.as_posix())
            data = mujoco.MjData(model)
            # Get control ranges of all actuators
            cost = rollout_with_random_init(
                initial_position=initial_position,
                model=model,
                data=data,
                random_traj=random_traj,
                fps=fps,
                use_left=use_left,
            )
            total_cost += cost
            print(f"Task: {task}, Rollout: {i}, Cost: {cost}")
    return total_cost


if __name__ == "__main__":
    args = parser.parse_args()
    random_traj = get_random_target_trajectory(task=args.task, seed=args.seed)
    fps = args.fps
    use_left = args.left

    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Get control ranges of all actuators
    control_ranges = model.actuator_ctrlrange
    ctrl_upper = control_ranges[:, 1].copy()
    ctrl_lower = control_ranges[:, 0].copy()

    default_init_position = np.array([
        0,
        0.502701,
        -0.414997,
        0,
        0,
        0,
        0,
        0,
        0.502701,
        -0.414997,
        0,
        0,
        0,
        0,
    ])
    # Nevergrad parametrization
    params = ng.p.Array(init=default_init_position, lower=ctrl_lower, upper=ctrl_upper)
    optimizer = ng.optimizers.NGOpt(parametrization=params, budget=2)
    to_optimize = partial(
        rollout_with_random_tasks_and_fixed_init,
        seed=args.seed,
        n_rollouts_per_task=1,
        fps=fps,
        use_left=use_left,
    )
    recommendation = optimizer.minimize(to_optimize)
    print(recommendation)
