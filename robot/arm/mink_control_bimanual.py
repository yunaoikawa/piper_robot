import argparse
from pathlib import Path

import mink
import numpy as np
from loop_rate_limiters import RateLimiter

import mujoco
import mujoco.viewer
import piper_utils
import utils

_HERE = Path(__file__).parent
_XML = _HERE / "mujoco_visual" / "shoulder_mounted_pipers.xml"
WORLD_TO_SITE = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
])


parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--task", type=str, default=None, choices=utils.TASKS.keys())
parser.add_argument("--use_left", action="store_true")
parser.add_argument("--demo_fps", type=int, default=30)

if __name__ == "__main__":
    args = parser.parse_args()
    use_real = args.real
    fps = args.demo_fps
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

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
    rate = RateLimiter(frequency=200.0, warn=False)
    current_target_idx = 0
    current_time = 0
    mujoco.mj_resetDataKeyframe(model, data, model.key("optim").id)
    mujoco.mj_step(model, data)

    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)

    # Initialize the mocap target at the end-effector sites.
    mink.move_mocap_to_frame(
        model, data, "right_target", "right_attachment_site", "site"
    )
    mink.move_mocap_to_frame(model, data, "left_target", "left_attachment_site", "site")
    use_left = args.use_left
    # Make the IK solver arm-agnostic. We should be able to do the tasks with either arm
    target_name = "left_target" if use_left else "right_target"
    other_target_name = "right_target" if use_left else "left_target"
    end_effector_task = left_end_effector_task if use_left else right_end_effector_task
    other_eef_task = right_end_effector_task if use_left else left_end_effector_task
    left_gripper_idx = 13
    right_gripper_idx = 6

    # If task is given, use task target.
    if args.task:
        trajectory, gripper_trajectory = utils.get_random_target_trajectory(args.task)
    else:
        trajectory, gripper_trajectory = None, None

    # Now set the static target for the other arm.
    other_arm_init_T_wt_with_rot = mink.SE3.from_mocap_name(
        model, data, other_target_name
    )
    init_T_wt_with_rot = mink.SE3.from_mocap_name(model, data, target_name)

    other_arm_init_T_wt = mink.SE3.from_rotation_and_translation(
        translation=other_arm_init_T_wt_with_rot.translation(),
        rotation=mink.SO3.from_matrix(WORLD_TO_SITE),
    )
    other_eef_task.set_target(other_arm_init_T_wt)
    init_T_wt = mink.SE3.from_rotation_and_translation(
        translation=init_T_wt_with_rot.translation(),
        rotation=mink.SO3.from_matrix(WORLD_TO_SITE),
    )

    def run_simulation_step(target_SE3=None, other_target_SE3=None):
        if target_SE3 is None:
            target_SE3 = mink.SE3.from_mocap_name(model, data, target_name)

        if other_target_SE3 is None:
            other_target_SE3 = mink.SE3.from_mocap_name(model, data, other_target_name)

        # Update task target.
        end_effector_task.set_target(target_SE3)
        other_eef_task.set_target(other_target_SE3)

        # Solve IK in a few iterations
        for _ in range(max_iters):
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
        data.ctrl[7:13] = configuration.q[8:14]
        data.ctrl[left_gripper_idx] = data.ctrl[right_gripper_idx] = 0.0
        mujoco.mj_step(model, data)

        return True, configuration.q

    if use_real:
        # Start the piper controllers.
        piper_left, piper_right = piper_utils.setup_arms(
            left_name="can_left", right_name="can_right"
        )

    current_t = 0

    if not args.headless:
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)

            frame_count = 0

            while viewer.is_running():
                # Step the simulation
                current_t += rate.dt
                current_target_SE3 = None
                if trajectory is not None:
                    if len(trajectory) > int(current_t * fps):
                        current_target = trajectory[int(current_t * fps)]
                        current_target_SE3 = init_T_wt.multiply(
                            mink.SE3.from_matrix(current_target)
                        )
                    else:
                        break

                is_ok, q = run_simulation_step(current_target_SE3)
                if not is_ok:
                    break

                if use_real:
                    # with SignalBlocker():
                    piper_utils.send_to_arms(piper_left, piper_right, q)

                # Render
                viewer.sync()
                rate.sleep()
                frame_count += 1
