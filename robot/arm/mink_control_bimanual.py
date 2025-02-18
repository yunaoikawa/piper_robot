import argparse
import signal
import time
from pathlib import Path

import mink
import numpy as np
from loop_rate_limiters import RateLimiter
from piper_sdk import C_PiperInterface

import constants
import mujoco
import mujoco.viewer


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


_HERE = Path(__file__).parent
_XML = _HERE / "mujoco_visual" / "shoulder_mounted_pipers.xml"
WORLD_TO_SITE = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
])


def enable_fun(piper: C_PiperInterface):
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = (
            piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        )
        print("Enable status:", enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        print("--------------------")
        if elapsed_time > timeout:
            print("Timeout....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if elapsed_time_flag:
        print("The program automatically enables timeout, exit the program")
        exit(0)


def scale_and_clip_control(q: np.ndarray) -> np.ndarray:
    min_val = np.array(constants.ARM_JOINT_LIMITS_MIN)
    max_val = np.array(constants.ARM_JOINT_LIMITS_MAX)
    N = len(constants.ARM_JOINT_LIMITS_MAX)

    clipped_q = np.zeros(N + 1)
    clipped_q[:N] = np.round(constants.ARM_JOINT_SCALING_CONSTANT * q[:N])
    clipped_q[:N] = np.clip(clipped_q[:N], a_min=min_val, a_max=max_val)

    clipped_q[N] = np.round(constants.WRIST_SCALING_CONSTANT * q[N])

    return clipped_q.astype(int)


parser = argparse.ArgumentParser()
parser.add_argument("--real", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()
    use_real = args.real
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
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_step(model, data)

    configuration.update(data.qpos)
    mujoco.mj_forward(model, data)

    # Initialize the mocap target at the end-effector sites.
    mink.move_mocap_to_frame(
        model, data, "right_target", "right_attachment_site", "site"
    )
    mink.move_mocap_to_frame(model, data, "left_target", "left_attachment_site", "site")
    # TODO fix this
    use_left = False
    # Make the IK solver arm-agnostic. We should be able to do the tasks with either arm
    target_name = "left_target" if use_left else "right_target"
    other_target_name = "right_target" if use_left else "left_target"
    end_effector_task = left_end_effector_task if use_left else right_end_effector_task
    other_eef_task = right_end_effector_task if use_left else left_end_effector_task
    left_gripper_idx = 13
    right_gripper_idx = 6
    # Now set the static target for the other arm.
    other_arm_init_T_wt_with_rot = mink.SE3.from_mocap_name(
        model, data, other_target_name
    )
    other_arm_init_T_wt = mink.SE3.from_rotation_and_translation(
        translation=other_arm_init_T_wt_with_rot.translation(),
        rotation=mink.SO3.from_matrix(WORLD_TO_SITE),
    )
    other_eef_task.set_target(other_arm_init_T_wt)

    def run_simulation_step():
        # Update task target.
        target_SE3 = mink.SE3.from_mocap_name(model, data, target_name)
        end_effector_task.set_target(target_SE3)

        other_target_SE3 = mink.SE3.from_mocap_name(model, data, other_target_name)
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

    # Get initial target pose for reference:
    init_T_wt_with_rot = mink.SE3.from_mocap_name(model, data, target_name)
    init_T_wt = mink.SE3.from_rotation_and_translation(
        translation=init_T_wt_with_rot.translation(),
        rotation=mink.SO3.from_matrix(WORLD_TO_SITE),
    )

    if use_real:
        # Start the piper controllers.
        piper_left = C_PiperInterface("can_left")
        piper_left.ConnectPort()
        piper_left.EnableArm(7)
        enable_fun(piper=piper_left)
        piper_right = C_PiperInterface("can_right")
        piper_right.ConnectPort()
        piper_right.EnableArm(7)
        enable_fun(piper=piper_right)

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
            is_ok, q = run_simulation_step()
            if not is_ok:
                break

            if use_real:
                with SignalBlocker():
                    right_q, left_q = q[:6], q[8:14]
                    right_joint_values = scale_and_clip_control(right_q).tolist()
                    piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)
                    piper_right.JointCtrl(*right_joint_values[:6])
                    # piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                    piper_right.MotionCtrl_2(0x01, 0x01, 50, 0x00)

                    left_joint_values = scale_and_clip_control(left_q).tolist()
                    piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)
                    piper_left.JointCtrl(*left_joint_values[:6])
                    # piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                    piper_left.MotionCtrl_2(0x01, 0x01, 50, 0x00)

            # Render
            viewer.sync()
            rate.sleep()
            frame_count += 1
