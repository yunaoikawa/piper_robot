import time
from pathlib import Path

import mink
import mujoco
import mujoco.viewer
import numpy as np
from constants import (
    ARM_JOINT_LIMITS_MAX,
    ARM_JOINT_LIMITS_MIN,
    ARM_JOINT_SCALING_CONSTANT,
    WRIST_SCALING_CONSTANT,
)
from loop_rate_limiters import RateLimiter
from piper_sdk import C_PiperInterface

_HERE = Path(__file__).parent
_XML = _HERE / "mujoco_visual" / "scene_gripper.xml"


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
        print("使能状态:", enable_flag)
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
    min_val = np.array(ARM_JOINT_LIMITS_MIN)
    max_val = np.array(ARM_JOINT_LIMITS_MAX)
    N = len(ARM_JOINT_LIMITS_MAX)

    clipped_q = np.zeros(N + 1)
    clipped_q[:N] = np.round(ARM_JOINT_SCALING_CONSTANT * q[:N])
    clipped_q[:N] = np.clip(clipped_q[:N], a_min=min_val, a_max=max_val)

    clipped_q[N] = np.round(WRIST_SCALING_CONSTANT * q[N])

    return clipped_q.astype(int)


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Enable collision avoidance between the following geoms:
    collision_pairs = [
        (["gripper_geom"], ["floor", "wall"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
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

    mid = model.body("target").mocapid[0]
    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    # Set up the piper arm
    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    position = [0, 0, 0, 0, 0, 0, 0]

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            q = configuration.q
            # Now convert to the joint values.
            joint_values = scale_and_clip_control(q).tolist()
            piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
            piper.JointCtrl(*joint_values[:6])
            # piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
            piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)

            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
