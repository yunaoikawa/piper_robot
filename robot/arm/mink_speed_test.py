from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
from loop_rate_limiters import RateLimiter
from collections import deque

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "mujoco" / "scene_piper.xml"

from robot.arm import fps_counter
from robot.arm.fps_counter import FPSCounter


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]

    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name+"_pos").id for name in joint_names])

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="ee",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    tasks = [end_effector_task]

    limits = [mink.ConfigurationLimit(model)]

    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    configuration.update(data.qpos)

    mujoco.mj_forward(model, data)

    # Initialize the mocap target at the end-effector site.
    mink.move_mocap_to_frame(model, data, "pinch_site_target", "ee", "site")

    fps_counter = FPSCounter()
    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period

    avg_iter = deque(maxlen=50)

    while True:
        T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
        end_effector_task.set_target(T_wt)

        for i in range(max_iters):
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
            configuration.integrate_inplace(vel, rate.dt)

            # Exit condition
            pos_achieved = True
            ori_achieved = True
            err = end_effector_task.compute_error(configuration)
            pos_achieved &= bool(np.linalg.norm(err[:3]) <= pos_threshold)
            ori_achieved &= bool(np.linalg.norm(err[3:]) <= ori_threshold)
            if pos_achieved and ori_achieved:
                avg_iter.append(i)
                break

        data.ctrl[actuator_ids] = configuration.q[dof_ids]
        mujoco.mj_step(model, data)
        fps_counter.getAndPrintFPS(avg_iter)



