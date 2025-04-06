from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

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

    while True:
        data.ctrl[actuator_ids] = configuration.q[dof_ids]
        mujoco.mj_step(model, data)
        fps_counter.getAndPrintFPS()



