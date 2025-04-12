from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import lcm

import mink

from robot.arm.fps_counter import FPSCounter
from robot.msgs.pose import Pose

_HERE = Path(__file__).parent
_XML = _HERE / "mujoco" / "scene_piper.xml"
CTRL_ENABLED = False

class ArmIK:
    def __init__(self, mjcf_path: str, solver_dt=0.033):
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.solver_dt = solver_dt

        joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]

        velocity_limits = {k: np.pi/2 if "joint" in k else 0.05 for k in joint_names}
        self.dof_ids = np.array([model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([model.actuator(name + "_pos").id for name in joint_names])

        self.configuration = mink.Configuration(model)
        self.end_effector_task = mink.FrameTask(
            frame_name="ee",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.1,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model, cost=np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))
        self.tasks = [self.end_effector_task, self.posture_task]
        self.limits = [mink.ConfigurationLimit(model), mink.VelocityLimit(model, velocity_limits)]

        # initial setup
        self.initalized_ = False

    def init(self, q):
        self.configuration.update(q)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.initalized_ = True

    def solve_ik(self, T_wt: mink.SE3):
        self.end_effector_task.set_target(T_wt)
        vel = mink.solve_ik(
            self.configuration, self.tasks, self.solver_dt, solver="quadprog", damping=1e-3, limits=self.limits
        )
        self.configuration.integrate_inplace(vel, self.solver_dt)
        return self.configuration.q


def main():
    target: mink.SE3 | None = None
    lc = lcm.LCM()
    rate = RateLimiter(frequency=100.0, warn=False)

    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    arm_ik = ArmIK(model, solver_dt=rate.dt)

    def arm_command_handler(channel, data):
        nonlocal target
        pose = Pose.decode(data)
        target = mink.SE3.from_rotation_and_translation(rotation=mink.SO3(pose.orientation), translation=pose.position)

    lc.subscribe("/arm_command", arm_command_handler)
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        arm_ik.init(data.qpos)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "ee", "site")

        fps_counter = FPSCounter()
        while viewer.is_running():
            if target is not None:
                data.mocap_pos[model.body("pinch_site_target").mocapid[0]] = target.translation()
                data.mocap_quat[model.body("pinch_site_target").mocapid[0]] = target.rotation().wxyz
            qd = arm_ik.solve_ik(T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target"))
            data.qpos = qd
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()
            fps = fps_counter.tick()
            if fps is not None:
                print(f"{fps=:.3f}")


if __name__ =="__main__":
    main()







