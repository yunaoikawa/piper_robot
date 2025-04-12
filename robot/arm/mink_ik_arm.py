import numpy as np
import mujoco
import mink

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
        if not self.initalized_:
            raise ValueError("IK solver not initialized")
        self.end_effector_task.set_target(T_wt)
        vel = mink.solve_ik(
            self.configuration, self.tasks, self.solver_dt, solver="quadprog", damping=1e-3, limits=self.limits
        )
        self.configuration.integrate_inplace(vel, self.solver_dt)
        return self.configuration.q