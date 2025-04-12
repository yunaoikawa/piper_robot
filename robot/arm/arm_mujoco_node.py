import mujoco
import mujoco.viewer
import mink
import threading
import numpy as np
from loop_rate_limiters import RateLimiter

from robot.arm.mink_ik_arm import ArmIK
from robot.msgs.pose import Pose
from robot.network.node import Node

class ArmMujoco(Node):
    def __init__(self, mjcf_path: str, control_frequency: float = 200.0):
        super().__init__()
        self.mjcf_path = mjcf_path
        self.ik_solver = ArmIK(mjcf_path, solver_dt=1.0 / control_frequency)
        self.control_frequency = control_frequency

        # shared between threads
        self.target_lock_ = threading.Lock()
        self.target: mink.SE3 | None = None

        self.arm_command_sub = self.subscribe("ARM_COMMAND", self.arm_command_handler)
        self.create_thread(self.control_loop)

    def arm_command_handler(self, channel, data):
        pose = Pose.decode(data)
        target = mink.SE3.from_rotation_and_translation(rotation=mink.SO3(np.array(pose.rotation
        )), translation=np.array(pose.translation))
        with self.target_lock_:
            self.target = target
            print(f"{target=}")

    def init(self, model, data):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "ee", "site")
        self.ik_solver.init(data.qpos)
        with self.target_lock_:
            self.target = mink.SE3.from_mocap_name(model, data, "pinch_site_target")

    def get_ee_pose(self, model, data):
        pose = Pose()
        se3 = mink.SE3.from_rotation_and_translation(
            rotation = mink.SO3.from_matrix(data.site_xmat[model.site("ee").id].reshape(3, 3)),
            translation = data.site_xpos[model.site("ee").id]
        )
        pose.translation = se3.translation().tolist()
        pose.rotation = se3.rotation().wxyz.tolist()
        return pose

    def control_loop(self):
        model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        data = mujoco.MjData(model)

        self.init(model, data)

        rate = RateLimiter(frequency=self.control_frequency, warn=True)
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            while viewer.is_running():
                with self.target_lock_:
                    T_wt = self.target
                if T_wt is not None:
                    data.mocap_pos[model.body("pinch_site_target").mocapid[0]] = T_wt.translation()
                    data.mocap_quat[model.body("pinch_site_target").mocapid[0]] = T_wt.rotation().wxyz
                    qd = self.ik_solver.solve_ik(T_wt)
                    data.qpos = qd
                mujoco.mj_step(model, data)
                ee_pose = self.get_ee_pose(model, data)
                self.publish("EE_POSE", ee_pose)
                viewer.sync()
                rate.sleep()

if __name__ == "__main__":
    arm_mujoco = ArmMujoco(mjcf_path="mujoco/scene_piper.xml")
    arm_mujoco.spin()