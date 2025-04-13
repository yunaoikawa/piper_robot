import mujoco
import mujoco.viewer
import mink
import time
from typing import Any
from pathlib import Path

from dora import Node

from robot.arm.mink_ik_arm import ArmIK
from robot.msgs.pose import Pose

class ArmMujoco:
    def __init__(self, mjcf_path: str, control_frequency: float = 200.0):
        super().__init__()
        self.mjcf_path = mjcf_path
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)

        # launch viewer
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # initialize arm
        self.target: mink.SE3 | None = None
        self.ik_solver = ArmIK(mjcf_path, solver_dt=1.0 / control_frequency)
        self.control_frequency = control_frequency

        # communication
        self.node = Node()
        self.init(self.model, self.data)

    def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
        current_time = time.perf_counter_ns()
        delay = (current_time - timestamp) / 1e9
        if delay > max_delay or delay < 0:
            print(f"Skipping message because of delay: {delay}s")
            return False
        return True

    def arm_command_handler(self, event: dict[str, Any]):
        pose = Pose.decode(event["value"], event["metadata"])
        if not self.check_timestamp(pose.timestamp, 0.1):
            return

        target = mink.SE3(pose.wxyz_xyz)
        self.target = target

    def init(self, model, data):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "ee", "site")
        self.ik_solver.init(data.qpos)
        self.target = mink.SE3.from_mocap_name(model, data, "pinch_site_target")

    def get_ee_pose(self, model, data):
        se3 = mink.SE3.from_rotation_and_translation(
            rotation = mink.SO3.from_matrix(data.site_xmat[model.site("ee").id].reshape(3, 3)),
            translation = data.site_xpos[model.site("ee").id]
        )
        pose = Pose(time.perf_counter_ns(), se3.wxyz_xyz)
        return pose

    def step(self):
        T_wt = self.target
        if T_wt is not None:
            self.data.mocap_pos[self.model.body("pinch_site_target").mocapid[0]] = T_wt.translation()
            self.data.mocap_quat[self.model.body("pinch_site_target").mocapid[0]] = T_wt.rotation().wxyz
            qd = self.ik_solver.solve_ik(T_wt)
            self.data.qpos = qd
        mujoco.mj_step(self.model, self.data)
        ee_pose = self.get_ee_pose(self.model, self.data)
        self.node.send_output("ee_pose", *ee_pose.encode())
        self.viewer.sync()

    def stop(self):
        self.viewer.close()

    def spin(self):
        for event in self.node:
            event_type = event["type"]
            if event_type == "INPUT":
                event_id = event["id"]

                if event_id == "arm_command":
                    self.arm_command_handler(event)

                elif event_id == "tick":
                    self.step()

            elif event_type == "STOP":
                self.stop()


if __name__ == "__main__":
    _HERE = Path(__file__).parent
    arm_mujoco = ArmMujoco(mjcf_path=(_HERE / "mujoco/scene_piper.xml").as_posix())
    arm_mujoco.spin()