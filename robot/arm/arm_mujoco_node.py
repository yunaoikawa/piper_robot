import mujoco
import mujoco.viewer
import mink
import time
from typing import Any, Optional
from pathlib import Path

from dora import Node

from robot.arm.mink_ik_arm import ArmIK
from robot.msgs.pose import Pose


class ArmMujoco:
    def __init__(self, mjcf_path: str, solver_dt: float = 0.03):
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt

        # launch mujoco
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # initialize arm
        self.target: Optional[mink.SE3] = None
        self.ik_solver = ArmIK(mjcf_path, solver_dt=self.solver_dt)

        # communication
        self.node = Node()
        self.init()

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

    def init(self):
        # home
        home_q = self.ik_solver.get_home_q()
        print(f"home_q: {home_q}")
        self.data.qpos[self.ik_solver.dof_ids] = home_q
        mujoco.mj_forward(self.model, self.data)
        time.sleep(1.0)
        q = self.data.qpos.copy()
        mink.move_mocap_to_frame(self.model, self.data, "pinch_site_target", "ee", "site")
        self.viewer.sync()
        self.ik_solver.init(q)
        self.target = self.ik_solver.forward_kinematics()
        print(f"target: {self.target}")

    def step(self):
        q = self.data.qpos.copy()
        # self.ik_solver.update_configuration(q)
        ee_pose = self.ik_solver.forward_kinematics()
        if self.target is not None: # TODO: check timestamp for the target
            # update mocap viz
            self.data.mocap_pos[self.model.body("pinch_site_target").mocapid[0]] = self.target.translation()
            self.data.mocap_quat[self.model.body("pinch_site_target").mocapid[0]] = self.target.rotation().wxyz
            # solve ik
            qd = self.ik_solver.solve_ik(self.target)
            # self.data.ctrl[self.ik_solver.actuator_ids] = qd
            self.data.qpos[self.ik_solver.dof_ids] = qd
        # step mujoco
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()
        # send ee pose
        pose_msg = Pose(time.perf_counter_ns(), ee_pose.wxyz_xyz)
        self.node.send_output("ee_pose", *pose_msg.encode())

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