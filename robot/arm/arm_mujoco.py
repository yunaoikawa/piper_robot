import mujoco
import mujoco.viewer
import mink
import time
import numpy as np
import threading
from typing import Optional
from pathlib import Path

from loop_rate_limiters import RateLimiter
# from dora import Node

from robot.arm.ik_solver import ArmIK
# from robot.msgs.pose import Pose


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
        self.q_desired: Optional[np.ndarray] = None
        self.q_desired_lock = threading.Lock()
        # self.target: Optional[mink.SE3] = None
        self.ik_solver = ArmIK(mjcf_path, solver_dt=self.solver_dt)

        self.control_loop_thread: threading.Thread | None = threading.Thread(target=self.control_loop, daemon=True)
        self.control_loop_running = False

        # communication
        # self.node = Node()
        # self.init()

    def check_timestamp(self, timestamp: int, max_delay: float = 0.1) -> bool:
        current_time = time.perf_counter_ns()
        delay = (current_time - timestamp) / 1e9
        if delay > max_delay or delay < 0:
            print(f"Skipping message because of delay: {delay}s")
            return False
        return True

    # def arm_command_handler(self, event: dict[str, Any]):
    #     pose = Pose.decode(event["value"], event["metadata"])
    #     if not self.check_timestamp(pose.timestamp, 0.1):
    #         return

    #     target = mink.SE3(pose.wxyz_xyz)
    #     self.target = target

    def set_ee_target(self, target: mink.SE3):
        # self.target = target
        qd = self.ik_solver.solve_ik(target)
        with self.q_desired_lock:
            self.q_desired = qd
        # self.data.qpos[self.ik_solver.dof_ids] = qd
        # self.data.ctrl[self.ik_solver.actuator_ids] = qd

    def start_control(self):
        if self.control_loop_thread is None:
            print("To initiate a new control loop, create a new instance of ArmMujoco first")
            return
        self.init()
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self):
        if self.control_loop_thread is None:
            print("Control thread not running")
            return
        self.control_loop_running = False
        self.control_loop_thread.join()
        self.control_loop_thread = None
        self.viewer.close()

    def init(self):
        # home
        home_q = self.ik_solver.get_home_q()
        print(f"home_q: {home_q}")
        self.data.qpos[self.ik_solver.dof_ids] = home_q
        mujoco.mj_forward(self.model, self.data)
        time.sleep(0.1)
        q = self.data.qpos.copy()
        # mink.move_mocap_to_frame(self.model, self.data, "pinch_site_target", "ee", "site")
        self.viewer.sync()
        self.ik_solver.init(q)
        with self.q_desired_lock:
            self.q_desired = q[self.ik_solver.dof_ids]
        # self.target = self.ik_solver.forward_kinematics()
        # print(f"target: {self.target}")

    def control_loop(self):
        rate_limiter = RateLimiter(200)
        while True:
            # q = self.data.qpos.copy()
            # self.ik_solver.update_configuration(q)
            # ee_pose = self.ik_solver.forward_kinematics()
            if self.q_desired is not None: # TODO: check timestamp for the target
                # update mocap viz
                # self.data.mocap_pos[self.model.body("pinch_site_target").mocapid[0]] = self.target.translation()
                # self.data.mocap_quat[self.model.body("pinch_site_target").mocapid[0]] = self.target.rotation().wxyz
                # solve ik
                # qd = self.ik_solver.solve_ik(self.target)
                # self.data.qpos[self.ik_solver.dof_ids] = self.q_desired
                with self.q_desired_lock:
                    self.data.ctrl[self.ik_solver.actuator_ids] = self.q_desired
            # step mujoco
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            rate_limiter.sleep()
        # send ee pose
        # pose_msg = Pose(time.perf_counter_ns(), ee_pose.wxyz_xyz)
        # self.node.send_output("ee_pose", *pose_msg.encode())

    # def spin(self):
    #     for event in self.node:
    #         event_type = event["type"]
    #         if event_type == "INPUT":
    #             event_id = event["id"]

    #             if event_id == "arm_command":
    #                 self.arm_command_handler(event)

    #             elif event_id == "tick":
    #                 self.step()

    #         elif event_type == "STOP":
    #             self.stop()


if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent
    arm_mujoco = ArmMujoco(mjcf_path=(_HERE / "cone-e-description" / "scene.mjcf").as_posix())
    arm_mujoco.start_control()
    # time.sleep(1.0)
    # arm_mujoco.set_ee_target(mink.SE3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    input("Press Enter to stop control")
    arm_mujoco.stop_control()