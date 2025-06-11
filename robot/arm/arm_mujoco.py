import mujoco
import mujoco.viewer
import mink
import time
import numpy as np
import threading
from typing import Optional
from pathlib import Path
import atexit

from loop_rate_limiters import RateLimiter

from robot.arm.ik_solver import ArmIK
from robot.rpc import RPCServer


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
        self.ik_solver = ArmIK(mjcf_path, solver_dt=self.solver_dt)

        self.control_loop_thread: threading.Thread | None = threading.Thread(target=self.control_loop, daemon=True)
        self.control_loop_running = False

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos.copy()[self.ik_solver.dof_ids]

    def get_ee_pose(self) -> mink.SE3:
        q = self.data.qpos.copy()
        self.ik_solver.update_configuration(q)
        return self.ik_solver.forward_kinematics()

    def set_ee_target(self, target: mink.SE3):
        self.ik_solver.update_configuration(self.data.qpos.copy())
        qd, is_solved = self.ik_solver.solve_ik(target)
        print(f"desired q: {np.round(qd, 4)} | is_solved: {is_solved}")
        with self.q_desired_lock:
            self.q_desired = qd

    def set_joint_target(self, joint_target: np.ndarray, lift_target: float=0.0):
        q = np.concatenate([joint_target, np.array([lift_target/2, lift_target/2])])
        with self.q_desired_lock:
            self.q_desired = q

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
        self.viewer.sync()
        self.ik_solver.init(q)
        with self.q_desired_lock:
            self.q_desired = q[self.ik_solver.dof_ids]

    def control_loop(self):
        rate_limiter = RateLimiter(200)
        while self.control_loop_running:
            if self.q_desired is not None:
                with self.q_desired_lock:
                    self.data.ctrl[self.ik_solver.actuator_ids] = self.q_desired[:-2]
                    self.data.ctrl[self.ik_solver.lift_actuator_id] = self.q_desired[-2:].sum()
            # step mujoco
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            rate_limiter.sleep()

if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent
    arm_mujoco = ArmMujoco(mjcf_path=(_HERE / "cone-e-description" / "scene.mjcf").as_posix())
    rpc_server = RPCServer(arm_mujoco, 'localhost', 8081, threaded=False)
    atexit.register(rpc_server.stop)
    rpc_server.start()