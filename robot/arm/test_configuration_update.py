from robot.arm.mink_ik_arm import ArmIK
from pathlib import Path
import numpy as np
from loop_rate_limiters import RateLimiter
import mujoco
import mujoco.viewer
import mink

def main():
    _HERE = Path(__file__).parent

    arm_ik = ArmIK(mjcf_path=(_HERE / "mujoco/scene_piper.xml").as_posix(), solver_dt=0.01)
    arm_ik.init(np.zeros(6))
    # arm_ik.update_configuration(np.ones(6))

    model = mujoco.MjModel.from_xml_path((_HERE / "mujoco/scene_piper.xml").as_posix())
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        rate = RateLimiter(100)
        update_q_counter = 0
        while viewer.is_running():
            ee_pose = arm_ik.forward_kinematics()
            mocap_pose = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            qd = arm_ik.solve_ik(mocap_pose)
            data.qpos[arm_ik.dof_ids] = qd
            # data.ctrl[arm_ik.actuator_ids] = qd
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()
            update_q_counter += 1

if __name__ == "__main__":
    main()
