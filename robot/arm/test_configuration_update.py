from robot.arm.mink_ik_arm import ArmIK
from pathlib import Path
import numpy as np
from loop_rate_limiters import RateLimiter
import mujoco
import mujoco.viewer

def main():
    _HERE = Path(__file__).parent

    arm_ik = ArmIK(mjcf_path=(_HERE / "mujoco/scene_piper.xml").as_posix())
    arm_ik.init(np.zeros(6))
    arm_ik.update_configuration(np.ones(6))

    rate = RateLimiter(100)

    model = mujoco.MjModel.from_xml_path(_HERE / "mujoco/scene_piper.xml")
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    while viewer.is_running():
        q = data.qpos.copy()
        arm_ik.update_configuration(q)
        ee_pose = arm_ik.forward_kinematics()

        qd = arm_ik.solve_ik(ee_pose)
        data.qpos[arm_ik.dof_ids] = qd
        mujoco.mj_step(model, data)
        viewer.sync()
        rate.sleep()

if __name__ == "__main__":
    main()
