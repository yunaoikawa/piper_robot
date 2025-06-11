from robot.rpc import RPCClient
import numpy as np

def main():
    arm_mujoco = RPCClient('localhost', 8081)

    input("Press Enter to initialize ArmMujoco")
    arm_mujoco.start_control()

    input("Press Enter to set ee target")
    target_1 = arm_mujoco.get_ee_pose()
    target_1.wxyz_xyz[-3] += 0.1
    arm_mujoco.set_ee_target(target_1)

    input("Next step..")
    # get difference between target_1 and target_2
    target_2 = arm_mujoco.get_ee_pose()
    q_err = arm_mujoco.get_joint_positions() - arm_mujoco.q_desired
    print(f"q_err: {q_err}")
    tracking_error = target_2.minus(target_1)
    print(f"pos_err: {np.linalg.norm(tracking_error[:3])} | ori_err: {np.linalg.norm(tracking_error[3:])}")

    target_2.wxyz_xyz[-2] += 0.1
    arm_mujoco.set_ee_target(target_2)

    input("Next step..")
    target_3 = arm_mujoco.get_ee_pose()
    tracking_error = target_3.minus(target_2)
    print(f"pos_err: {np.linalg.norm(tracking_error[:3])} | ori_err: {np.linalg.norm(tracking_error[3:])}")

    target_3.wxyz_xyz[-2] += 0.1
    arm_mujoco.set_ee_target(target_3)

    input("Next step..")
    target_4 = arm_mujoco.get_ee_pose()
    tracking_error = target_4.minus(target_3)
    print(f"pos_err: {np.linalg.norm(tracking_error[:3])} | ori_err: {np.linalg.norm(tracking_error[3:])}")

    target_4.wxyz_xyz[-1] += 0.1
    arm_mujoco.set_ee_target(target_4)

    input("Next step..")
    target_5 = arm_mujoco.get_ee_pose()
    tracking_error = target_5.minus(target_4)
    print(f"pos_err: {np.linalg.norm(tracking_error[:3])} | ori_err: {np.linalg.norm(tracking_error[3:])}")

    input("Press Enter to stop control")
    arm_mujoco.stop_control()

if __name__ == "__main__":
    main()