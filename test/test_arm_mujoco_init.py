from robot.rpc import RPCClient

def main():
    arm_mujoco = RPCClient('localhost', 8081)

    input("Press Enter to initialize ArmMujoco")
    arm_mujoco.start_control()

    input("Press Enter to get ee pose")
    ee_pose = arm_mujoco.get_ee_pose()
    print(f"ee_pose: {ee_pose}")

    input("Press Enter to set ee target")
    target_ee_pose = ee_pose.copy()
    target_ee_pose[-3] += 0.1
    arm_mujoco.set_ee_target(target_ee_pose)

    input("Press Enter to stop control")
    arm_mujoco.stop_control()

if __name__ == "__main__":
    main()