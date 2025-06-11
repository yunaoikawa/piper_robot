from robot.rpc import RPCClient

def main():
    arm_mujoco = RPCClient('localhost', 8081)

    input("Press Enter to initialize ArmMujoco")
    arm_mujoco.start_control()
    input("Press Enter to stop control")
    arm_mujoco.stop_control()

if __name__ == "__main__":
    main()