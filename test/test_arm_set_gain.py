from robot.rpc import RPCClient
import numpy as np

def main():
    cone_e = RPCClient("localhost", 8081)

    cone_e.set_left_gain(np.array([15, 15, 15, 15, 15, 15]), np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
    cone_e.set_right_gain(np.array([5, 5, 5, 5, 5, 5]), np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))

    input("Press Enter to continue")

if __name__ == "__main__":
    main()