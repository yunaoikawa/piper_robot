from robot.rpc import RPCClient
import rerun as rr
from loop_rate_limiters import RateLimiter

ENES_MACBOOK_IP = "10.19.143.218:9876"

def main():
    cone_e = RPCClient("localhost", 8081)
    print("Starting live plot with Rerun")

    # Initialize Rerun
    rr.init("arm_joint_plotter")
    rr.connect(addr=ENES_MACBOOK_IP)

    rate_limiter = RateLimiter(10)

    try:
        while True:
            new_positions = cone_e.get_left_joint_positions()[:6]

            for i in range(6):
                rr.log(f"joint_{i}", rr.Scalar(new_positions[i]))

            rate_limiter.sleep()

    except KeyboardInterrupt:
        print("\nStopping visualization")
    except Exception as e:
        print(f"Error updating plot: {e}")

if __name__ == "__main__":
    main()