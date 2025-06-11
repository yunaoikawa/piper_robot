from robot.rpc import RPCClient
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def main():
    cone_e = RPCClient("localhost", 8081)

    print("Starting live plot")

    # Create figure and subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Live Joint Positions')

    # Initialize data storage
    max_points = 10000
    joint_positions = np.zeros((6, max_points))
    time_points = np.arange(max_points)

    # Initialize lines for each joint
    lines = []
    for i in range(2):
        for j in range(3):
            line, = axs[i, j].plot([], [], 'b-')
            lines.append(line)
            axs[i, j].set_xlim(0, max_points)
            axs[i, j].set_ylim(-1.0, 1.0)
            axs[i, j].set_xlabel("Time")
            axs[i, j].set_ylabel("Joint Position")
            axs[i, j].set_title(f"Joint {i * 3 + j}")

    plt.tight_layout()

    def update(frame):
        nonlocal joint_positions, time_points
        try:
            # Get new joint positions
            new_positions = cone_e.get_left_joint_positions()[:6]

            # Update data
            joint_positions = np.roll(joint_positions, -1, axis=1)
            joint_positions[:, -1] = new_positions

            # Update each line
            for i, line in enumerate(lines):
                line.set_data(time_points, joint_positions[i])

            return lines
        except Exception as e:
            print(f"Error updating plot: {e}")
            return lines

    # Create animation
    ani = FuncAnimation(fig, update, interval=10, blit=True, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()