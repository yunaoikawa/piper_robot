import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "mujoco_menagerie/agilex_piper/scene.xml"

traj = np.load("traj_piper.npz")
qpos_seq = traj["qpos"]
time_seq = traj["time"]

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

n_frames = len(qpos_seq)

# GLFW key codes
GLFW_KEY_SPACE = 32
GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT  = 263
GLFW_KEY_R     = 82

state = {"frame": 0, "paused": False, "step": 0}

def key_callback(keycode):
    if keycode == GLFW_KEY_SPACE:
        state["paused"] = not state["paused"]
    elif keycode == GLFW_KEY_R:
        state["frame"] = 0
        state["paused"] = False
    elif keycode == GLFW_KEY_RIGHT and state["paused"]:
        state["step"] = 1
    elif keycode == GLFW_KEY_LEFT and state["paused"]:
        state["step"] = -1

print("Controls: Space=pause/resume  R=restart  ←/→=frame step (when paused)  Close window to quit")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        i = state["frame"]

        data.qpos[:6] = qpos_seq[i]
        mujoco.mj_forward(model, data)
        viewer.sync()

        if state["paused"]:
            if state["step"] != 0:
                state["frame"] = int(np.clip(i + state["step"], 0, n_frames - 1))
                state["step"] = 0
            time.sleep(0.016)
        else:
            next_i = (i + 1) % n_frames
            if next_i > 0:
                dt = max(0.0, float(time_seq[next_i] - time_seq[i]))
            else:
                dt = float(time_seq[1] - time_seq[0]) if n_frames > 1 else 0.01
            time.sleep(dt)
            state["frame"] = next_i

print("replay finished")
