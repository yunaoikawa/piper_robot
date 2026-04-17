import time
import mujoco
import mujoco.viewer
import numpy as np

traj = np.load("traj.npz")
qpos_seq = traj["qpos"]
qvel_seq = traj["qvel"]
time_seq = traj["time"]

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for i in range(len(qpos_seq)):
        data.qpos[:] = qpos_seq[i]
        data.qvel[:] = qvel_seq[i]
        mujoco.mj_forward(model, data)
        viewer.sync()

        if i > 0:
            dt = max(0.0, float(time_seq[i] - time_seq[i - 1]))
            time.sleep(dt)

print("replay finished")