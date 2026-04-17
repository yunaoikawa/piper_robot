import mujoco
import numpy as np

xml = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <light pos="0 0 3"/>
    <body name="box" pos="0 0 0.5">
      <joint type="free"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

qpos_list = []
qvel_list = []
time_list = []

for _ in range(300):
    mujoco.mj_step(model, data)
    qpos_list.append(data.qpos.copy())
    qvel_list.append(data.qvel.copy())
    time_list.append(data.time)

np.savez(
    "traj.npz",
    qpos=np.asarray(qpos_list),
    qvel=np.asarray(qvel_list),
    time=np.asarray(time_list),
)

print("saved: traj.npz")
print("qpos shape:", np.asarray(qpos_list).shape)
print("qvel shape:", np.asarray(qvel_list).shape)