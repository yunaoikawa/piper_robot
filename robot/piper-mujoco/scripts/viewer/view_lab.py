import mujoco
import mujoco.viewer
import time

m = mujoco.MjModel.from_xml_path("xml/lab-scene.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 1)

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(0.005)