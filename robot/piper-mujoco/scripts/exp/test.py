import mujoco

xml = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 0.5">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

for _ in range(100):
    mujoco.mj_step(model, data)

print("MuJoCo step OK")
print("time =", data.time)