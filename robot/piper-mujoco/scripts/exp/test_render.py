import mujoco
import numpy as np
from PIL import Image

xml = """
<mujoco>
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

for _ in range(100):
    mujoco.mj_step(model, data)

renderer = mujoco.Renderer(model, width=640, height=480)
renderer.update_scene(data)
img = renderer.render()

Image.fromarray(img).save("frame.png")
print("saved: frame.png", img.shape)