import pyarrow as pa
import numpy as np
from threading import Event

from dora import Node
from record3d import Record3DStream

new_frame_event = Event()
stop_event = Event()

def on_new_frame(): new_frame_event.set()

def on_stream_stopped(): stop_event.set()

def main(device_id: int):
  # start stream
  devices = Record3DStream.get_connected_devices()
  for dev in devices: print(f"\tID: {dev.product_id}")
  device = devices[device_id]
  session = Record3DStream()
  session.on_new_frame = on_new_frame
  session.on_stream_stopped = on_stream_stopped
  session.connect(device)

  # create node
  node = Node(node_id=f"camera-{device_id}")

  for event in node:
    if event["type"] == "INPUT" and event["id"] == "tick":
      if stop_event.is_set(): break
      new_frame_event.wait()  # wait for new frame

      rgb = session.get_rgb_frame()
      depth = session.get_depth_frame()
      intrinsics = session.get_intrinsic_mat_from_coeffs(session.get_intrinsic_mat())
      confidence = session.get_confidence_frame()
      pose = session.get_camera_pose()
      timestamp = session.get_misc_data()["metadata"]["relativeTimestamp"]

      if depth.shape != rgb.shape: pass  # TODO: resize depth map

      # RGB output
      node.send_output("image", pa.array(rgb.ravel()), {"timestamp": timestamp, "encoding": "8UC3", "width": rgb.shape[1], "height": rgb.shape[0]})

      # Depth output with focal length and encoding info
      node.send_output(
        "depth",
        pa.array(depth.ravel().astype(np.float64)),
        {
          "timestamp": timestamp,
          "width": depth.shape[1],
          "height": depth.shape[0],
          "encoding": "64FC1",
          "focal": [int(intrinsics[0, 0]), int(intrinsics[1, 1])],
          "resolution": [int(intrinsics[0, 2]), int(intrinsics[1, 2])],
        },
      )

      # Confidence output
      node.send_output(
        "confidence",
        pa.array(confidence.ravel()),
        {"timestamp": timestamp, "width": confidence.shape[1], "height": confidence.shape[0], "encoding": "8UC1"},
      )

      # Pose output
      node.send_output("pose", pa.array(pose.ravel()), {"timestamp": timestamp})
    new_frame_event.clear()


if __name__ == "__main__":
  main(0)