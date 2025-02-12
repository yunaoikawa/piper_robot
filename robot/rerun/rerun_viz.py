import numpy as np
import cv2
import blosc as bl
import zmq
import pickle
import rerun as rr
import time
from robot.nav.mapping import get_pcd_from_image_and_depth


def log_image(image): rr.log("iphone/image", rr.Image(image))


def log_depth(depth): rr.log("iphone/depth", rr.DepthImage(depth))


def log_pose(all_poses, event):
  pose_buffer: np.ndarray = event["value"].to_numpy().astype(np.float32)
  all_poses.append(pose_buffer)
  for pose in all_poses:
    quaternion, translation = pose[:4], pose[4:7]
    rr.log("world/camera", rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=quaternion)))
  return all_poses


def log_map(curr_map, all_poses, image, depth, confidence, pose, focal, resolution):
  all_poses.append(pose)
  rr.log("world/pose", rr.Points3D(positions=[pose[4:7] for pose in all_poses], radii=[0.025 for _ in all_poses]))

  rr.log("world/camera", rr.Transform3D(translation=pose[4:7], rotation=rr.Quaternion(xyzw=pose[:4])))
  rr.log(
    "world/camera",
    rr.Pinhole(
      resolution=(image.shape[1], image.shape[0]),
      focal_length=focal,
      principal_point=resolution,
      camera_xyz=rr.ViewCoordinates.RDF,
      image_plane_distance=0.1,
    ),
  )
  pcd = get_pcd_from_image_and_depth(image, depth, confidence, pose, focal, resolution)

  if curr_map is None:
    curr_map = pcd
  else:
    curr_map += pcd
    curr_map = curr_map.voxel_down_sample(voxel_size=0.05)
  rr.log("world/map", rr.Points3D(positions=np.asarray(curr_map.points), colors=np.asarray(curr_map.colors)))
  return curr_map, all_poses


def main():
  robot_ip = "100.96.33.32"
  context = zmq.Context()
  socket = context.socket(zmq.SUB)
  socket.setsockopt(zmq.CONFLATE, 1)
  socket.setsockopt(zmq.SUBSCRIBE, b"map_info")

  socket.connect(f"tcp://{robot_ip}:5555")

  rr.init("rerun_visualizer", spawn=True)
  rr.log("world/axis", rr.Transform3D(translation=[0, 0, 0], rotation=rr.Quaternion(xyzw=[1, 0, 0, 0]), axis_length=0.5), static=True)
  rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

  curr_map = None
  all_poses = []
  init_timestamp = None

  try:
    while True:
      message = socket.recv()
      message = message.lstrip(b"map_info")
      data = pickle.loads(message)

      encoded_image = np.frombuffer(data["image"], np.uint8)
      image = cv2.imdecode(encoded_image, 1)
      depth = np.array(bl.unpack_array(data["depth"]), dtype=np.float32)
      confidence = np.array(bl.unpack_array(data["confidence"]), dtype=np.uint8)
      pose = np.array(data["pose"], dtype=np.float32)
      focal = data["focal"]
      resolution = data["resolution"]

      if init_timestamp is None: init_timestamp = data["timestamp"]
      print(f"visualizing at timestamp: {data["timestamp"]} | relative: {data["timestamp"] - init_timestamp}")

      tic = time.time()
      curr_map, all_poses = log_map(curr_map, all_poses, image, depth, confidence, pose, focal, resolution)
      print(f"log_map elapsed time: {time.time() - tic:.3f}s")

  finally:
    rr.disconnect()

if __name__ == "__main__":
  main()