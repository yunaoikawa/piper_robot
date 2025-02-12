import numpy as np
import cv2
import blosc as bl
import zmq
import pickle
import rerun as rr
from robot.nav.mapping import get_pcd_from_image_and_depth


def log_image(event):
    image_buffer: np.ndarray = event["value"].to_numpy().astype(np.uint8)
    encoding = event["metadata"]["encoding"]
    width = event["metadata"]["width"]
    height = event["metadata"]["height"]

    print(encoding, width, height)

    if encoding == "8UC3":
        image = image_buffer.reshape((height, width, 3))
        rr.log("iphone/image", rr.Image(image))

def log_depth(event):
    depth_buffer: np.ndarray = event["value"].to_numpy().astype(np.float32)
    width = event["metadata"]["width"]
    height = event["metadata"]["height"]
    depth = depth_buffer.reshape((height, width))
    rr.log("iphone/depth", rr.DepthImage(depth))

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
    rr.log("world/camera",
        rr.Pinhole(resolution=(image.shape[1], image.shape[0]), focal_length=focal, principal_point=resolution, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.1)
    )
    pcd = get_pcd_from_image_and_depth(image, depth, confidence, pose, focal, resolution)

    if curr_map is None: curr_map = pcd
    else:
        curr_map += pcd
        curr_map = curr_map.voxel_down_sample(voxel_size=0.02)
    rr.log("world/map", rr.Points3D(positions=np.asarray(curr_map.points), colors=np.asarray(curr_map.colors)))
    return curr_map, all_poses

def main():
    robot_ip = ""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, b"map_info")

    socket.connect(f"tcp://{robot_ip}:5555")

    rr.init("rerun_visualizer", spawn=True)
    # rr.serve_web(open_browser=False, server_memory_limit="1GB")
    # rr.save("rerun_test.rrd")
    rr.log("world/axis", rr.Transform3D(translation=[0, 0, 0], rotation=rr.Quaternion(xyzw=[1, 0, 0, 0]), axis_length=0.5), static=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

    curr_map = None
    all_poses = []

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

            curr_map, all_poses = log_map(curr_map, all_poses, image, depth, confidence, pose, focal, resolution)
    finally:
        rr.disconnect()


