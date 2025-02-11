import numpy as np

import rerun as rr
from dora import Node

from robot.nav.mapping import get_pcd_from_image_and_depth

def get_int_mat(focal, resolution):
    return np.array([[focal[0], 0, resolution[0]], [0, focal[1], resolution[1]], [0, 0, 1]], dtype=np.float32)


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

def log_map(curr_map, all_poses, image_event, depth_event, curr_confidence_event, pose_event):
    image_time = float(image_event["metadata"]["timestamp"])
    depth_time = float(depth_event["metadata"]["timestamp"])
    pose_time = float(pose_event["metadata"]["timestamp"])
    max_diff = max(abs(image_time - depth_time), abs(image_time - pose_time), abs(depth_time - pose_time))
    if max_diff > 0.04:
        print(f"Error: Event timestamps not synchronized. Max difference: {max_diff:.3f}s")
        return

    print("Logging map")

    image = image_event["value"].to_numpy().astype(np.uint8).reshape((image_event["metadata"]["height"], image_event["metadata"]["width"], 3))
    depth = depth_event["value"].to_numpy().astype(np.float32).reshape((depth_event["metadata"]["height"], depth_event["metadata"]["width"]))
    confidence = curr_confidence_event["value"].to_numpy().astype(np.uint8).reshape((depth_event["metadata"]["height"], depth_event["metadata"]["width"]))
    pose = pose_event["value"].to_numpy().astype(np.float32)

    focal = depth_event["metadata"]["focal"]
    resolution = depth_event["metadata"]["resolution"]

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
    node = Node("rerun")

    rr.init("rerun_visualizer")
    # rr.serve_web(open_browser=False, server_memory_limit="1GB")
    rr.save("rerun_test.rrd")
    rr.log("world/axis", rr.Transform3D(translation=[0, 0, 0], rotation=rr.Quaternion(xyzw=[1, 0, 0, 0]), axis_length=0.5), static=True)

    curr_image_event = None
    curr_depth_event = None
    curr_confidence_event = None
    curr_pose_event = None

    curr_map = None
    all_poses = []

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "iphone/image":
                curr_image_event = event
            elif event["id"] == "iphone/depth":
                curr_depth_event = event
            elif event["id"] == "iphone/pose":
                curr_pose_event = event
            elif event["id"] == "iphone/confidence":
                curr_confidence_event = event
            elif event["id"] == "tick":
                if curr_image_event is not None: log_image(curr_image_event)
                if curr_depth_event is not None: log_depth(curr_depth_event)
                if curr_confidence_event is not None: pass
                # if curr_pose_event is not None: all_poses = log_pose(all_poses, curr_pose_event)

                if curr_image_event is not None and curr_depth_event is not None and curr_confidence_event is not None and curr_pose_event is not None:
                    curr_map, all_poses = log_map(curr_map, all_poses, curr_image_event, curr_depth_event, curr_confidence_event, curr_pose_event)

    rr.disconnect()
if __name__ == "__main__":
    main()

