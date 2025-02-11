import numpy as np
import time

import rerun as rr
from dora import Node

from robot.constants import LENGTH, WIDTH
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

def log_pose(event):
    # Create coordinate frame arrows
    frame_length = 1.0  # 1 meter arrows
    x_axis = np.array([[0, 0], [frame_length, 0]])
    y_axis = np.array([[0, 0], [0, -frame_length]])

    buffer = event["value"].to_numpy()
    state = np.frombuffer(buffer, dtype=float)

    rr.log("world/x_axis", rr.LineStrips2D([x_axis], colors=(255, 0, 0)))
    rr.log("world/y_axis", rr.LineStrips2D([y_axis], colors=[0, 0, 255]))

    x, y, theta = state
    y = -y  # Invert y-axis
    theta = -theta  # Invert theta
    corners = np.array([[LENGTH, WIDTH], [LENGTH, -WIDTH], [-LENGTH, -WIDTH], [-LENGTH, WIDTH], [LENGTH, WIDTH]])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = (R @ corners.T).T + np.array([x, y])

    rr.set_time_seconds("real_time", time.time())
    rr.log('robot/base', rr.LineStrips2D([corners]))

    arrow_length = 0.7 * LENGTH * 2
    arrow_tip = np.array([x, y]) + arrow_length * np.array([np.cos(theta), np.sin(theta)])
    arrow_line = np.array([[x, y], arrow_tip])
    rr.log("robot/arrow", rr.LineStrips2D([arrow_line]))

def log_map(curr_map, image_event, depth_event, curr_confidence_event, pose_event):
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
    focal = depth_event["metadata"]["focal"]
    resolution = depth_event["metadata"]["resolution"]
    # pose = pose_event["value"].to_numpy().astype(np.float32)

    pcd = get_pcd_from_image_and_depth(image, depth, confidence, focal, resolution)

    if curr_map is None: curr_map = pcd
    else:
        curr_map += pcd
        curr_map = curr_map.voxel_down_sample(voxel_size=0.02)
    rr.log("world/map", rr.Points3D(positions=np.asarray(pcd.points), colors=np.asarray(pcd.colors)))


def main():
    node = Node("rerun")

    rr.init("rerun_visualizer")
    # rr.serve_web(open_browser=False, server_memory_limit="1GB")
    rr.save("rerun_test.rrd")

    curr_image_event = None
    curr_depth_event = None
    curr_confidence_event = None
    curr_pose_event = None

    curr_map = None

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
                if curr_pose_event is not None: pass # log_pose(curr_pose_event)

                if curr_image_event is not None and curr_depth_event is not None and curr_confidence_event is not None and curr_pose_event is not None:
                    log_map(curr_map, curr_image_event, curr_depth_event, curr_confidence_event, curr_pose_event)

    rr.disconnect()
if __name__ == "__main__":
    main()

