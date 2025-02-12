import numpy as np
import zmq
import cv2
import blosc as bl
import pickle
import time

from dora import Node

def pub_map_info(socket: zmq.Socket, image_event, depth_event, curr_confidence_event, pose_event):
    image_time = float(image_event["metadata"]["timestamp"])
    depth_time = float(depth_event["metadata"]["timestamp"])
    pose_time = float(pose_event["metadata"]["timestamp"])
    max_diff = max(abs(image_time - depth_time), abs(image_time - pose_time), abs(depth_time - pose_time))
    if max_diff > 0.04:
        print(f"Error: Event timestamps not synchronized. Max difference: {max_diff:.3f}s")
        return
    print("delay @ pub_map_info:", time.time() - image_time)

    image_md, depth_md = image_event["metadata"], depth_event["metadata"]

    image = image_event["value"].to_numpy().astype(np.uint8).reshape((image_md["height"], image_md["width"], 3))
    depth = depth_event["value"].to_numpy().astype(np.float32).reshape((depth_md["height"], depth_md["width"]))
    confidence = curr_confidence_event["value"].to_numpy().astype(np.uint8).reshape((depth_md["height"], depth_md["width"]))
    pose = pose_event["value"].to_numpy().astype(np.float32)

    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    compressed_depth = bl.pack_array(
        depth, cname="zstd", clevel=1, shuffle=bl.NOSHUFFLE
    )
    compressed_confidence = bl.pack_array(
        confidence, cname="zstd", clevel=1, shuffle=bl.NOSHUFFLE
    )
    data = {
        "image": buffer, "depth": compressed_depth, "confidence": compressed_confidence,
        "pose": pose, "focal": depth_md["focal"], "resolution": depth_md["resolution"], "timestamp": image_md["timestamp"]
    }
    print(f"Publishing map info with timestamp: {image_md['timestamp']}")
    socket.send(b"map_info" + pickle.dumps(data, protocol=-1))


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    socket.setsockopt(zmq.HWM, 2)

    node = Node("rerun")

    curr_image_event = None
    curr_depth_event = None
    curr_confidence_event = None
    curr_pose_event = None

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

                if curr_image_event is not None and curr_depth_event is not None and curr_confidence_event is not None and curr_pose_event is not None:
                    tic = time.time()
                    pub_map_info(socket, curr_image_event, curr_depth_event, curr_confidence_event, curr_pose_event)
                    print(f"Publishing took {time.time() - tic:.3f}s")


if __name__ == "__main__":
    main()

