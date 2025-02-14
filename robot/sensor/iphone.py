import pyarrow as pa
import numpy as np
import json
from threading import Event

from dora import Node
from record3d import Record3DStream, CameraPose

new_frame_event = Event()
stop_event = Event()


def on_new_frame():
    new_frame_event.set()


def on_stream_stopped():
    stop_event.set()


def get_timestamp_from_misc_data(misc_data: np.ndarray) -> float:
    metadata = misc_data.tobytes().decode("ascii")
    return json.loads(metadata)["metadata"]["unixTimestampOnReceivedFrame"]


def get_pose_array_from_pose(pose: CameraPose) -> np.ndarray:
    return np.array([pose.qx, pose.qy, pose.qz, pose.qw, pose.tx, pose.ty, pose.tz])


def main(device_id: int):
    # start stream
    devices = Record3DStream.get_connected_devices()
    for dev in devices:
        print(f"\tID: {dev.product_id}")
    device = devices[device_id]
    session = Record3DStream()
    session.on_new_frame = on_new_frame
    session.on_stream_stopped = on_stream_stopped
    session.connect(device)

    # create node
    node = Node(node_id=f"camera-{device_id}")

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "tick":
            if stop_event.is_set():
                break
            rx_frame = new_frame_event.wait(0.1)  # wait for new frame
            if not rx_frame:
                continue

            rgb = session.get_rgb_frame()
            depth = session.get_depth_frame()
            intrinsics = session.get_intrinsic_mat()
            confidence = session.get_confidence_frame()
            pose = get_pose_array_from_pose(session.get_camera_pose())
            timestamp = str(get_timestamp_from_misc_data(session.get_misc_data()))

            # RGB output
            node.send_output(
                "image",
                pa.array(rgb.ravel()),
                {"timestamp": timestamp, "encoding": "8UC3", "width": rgb.shape[1], "height": rgb.shape[0]},
            )

            # Depth output with focal length and encoding info
            node.send_output(
                "depth",
                pa.array(depth.ravel().astype(np.float32)),
                {
                    "timestamp": timestamp,
                    "width": depth.shape[1],
                    "height": depth.shape[0],
                    "encoding": "32FC1",
                    "focal": [int(intrinsics.fx), int(intrinsics.fy)],
                    "resolution": [int(intrinsics.tx), int(intrinsics.ty)],
                },
            )

            # Confidence output
            node.send_output(
                "confidence",
                pa.array(confidence.ravel()),
                {
                    "timestamp": timestamp,
                    "width": confidence.shape[1],
                    "height": confidence.shape[0],
                    "encoding": "8UC1",
                },
            )

            # Pose output
            node.send_output("pose", pa.array(pose.ravel()), {"timestamp": timestamp})
        new_frame_event.clear()


if __name__ == "__main__":
    main(0)
