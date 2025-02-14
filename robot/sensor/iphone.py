import numpy as np
from threading import Event
import zmq
import time

from record3d import Record3DStream, CameraPose
from robot.msgs import EncodedImage, EncodedDepth, Pose
from robot.communications import Publisher, BASE_CAMERA_PORT, FrequencyTimer

new_frame_event = Event()
stop_event = Event()


def on_new_frame():
    new_frame_event.set()


def on_stream_stopped():
    stop_event.set()


def get_pose_array_from_pose(pose: CameraPose) -> np.ndarray:
    return np.array([pose.qx, pose.qy, pose.qz, pose.qw, pose.tx, pose.ty, pose.tz])


def main(device_id: int):
    # create publisher
    ctx = zmq.Context()
    pub = Publisher(ctx, BASE_CAMERA_PORT)
    timer = FrequencyTimer("Camera", 50)

    # start stream
    devices = Record3DStream.get_connected_devices()
    device = devices[device_id]

    session = Record3DStream()
    session.on_new_frame = on_new_frame
    session.on_stream_stopped = on_stream_stopped
    session.connect(device)

    while True:
        with timer:
            if stop_event.is_set(): break

            if not new_frame_event.wait(0.1):
                continue

            rgb = session.get_rgb_frame()
            depth = session.get_depth_frame()
            confidence = session.get_confidence_frame()
            intrinsics = session.get_intrinsic_mat()
            pose = get_pose_array_from_pose(session.get_camera_pose())

            timestamp = time.perf_counter_ns()
            pub.publish("/base/image", EncodedImage(
                timestamp=timestamp,
                image=rgb,
                encoding="jpg"
            ))

            pub.publish("/base/depth", EncodedDepth(
                timestamp=timestamp,
                depth=depth,
                confidence=confidence,
                focal=[int(intrinsics.fx), int(intrinsics.fy)],
                resolution=[int(intrinsics.tx), int(intrinsics.ty)]
            ))

            pub.publish("/base/pose", Pose(
                timestamp=timestamp,
                pose=pose
            ))

        new_frame_event.clear()

if __name__ == "__main__":
    main(0)
