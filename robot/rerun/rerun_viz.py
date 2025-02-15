from typing import Any, cast
import numpy as np
import zmq
import cv2
import liblzfse
import rerun as rr

from robot.network import Subscriber, BASE_CAMERA_PORT, ROBOT_IP
from robot.network.timer import FrequencyTimer
from robot.network.msgs import EncodedImage, EncodedDepth, Pose, Buffer
from robot.nav.mapping import get_pcd_from_image_and_depth


def check_timestamp(image_ts, depth_ts, pose_ts) -> bool:
    return max(abs(image_ts - depth_ts), abs(image_ts - pose_ts), abs(depth_ts - pose_ts)) > 1e6  # 1ms


def log_image(image):
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rr.log("base/image", rr.Image(image))


def log_depth(depth):
    depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
    rr.log("base/depth", rr.DepthImage(depth))


def log_pose(all_poses, event):
    pose_buffer: np.ndarray = event["value"].to_numpy().astype(np.float32)
    all_poses.append(pose_buffer)
    for pose in all_poses:
        quaternion, translation = pose[:4], pose[4:7]
        rr.log("world/camera", rr.Transform3D(translation=translation, rotation=rr.Quaternion(xyzw=quaternion)))
    return all_poses


def log_map(
    curr_map,
    all_poses: list[Buffer],
    image: Buffer,
    depth: Buffer,
    confidence: Buffer,
    pose: Buffer,
    focal: list,
    resolution: list,
):
    all_poses.append(pose)
    rr.log(
        "world/pose",
        rr.Points3D(
            positions=[pose[4:7] for pose in all_poses],
            radii=[0.05 for _ in all_poses],
            colors=[(0, 255, 0) for _ in all_poses],
        ),
    )

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
    ctx = zmq.Context()
    sub = Subscriber(
        ctx,
        BASE_CAMERA_PORT,
        ["/base/image", "/base/depth", "/base/pose"],
        [EncodedImage.deserialize, EncodedDepth.deserialize, Pose.deserialize],
        host=ROBOT_IP,
    )
    timer = FrequencyTimer("Rerun", 2, delay_warn_threshold=0.01)

    rr.init("rerun_visualizer", spawn=True)
    rr.log(
        "world/axis",
        rr.Transform3D(translation=[0, 0, 0], rotation=rr.Quaternion(xyzw=[1, 0, 0, 0]), axis_length=0.5),
        static=True,
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

    curr_map = None
    all_poses: list[Buffer] = []

    try:
        while True:
            with timer:
                datum: dict[str, Any] = {}
                while len(datum) < 3:
                    topic, data = sub.receive()
                    if topic is not None:
                        datum[topic] = data

                image_msg = cast(EncodedImage, datum["/base/image"])
                depth_msg = cast(EncodedDepth, datum["/base/depth"])
                pose_msg = cast(Pose, datum["/base/pose"])

                if check_timestamp(image_msg.timestamp, depth_msg.timestamp, pose_msg.timestamp):
                    print("Timestamps do not match")
                    continue

                image = cv2.cvtColor(cv2.imdecode(image_msg.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                depth = np.frombuffer(liblzfse.decompress(depth_msg.depth), dtype=np.float32).reshape(
                    depth_msg.width, depth_msg.height
                )
                confidence = np.frombuffer(liblzfse.decompress(depth_msg.confidence), dtype=np.uint8).reshape(
                    depth_msg.width, depth_msg.height
                )
                pose = pose_msg.pose

                curr_map, all_poses = log_map(
                    curr_map, all_poses, image, depth, confidence, pose, depth_msg.focal, depth_msg.resolution
                )

                log_image(image)
                log_depth(depth)

    finally:
        rr.disconnect()


if __name__ == "__main__":
    main()
