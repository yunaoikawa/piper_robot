from typing import Any, cast
import numpy as np
import numpy.typing as npt
import zmq
import cv2
import liblzfse
import rerun as rr

from robot.communications import Subscriber, FrequencyTimer, BASE_CAMERA_PORT, ROBOT_IP
from robot.msgs import EncodedImage, EncodedDepth, Pose, s
from robot.nav.mapping import get_pcd_from_image_and_depth


def log_image(image: npt.NDArray[np.uint8]):
    image = cast(npt.NDArray[np.uint8], cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    rr.log("base/image", rr.Image(image))


def log_depth(depth: npt.NDArray[np.float32]):
    depth = cast(npt.NDArray[np.float32], cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE))
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
    all_poses: list[np.ndarray],
    image: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    pose: np.ndarray,
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
    timer = FrequencyTimer("Rerun", 10, delay_warn_threshold=s(0.01))

    rr.init("rerun_visualizer", spawn=True)
    rr.log(
        "world/axis",
        rr.Transform3D(translation=[0, 0, 0], rotation=rr.Quaternion(xyzw=[1, 0, 0, 0]), axis_length=0.5),
        static=True,
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

    curr_map = None
    all_poses: list[np.ndarray] = []

    try:
        while True:
            with timer:
                datum: dict[str, Any] = {}
                while len(datum) < 3:
                    topic, data = sub.receive()
                    if topic is not None:
                        datum[topic] = data

                image_buffer: np.ndarray = datum["/base/image"].image
                image = cast(npt.NDArray[np.uint8], cv2.imdecode(image_buffer, cv2.IMREAD_COLOR))

                encoded_depth: EncodedDepth = datum["/base/depth"]
                depth = np.frombuffer(liblzfse.decompress(encoded_depth.depth), dtype=np.float32).reshape(
                    encoded_depth.width, encoded_depth.height
                )
                confidence = np.frombuffer(liblzfse.decompress(encoded_depth.confidence), dtype=np.uint8).reshape(
                    encoded_depth.width, encoded_depth.height
                )
                pose = datum["/base/pose"].pose

                curr_map, all_poses = log_map(
                    curr_map, all_poses, image, depth, confidence, pose, encoded_depth.focal, encoded_depth.resolution
                )

                log_image(image)
                log_depth(depth)

            # encoded_image = np.frombuffer(data["image"], np.uint8)
            # image = cv2.imdecode(encoded_image, 1)
            # depth = np.array(bl.unpack_array(data["depth"]), dtype=np.float32)
            # confidence = np.array(bl.unpack_array(data["confidence"]), dtype=np.uint8)
            # pose = np.array(data["pose"], dtype=np.float32)
            # focal = data["focal"]
            # resolution = data["resolution"]

            # if init_timestamp is None:
            #     init_timestamp = float(data["timestamp"])
            # print(
            #     f"visualizing at timestamp: {data['timestamp']} | relative: {float(data['timestamp']) - init_timestamp}"
            # )
            # print(f"delay @ viz: {time.time() - float(data['timestamp']):.3f}s")

            # tic = time.time()
            # curr_map, all_poses = log_map(curr_map, all_poses, image, depth, confidence, pose, focal, resolution)
            # log_image(image)
            # log_depth(depth)
            # print(f"log_map elapsed time: {time.time() - tic:.3f}s")

    finally:
        rr.disconnect()


if __name__ == "__main__":
    main()
