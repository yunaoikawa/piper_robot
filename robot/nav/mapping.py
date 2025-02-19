from PIL import Image
import numpy as np
import cv2
import open3d as o3d
from quaternion import as_rotation_matrix, quaternion


def get_pcd_from_image_and_depth(
    image: np.ndarray, depth: np.ndarray, confidence: np.ndarray, pose: np.ndarray, focal: list, resolution: list
) -> np.ndarray:
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)

    rgb_width, rgb_height = image.shape[0], image.shape[1]
    depth_scale = 1000.0
    # Resize depth and confidence maps to match RGB image dimensions
    depth = depth_scale * np.asarray(Image.fromarray(depth).resize((rgb_height, rgb_width)))
    confidence = np.asarray(Image.fromarray(confidence).resize((rgb_height, rgb_width)))
    depth[confidence != 2] = np.nan

    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(image).astype(np.uint8))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=rgb_width,
        height=rgb_height,
        fx=focal[0],
        fy=focal[1],
        cx=resolution[0],
        cy=resolution[1],
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

    # Flip the pcd
    flip_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # flip_transform_inverse = np.linalg.inv(flip_transform)
    # pcd.transform(flip_transform)

    quat, translation = pose[:4], pose[4:7]
    qx, qy, qz, qw = quat
    transform = np.eye(4)
    transform[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
    transform[:3, 3] = translation
    pcd.transform(transform)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    # pcd.transform(flip_transform_inverse)

    return pcd
