from PIL import Image
import numpy as np
import open3d as o3d

def get_pcd_from_image_and_depth(image: np.ndarray, depth: np.ndarray, confidence: np.ndarray, focal: np.ndarray, resolution: np.ndarray) -> np.ndarray:
    rgb_width, rgb_height = image.shape[1], image.shape[0]
    depth_scale = 1000.0
    # Resize depth and confidence maps to match RGB image dimensions
    depth = depth_scale * np.asarray(Image.fromarray(depth).resize((rgb_width, rgb_height)))
    confidence = np.asarray(Image.fromarray(confidence).resize((rgb_width, rgb_height)))
    depth[confidence != 2] = np.nan

    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth).astype(np.float32))
    rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(image).astype(np.uint8))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
    )

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
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    pcd = pcd.voxel_down_sample(voxel_size=0.02)
    return pcd
