"""Shadow-independent incubator front-plane estimation from head RGB-D."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from rollout.apriltag_retarget import detect_tags


def wrap_degrees(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def _robot_tag_square(tag_config: dict, tag_id: int, size_m: float, z_m: float):
    plane = np.asarray(tag_config["fixed_plane_corners"][str(tag_id)], dtype=float)
    mapping = np.asarray(tag_config["plane_to_robot_xy"], dtype=float)
    approximate = plane @ mapping.T
    center = approximate.mean(axis=0)
    axis_x = ((approximate[1] - approximate[0]) + (approximate[2] - approximate[3])) / 2
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.asarray([-axis_x[1], axis_x[0]])
    if axis_y @ (approximate[0] - approximate[3]) < 0.0:
        axis_y *= -1.0
    half = float(size_m) / 2.0
    xy = np.asarray(
        [
            center - half * axis_x + half * axis_y,
            center + half * axis_x + half * axis_y,
            center + half * axis_x - half * axis_y,
            center - half * axis_x - half * axis_y,
        ]
    )
    return np.c_[xy, np.full(4, float(z_m))]


def fit_vertical_plane(points, settings: dict, *, seed: int = 0) -> dict:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < int(settings["minimum_candidate_points"]):
        raise RuntimeError("too few RGB-D points for incubator plane")
    rng = np.random.default_rng(int(seed))
    best = None
    for _ in range(int(settings["ransac_iterations"])):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        if abs(float(normal[2])) > float(settings["maximum_abs_normal_z"]):
            continue
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) < float(
            settings["ransac_distance_m"]
        )
        count = int(inliers.sum())
        if best is None or count > best[0]:
            best = (count, inliers)
    if best is None or best[0] < int(settings["minimum_inlier_points"]):
        raise RuntimeError("incubator front plane RANSAC failed")
    selected = points[best[1]]
    center = selected.mean(axis=0)
    _, _, vh = np.linalg.svd(selected - center, full_matrices=False)
    normal = vh[-1]
    if normal[0] < 0.0:
        normal *= -1.0
    residuals = (selected - center) @ normal
    return {
        "normal_xyz": normal.tolist(),
        "normal_yaw_deg": float(np.degrees(np.arctan2(normal[1], normal[0]))),
        "centroid_xyz_m": center.tolist(),
        "inlier_count": int(len(selected)),
        "rms_m": float(np.sqrt(np.mean(residuals**2))),
        "inlier_mask": best[1],
    }


def estimate_frame(image_bgr, depth_m, camera_matrix, tag_config, settings, *, seed=0):
    image = np.asarray(image_bgr)
    depth = np.asarray(depth_m, dtype=float)
    detections = detect_tags(
        image,
        settings["tag_family"],
        scales=tuple(settings.get("tag_detection_scales", [1, 2])),
    )
    by_id = {tag.tag_id: tag for tag in detections}
    fixed_id = int(settings["fixed_tag_id"])
    anchor_id = int(settings["incubator_anchor_tag_id"])
    if fixed_id not in by_id or anchor_id not in by_id:
        raise RuntimeError(
            f"need fixed tag {fixed_id} and incubator tag {anchor_id}; "
            f"saw {sorted(by_id)}"
        )
    robot_tag = _robot_tag_square(
        tag_config,
        fixed_id,
        float(settings["fixed_tag_size_m"]),
        float(settings["fixed_tag_support_z_m"]),
    ).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        robot_tag,
        by_id[fixed_id].corners.astype(np.float32),
        np.asarray(camera_matrix, dtype=float),
        np.zeros(5),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("fixed-tag camera-to-robot calibration failed")
    rotation_robot_to_camera = cv2.Rodrigues(rvec)[0]
    height, width = image.shape[:2]
    depth_height, depth_width = depth.shape
    scale_x = depth_width / width
    scale_y = depth_height / height
    depth_k = np.asarray(camera_matrix, dtype=float).copy()
    depth_k[0] *= scale_x
    depth_k[1] *= scale_y
    yy, xx = np.indices(depth.shape)
    z = depth
    camera_points = np.stack(
        [
            (xx - depth_k[0, 2]) * z / depth_k[0, 0],
            (yy - depth_k[1, 2]) * z / depth_k[1, 1],
            z,
        ],
        axis=-1,
    ).reshape(-1, 3)
    robot_points = (camera_points - tvec.reshape(3)) @ rotation_robot_to_camera
    reduced = cv2.resize(
        image, (depth_width, depth_height), interpolation=cv2.INTER_AREA
    )
    hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
    anchor = by_id[anchor_id].center
    x_center = float(anchor[0]) / width
    y_center = float(anchor[1]) / height
    x_fraction = xx / max(depth_width - 1, 1)
    y_fraction = yy / max(depth_height - 1, 1)
    candidate = (
        np.isfinite(z)
        & (z >= float(settings["minimum_depth_m"]))
        & (z <= float(settings["maximum_depth_m"]))
        & (np.abs(x_fraction - x_center) <= float(settings["roi_half_width_fraction"]))
        & (y_fraction >= y_center - float(settings["roi_above_fraction"]))
        & (y_fraction <= y_center + float(settings["roi_below_fraction"]))
        & (hsv[:, :, 1] <= int(settings["maximum_saturation"]))
        & (hsv[:, :, 2] >= int(settings["minimum_value"]))
    )
    flat_indices = np.flatnonzero(candidate.reshape(-1))
    plane = fit_vertical_plane(
        robot_points[flat_indices], settings, seed=seed
    )
    full_inlier = np.zeros(depth.size, dtype=bool)
    full_inlier[flat_indices[np.asarray(plane.pop("inlier_mask"), dtype=bool)]] = True
    plane["depth_inlier_mask"] = full_inlier.reshape(depth.shape)
    plane["detected_tag_ids"] = sorted(by_id)
    return plane


def estimate_bundle(capture_dir: str | Path, tag_config_path: str | Path, settings: dict):
    capture = Path(capture_dir)
    tag_config = json.loads(Path(tag_config_path).read_text())
    raw = capture / "raw" / "head"
    reports = []
    for index, frame in enumerate(sorted(path for path in raw.iterdir() if path.is_dir())):
        image = cv2.rotate(cv2.imread(str(frame / "rgb.png")), cv2.ROTATE_90_CLOCKWISE)
        depth = cv2.rotate(np.load(frame / "depth.npy"), cv2.ROTATE_90_CLOCKWISE)
        meta = json.loads((frame / "meta.json").read_text())
        report = estimate_frame(
            image,
            depth,
            meta["intrinsics"]["K_rgb_rotated_clockwise"],
            tag_config,
            settings,
            seed=index,
        )
        report.pop("depth_inlier_mask")
        report["frame"] = frame.name
        reports.append(report)
    yaws = np.asarray([item["normal_yaw_deg"] for item in reports], dtype=float)
    return {
        "schema": "piper_robot.incubator_door_plane/v1",
        "capture": str(capture.resolve()),
        "frame_count": len(reports),
        "normal_yaw_deg": float(np.median(yaws)),
        "normal_yaw_std_deg": float(np.std(yaws)),
        "normal_yaw_range_deg": float(np.ptp(yaws)),
        "frames": reports,
    }
