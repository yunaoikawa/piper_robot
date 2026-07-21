"""AprilTag-based planar object localization and replay retargeting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


APRILTAG_FAMILIES = (
    "DICT_APRILTAG_36h11",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_16h5",
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_ARUCO_ORIGINAL",
)


@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    corners: np.ndarray
    family: str

    @property
    def center(self):
        return self.corners.mean(axis=0)

    @property
    def perimeter(self):
        return float(cv2.arcLength(self.corners.astype(np.float32), True))

    @property
    def angle(self):
        edge = self.corners[1] - self.corners[0]
        return float(np.arctan2(edge[1], edge[0]))


def detect_tags(image_bgr: np.ndarray, family: str | None = None,
                scales=(1, 2, 4)) -> list[TagDetection]:
    """Detect the best-supported AprilTag family, or one explicitly selected."""
    families = (family,) if family else APRILTAG_FAMILIES
    best = []
    for name in families:
        dictionary_id = getattr(cv2.aruco, name)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.minMarkerPerimeterRate = 0.005
        parameters.maxMarkerPerimeterRate = 6.0
        # More tolerant than the default without accepting the many random IDs
        # observed at 1.0 on blurred Record3D frames.
        parameters.errorCorrectionRate = 0.7
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 53
        parameters.adaptiveThreshWinSizeStep = 4
        parameters.minCornerDistanceRate = 0.01
        # The laminated lid tag often appears locally contrast-inverted under
        # the microscope light. OpenCV can test both polarities cheaply.
        parameters.detectInvertedMarker = True
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary_id), parameters)
        by_id = {}
        # 4x is necessary for the 30 mm tag (~25 px in the head stream).
        for scale in scales:
            scaled = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_CUBIC) if scale > 1 else image_bgr
            gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            for box, tag_id in zip(corners, ids.ravel()):
                candidate = TagDetection(
                    int(tag_id), np.asarray(box, dtype=float).reshape(4, 2) / scale, name)
                previous = by_id.get(candidate.tag_id)
                if previous is None or candidate.perimeter > previous.perimeter:
                    by_id[candidate.tag_id] = candidate
        found = list(by_id.values())
        if len(found) > len(best):
            best = found
    return sorted(best, key=lambda tag: tag.tag_id)


def render_tags(image_bgr: np.ndarray, detections, roles=None) -> np.ndarray:
    out = image_bgr.copy()
    roles = roles or {}
    for tag in detections:
        pts = np.rint(tag.corners).astype(int)
        role = roles.get(tag.tag_id, "unassigned")
        color = (0, 255, 0) if role == "fixed" else (0, 255, 255) if role == "lid" else (255, 0, 255)
        cv2.polylines(out, [pts], True, color, 3)
        center = tuple(np.rint(tag.center).astype(int))
        cv2.putText(out, f"id={tag.tag_id} {role}", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return out


def estimate_tag_camera_pose(tag, camera_matrix, size_m, distortion=None):
    """Estimate one square tag pose and return (rvec, tvec, RMS pixels)."""
    half = float(size_m) / 2.0
    object_points = np.float32([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0],
    ])
    distortion = (np.zeros(5, dtype=float) if distortion is None
                  else np.asarray(distortion, dtype=float))
    ok, rvec, tvec = cv2.solvePnP(
        object_points, np.float32(tag.corners),
        np.asarray(camera_matrix, dtype=float), distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not np.all(np.isfinite(tvec)) or float(tvec[2]) <= 0:
        raise ValueError(f"PnP failed for tag {tag.tag_id}")
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, np.asarray(camera_matrix, dtype=float), distortion)
    residual = projected.reshape(4, 2) - tag.corners
    rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    return rvec.reshape(3), tvec.reshape(3), rms


def classify_roles(detections, lid_id=None):
    """Classify a known lid tag; size inference is discovery-only."""
    if not detections:
        raise ValueError("no AprilTags detected")
    ids = [tag.tag_id for tag in detections]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate tag IDs: {ids}")
    if lid_id is None:
        # The lid tag is 30 mm and fixed tags are 60 mm. In one head frame the
        # smallest perimeter is the only safe automatic size cue; persist the
        # resulting ID so this heuristic is never repeated during execution.
        max_y = max(float(tag.center[1]) for tag in detections)
        lower = [tag for tag in detections if tag.center[1] >= 0.45 * max_y]
        lid_id = min(lower or detections, key=lambda tag: tag.perimeter).tag_id
    if lid_id not in ids:
        raise ValueError(f"lid tag ID {lid_id} not detected; saw {ids}")
    lid = next(tag for tag in detections if tag.tag_id == lid_id)
    roles = {}
    for tag in detections:
        if tag.tag_id == lid_id:
            roles[tag.tag_id] = "lid"
        elif tag.perimeter >= 1.25 * lid.perimeter:
            roles[tag.tag_id] = "fixed"
        else:
            roles[tag.tag_id] = "ignored"
    return roles


def fit_image_to_robot(detections, fixed_robot_xy):
    by_id = {tag.tag_id: tag for tag in detections}
    ids = sorted(set(by_id) & {int(k) for k in fixed_robot_xy})
    if len(ids) < 3:
        raise ValueError(f"need >=3 calibrated fixed tags, found {ids}")
    pixels = np.float32([by_id[tag_id].center for tag_id in ids])
    robot = np.float32([fixed_robot_xy[str(tag_id)] for tag_id in ids])
    if len(ids) == 3:
        affine = cv2.getAffineTransform(pixels, robot)
    else:
        affine, inliers = cv2.estimateAffine2D(pixels, robot, method=cv2.RANSAC,
                                               ransacReprojThreshold=0.003)
        if affine is None or int(inliers.sum()) < 3:
            raise ValueError("fixed-tag robot-plane fit failed")
    return np.vstack([affine, [0.0, 0.0, 1.0]])


def fit_image_to_plane(detections, fixed_plane_corners):
    """Fit pixels to an arbitrary metric plane from calibrated tag corners."""
    by_id = {tag.tag_id: tag for tag in detections}
    ids = sorted(set(by_id) & {int(key) for key in fixed_plane_corners})
    if not ids:
        raise ValueError("no calibrated fixed tag is visible")
    pixels = np.float32(np.vstack([by_id[tag_id].corners for tag_id in ids]))
    plane = np.float32(np.vstack([fixed_plane_corners[str(tag_id)] for tag_id in ids]))
    transform, inliers = cv2.findHomography(
        pixels, plane, method=cv2.RANSAC, ransacReprojThreshold=0.003)
    if transform is None or int(inliers.sum()) < 4:
        raise ValueError(f"fixed-tag plane fit failed for IDs {ids}")
    return transform


def map_points(transform, points):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    hom = np.c_[points, np.ones(len(points))]
    mapped = hom @ np.asarray(transform, dtype=float).T
    return mapped[:, :2] / mapped[:, 2:3]


def detect_blue_cross(image_bgr, image_to_plane, plane_to_robot_xy,
                      reference_robot_xy, config):
    """Find the blue lid cross nearest its bounded teacher-plane position."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low = np.asarray(config.get("hsv_low", [100, 80, 50]), dtype=np.uint8)
    high = np.asarray(config.get("hsv_high", [125, 255, 255]), dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    min_area = int(config.get("min_area", 30))
    max_area = int(config.get("max_area", 500))
    reference = np.asarray(reference_robot_xy, dtype=float)
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        plane = map_points(image_to_plane, [centers[label]])[0]
        robot_xy = np.asarray(plane_to_robot_xy, dtype=float) @ plane
        distance = float(np.linalg.norm(robot_xy - reference))
        candidates.append((distance, -area, centers[label], robot_xy))
    if not candidates:
        raise ValueError("blue lid cross not detected")
    best = min(candidates, key=lambda item: (item[0], item[1]))
    if best[0] > float(config.get("max_distance_m", 0.12)):
        raise ValueError(f"blue lid cross is {best[0]*1000:.1f}mm from teacher bound")
    return {"center": np.asarray(best[2]), "robot_xy": np.asarray(best[3]),
            "distance_m": best[0], "mask": mask}


def lid_pose_robot(detections, lid_id, image_to_robot, plane_to_robot_xy=None):
    tag = next((tag for tag in detections if tag.tag_id == lid_id), None)
    if tag is None:
        raise ValueError(f"lid tag {lid_id} not detected")
    mapped = map_points(image_to_robot, tag.corners)
    if plane_to_robot_xy is not None:
        mapping = np.asarray(plane_to_robot_xy, dtype=float).reshape(2, 2)
        mapped = mapped @ mapping.T
    center = mapped.mean(axis=0)
    edge = mapped[1] - mapped[0]
    yaw = float(np.arctan2(edge[1], edge[0]))
    return np.array([center[0], center[1], yaw])


def wrap_angle(angle):
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def object_delta(current_pose, reference_pose):
    current = np.asarray(current_pose, dtype=float)
    reference = np.asarray(reference_pose, dtype=float)
    return np.array([current[0] - reference[0], current[1] - reference[1],
                     wrap_angle(current[2] - reference[2])])


def smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def retarget_weight(frame, phases):
    ramp_end = int(phases.get("approach_ramp_end", 60))
    hold_end = int(phases.get("retarget_hold_end", 140))
    blend_end = int(phases.get("retarget_blend_end", 190))
    if frame < ramp_end:
        return smoothstep(frame / max(ramp_end, 1))
    if frame <= hold_end:
        return 1.0
    if frame < blend_end:
        return 1.0 - smoothstep((frame - hold_end) / max(blend_end - hold_end, 1))
    return 0.0


def retarget_pose(wxyz_xyz, delta_se2, weight, pivot_xy):
    pose = np.asarray(wxyz_xyz, dtype=float).copy()
    delta = np.asarray(delta_se2, dtype=float) * float(weight)
    yaw = delta[2]
    c, s = np.cos(yaw), np.sin(yaw)
    rot2 = np.array([[c, -s], [s, c]])
    pose[4:6] = np.asarray(pivot_xy) + rot2 @ (pose[4:6] - np.asarray(pivot_xy)) + delta[:2]
    q_xyzw = np.r_[pose[1:4], pose[0]]
    rz = Rotation.from_euler("z", yaw)
    rotated = rz * Rotation.from_quat(q_xyzw)
    q = rotated.as_quat()
    pose[:4] = [q[3], q[0], q[1], q[2]]
    return pose


@dataclass
class TagProfile:
    family: str
    lid_id: int
    fixed_robot_xy: dict
    fixed_plane_corners: dict
    plane_to_robot_xy: np.ndarray | None
    reference_lid_pose: np.ndarray
    reference_robot_pivot_xy: np.ndarray
    reference_wrist_corners: np.ndarray | None
    phases: dict
    lid_tracker: dict | None
    max_translation_m: float = 0.10
    max_yaw_deg: float = 20.0

    @classmethod
    def load(cls, path):
        cfg = json.loads(Path(path).read_text())
        sizes = cfg.get("tag_sizes_m", {})
        if float(sizes.get("lid", 0.0)) != 0.03 or float(sizes.get("fixed", 0.0)) != 0.06:
            raise ValueError("tag profile must declare lid=0.03m and fixed=0.06m")
        reference = cfg.get("reference_lid_pose")
        if reference is None:
            raise ValueError("tag profile is not calibrated: missing reference_lid_pose")
        fixed = cfg.get("fixed_robot_xy", {})
        plane = cfg.get("fixed_plane_corners", {})
        plane_to_robot = cfg.get("plane_to_robot_xy")
        if len(fixed) < 3 and (not plane or plane_to_robot is None):
            raise ValueError("tag profile is not calibrated: missing fixed-plane mapping")
        corners = cfg.get("reference_wrist_corners")
        return cls(
            family=cfg["family"], lid_id=int(cfg["lid_id"]),
            fixed_robot_xy=fixed,
            fixed_plane_corners=plane,
            plane_to_robot_xy=(None if plane_to_robot is None
                               else np.asarray(plane_to_robot, dtype=float).reshape(2, 2)),
            reference_lid_pose=np.asarray(reference, dtype=float),
            reference_robot_pivot_xy=np.asarray(
                cfg.get("reference_robot_pivot_xy", reference[:2]), dtype=float),
            reference_wrist_corners=None if corners is None else np.asarray(corners, dtype=float).reshape(4, 2),
            phases=cfg.get("phases", {}),
            lid_tracker=cfg.get("lid_tracker"),
            max_translation_m=float(cfg.get("max_translation_m", 0.10)),
            max_yaw_deg=float(cfg.get("max_yaw_deg", 20.0)),
        )

    def fit_image_transform(self, detections):
        if self.fixed_plane_corners:
            return fit_image_to_plane(detections, self.fixed_plane_corners)
        return fit_image_to_robot(detections, self.fixed_robot_xy)

    def locate_lid(self, image_bgr, detections, transform):
        if self.lid_tracker and self.lid_tracker.get("type") == "blue_cross":
            result = detect_blue_cross(
                image_bgr, transform, self.plane_to_robot_xy,
                self.reference_lid_pose[:2], self.lid_tracker)
            return np.array([*result["robot_xy"], self.reference_lid_pose[2]]), result
        return (lid_pose_robot(detections, self.lid_id, transform,
                               self.plane_to_robot_xy), None)

    def validate_delta(self, delta):
        if np.linalg.norm(delta[:2]) > self.max_translation_m:
            raise ValueError(f"lid translation {np.linalg.norm(delta[:2])*1000:.1f}mm exceeds limit")
        if abs(np.degrees(delta[2])) > self.max_yaw_deg:
            raise ValueError(f"lid yaw {np.degrees(delta[2]):.1f}deg exceeds limit")


def servo_error(corners, reference_corners):
    corners = np.asarray(corners, dtype=float).reshape(4, 2)
    reference = np.asarray(reference_corners, dtype=float).reshape(4, 2)
    center_error = reference.mean(axis=0) - corners.mean(axis=0)
    angle = np.arctan2(*(corners[1] - corners[0])[::-1])
    reference_angle = np.arctan2(*(reference[1] - reference[0])[::-1])
    return np.array([center_error[0], center_error[1], wrap_angle(reference_angle - angle)])


def servo_step(jacobian, error, max_xy_m=0.002, max_yaw_rad=np.deg2rad(2.0), damping=1e-3):
    jacobian = np.asarray(jacobian, dtype=float).reshape(3, 3)
    error = np.asarray(error, dtype=float).reshape(3)
    lhs = jacobian.T @ jacobian + damping * np.eye(3)
    step = np.linalg.solve(lhs, jacobian.T @ error)
    step[:2] = np.clip(step[:2], -max_xy_m, max_xy_m)
    step[2] = np.clip(step[2], -max_yaw_rad, max_yaw_rad)
    return step
