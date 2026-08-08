"""Portable appliance-frame trajectories and lab-local pose enrollment.

Portable task data is expressed in a semantic appliance frame.  Camera and
AprilTag placement are installation details: each lab estimates the appliance
pose in the robot frame, and an optional tag is enrolled relative to that pose.
No absolute tag pose or tag id is stored in a portable trajectory.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import json
import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "piper_robot.appliance_frame_enrollment/v1"
TRAJECTORY_SCHEMA = "piper_robot.appliance_relative_trajectory/v1"


def matrix4(value, name: str = "transform") -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(result[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} rotation is reflected or singular")
    return result.copy()


def pose7_to_matrix(value) -> np.ndarray:
    """Convert Mink-style ``[qw,qx,qy,qz,x,y,z]`` to a matrix."""

    pose = np.asarray(value, dtype=float).reshape(7)
    quaternion = pose[:4]
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-9 or not np.all(np.isfinite(pose)):
        raise ValueError("pose must contain a finite nonzero quaternion")
    quaternion /= norm
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(
        [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
    ).as_matrix()
    result[:3, 3] = pose[4:]
    return result


def matrix_to_pose7(value) -> np.ndarray:
    transform = matrix4(value)
    qxyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    quaternion = np.asarray([qxyzw[3], *qxyzw[:3]], dtype=float)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return np.r_[quaternion, transform[:3, 3]]


def _semantic_object(scene: dict, semantic_name: str) -> dict:
    matches = [
        item
        for item in scene.get("objects", ())
        if item.get("semantic_name") == semantic_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"scene must contain exactly one {semantic_name!r}; found {len(matches)}"
        )
    return matches[0]


def appliance_pose_from_scene(
    scene: dict,
    semantic_name: str,
    T_robot_scene,
    *,
    minimum_confidence: float = 0.85,
    require_volume_fit: bool = True,
) -> tuple[np.ndarray, dict]:
    """Complete one SAM-labelled box into a robot-frame appliance pose."""

    transform = matrix4(T_robot_scene, "T_robot_scene")
    item = _semantic_object(scene, semantic_name)
    confidence = float(item.get("confidence", 0.0))
    if confidence < float(minimum_confidence):
        raise ValueError(
            f"{semantic_name} confidence {confidence:.3f} is below "
            f"{float(minimum_confidence):.3f}"
        )
    geometry = item.get("geometry", {})
    if geometry.get("kind") != "box":
        raise ValueError("portable appliance frame currently requires box geometry")
    center = np.asarray(geometry.get("center_xyz_m"), dtype=float)
    size = np.asarray(geometry.get("size_xyz_m"), dtype=float)
    yaw = float(geometry.get("yaw_rad", float("nan")))
    if (
        center.shape != (3,)
        or size.shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(size))
        or np.any(size <= 0.0)
        or not math.isfinite(yaw)
    ):
        raise ValueError("appliance box geometry is incomplete")
    fit = item.get("semantic_volume_fit", {})
    if require_volume_fit and not bool(fit.get("accepted", False)):
        raise ValueError("semantic appliance volume fit is not accepted")
    T_scene_appliance = np.eye(4)
    T_scene_appliance[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T_scene_appliance[:3, 3] = center
    result = transform @ T_scene_appliance
    return result, {
        "semantic_name": semantic_name,
        "instance_id": item.get("instance_id"),
        "confidence": confidence,
        "size_xyz_m": size.tolist(),
        "semantic_volume_fit_accepted": bool(fit.get("accepted", False)),
        "T_robot_appliance": result.tolist(),
    }


def load_accepted_robot_scene_transform(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load an independently accepted scene-to-robot calibration.

    The explicit scene transform avoids pretending that a gravity-levelled
    Record3D reconstruction is identical to its camera frame.
    """

    payload = json.loads(Path(path).read_text())
    if not bool(payload.get("accepted", False)):
        raise ValueError("scene-to-robot calibration is not accepted")
    if payload.get("transform_convention") not in (
        None,
        "p_robot = T_robot_scene @ p_scene",
    ):
        raise ValueError("unsupported scene-to-robot transform convention")
    return matrix4(payload.get("T_robot_scene"), "T_robot_scene"), payload


def registration_between_appliance_frames(
    T_robot_appliance_reference, T_robot_appliance_live
) -> np.ndarray:
    reference = matrix4(T_robot_appliance_reference, "reference appliance frame")
    live = matrix4(T_robot_appliance_live, "live appliance frame")
    return live @ np.linalg.inv(reference)


def registration_gate(
    T_robot_appliance_reference,
    T_robot_appliance_live,
    *,
    maximum_translation_m: float = 0.35,
    maximum_yaw_deg: float = 45.0,
    maximum_tilt_deg: float = 8.0,
) -> dict:
    registration = registration_between_appliance_frames(
        T_robot_appliance_reference, T_robot_appliance_live
    )
    rotation = Rotation.from_matrix(registration[:3, :3])
    yaw, pitch, roll = rotation.as_euler("zyx", degrees=True)
    translation = float(np.linalg.norm(registration[:3, 3]))
    accepted = bool(
        translation <= float(maximum_translation_m)
        and abs(float(yaw)) <= float(maximum_yaw_deg)
        and max(abs(float(pitch)), abs(float(roll))) <= float(maximum_tilt_deg)
    )
    return {
        "accepted": accepted,
        "translation_m": translation,
        "yaw_deg": float(yaw),
        "pitch_deg": float(pitch),
        "roll_deg": float(roll),
        "limits": {
            "maximum_translation_m": float(maximum_translation_m),
            "maximum_yaw_deg": float(maximum_yaw_deg),
            "maximum_tilt_deg": float(maximum_tilt_deg),
        },
        "T_registration": registration.tolist(),
        "registration_wxyz_xyz": matrix_to_pose7(registration).tolist(),
    }


def trajectory_to_appliance_frame(
    trajectory: Sequence[dict], T_robot_appliance
) -> dict:
    """Remove the reference lab pose from an absolute EE trajectory."""

    appliance_from_robot = np.linalg.inv(
        matrix4(T_robot_appliance, "T_robot_appliance")
    )
    samples = []
    for sample in trajectory:
        relative = appliance_from_robot @ pose7_to_matrix(sample["pose_wxyz_xyz"])
        samples.append({**sample, "pose_wxyz_xyz": matrix_to_pose7(relative).tolist()})
    return {
        "schema": TRAJECTORY_SCHEMA,
        "frame": "appliance",
        "samples": samples,
    }


def retarget_appliance_trajectory(
    portable: dict | Sequence[dict], T_robot_appliance
) -> list[dict]:
    """Express appliance-relative samples in the current robot frame."""

    if isinstance(portable, dict):
        if portable.get("schema") != TRAJECTORY_SCHEMA:
            raise ValueError("unsupported portable trajectory schema")
        samples = portable["samples"]
    else:
        samples = portable
    robot_from_appliance = matrix4(T_robot_appliance, "T_robot_appliance")
    return [
        {
            **sample,
            "pose_wxyz_xyz": matrix_to_pose7(
                robot_from_appliance @ pose7_to_matrix(sample["pose_wxyz_xyz"])
            ).tolist(),
        }
        for sample in samples
    ]


def enroll_local_tag(
    T_robot_appliance,
    T_robot_tag,
    *,
    tag_id: int,
    appliance_semantic_name: str,
) -> dict:
    """Store arbitrary lab-local tag placement relative to the appliance."""

    appliance = matrix4(T_robot_appliance, "T_robot_appliance")
    tag = matrix4(T_robot_tag, "T_robot_tag")
    return {
        "schema": SCHEMA,
        "accepted": True,
        "appliance_semantic_name": appliance_semantic_name,
        "local_tag": {
            "id": int(tag_id),
            "T_appliance_tag": (np.linalg.inv(appliance) @ tag).tolist(),
            "placement_is_lab_local": True,
        },
        "T_robot_appliance_at_enrollment": appliance.tolist(),
    }


def appliance_pose_from_local_tag(T_robot_tag, enrollment: dict) -> np.ndarray:
    """Track an appliance without assuming a common tag pose across labs."""

    if enrollment.get("schema") != SCHEMA or not enrollment.get("accepted"):
        raise ValueError("appliance enrollment is not accepted")
    T_appliance_tag = matrix4(
        enrollment["local_tag"]["T_appliance_tag"], "T_appliance_tag"
    )
    return matrix4(T_robot_tag, "T_robot_tag") @ np.linalg.inv(T_appliance_tag)
