"""Offline wrist RGB-D target localization and hand-eye fitting.

This module is deliberately observation-only.  It reads completed capture
bundles, their synchronized read-only robot states, and a semantic marker
profile.  It never imports a robot client or sends a command.

The intended transparent-object policy is:

* SAM supplies the semantic object identity in the reconstructed scene.
* A small opaque marker supplies repeatable metric pixels on the otherwise
  unreliable transparent surface.
* The largest tool-coloured component is treated as the gripper reference.
* The target is the cross-shaped component nearest that reference, using
  dimensionless image fractions rather than fixed pixel coordinates.
* Multiple stopped poses jointly fit the camera-to-EE transform and one
  static target point per capture episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable
import warnings

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


SCHEMA = "piper_robot.wrist_rgbd_target_calibration/v1"
OBJECT_SCENE_SCHEMA = "piper_robot.dynamic_dish_lid_scene/v2"


@dataclass(frozen=True)
class TargetObservation:
    capture: str
    session_id: str
    episode: str
    split: str
    center_px: tuple[float, float]
    point_camera_m: tuple[float, float, float]
    ee_quaternion_wxyz: tuple[float, float, float, float]
    ee_translation_robot_m: tuple[float, float, float]
    right_q_physical_rad: tuple[float, float, float, float, float, float]
    candidate_area_fraction: float
    candidate_cross_score: float
    normalized_tool_distance: float
    depth_samples: int

    @property
    def camera_point(self) -> np.ndarray:
        return np.asarray(self.point_camera_m, dtype=float)

    @property
    def robot_from_ee_rotation(self) -> np.ndarray:
        quaternion = np.asarray(self.ee_quaternion_wxyz, dtype=float)
        return Rotation.from_quat(
            np.r_[quaternion[1:], quaternion[0]]
        ).as_matrix()

    @property
    def robot_from_ee_translation(self) -> np.ndarray:
        return np.asarray(self.ee_translation_robot_m, dtype=float)


def _read_frame_index(capture: Path) -> list[dict]:
    records = [
        json.loads(line)
        for line in (capture / "frames.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"capture has no frame records: {capture}")
    return records


def _validate_capture_manifest(capture: Path) -> dict:
    manifest = json.loads((capture / "manifest.json").read_text())
    robot_state = manifest.get("robot_state", {})
    if robot_state.get("commands_sent") is not False:
        raise ValueError(f"capture lacks commands_sent=false: {capture}")
    if robot_state.get("stability", {}).get("stationary") is not True:
        raise ValueError(f"capture was not stationary: {capture}")
    if manifest.get("status") != "complete":
        raise ValueError(f"capture is incomplete: {capture}")
    return manifest


def median_rgbd(capture: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median BGR, depth, and raw-RGB camera matrix for one burst."""

    capture = Path(capture)
    records = _read_frame_index(capture)
    images = []
    depths = []
    for record in records:
        files = record["files"]
        image = cv2.imread(str(capture / files["rgb_png"]["path"]))
        if image is None:
            raise FileNotFoundError(capture / files["rgb_png"]["path"])
        depth = np.load(capture / files["depth_npy"]["path"]).astype(float)
        if image.shape[:2] != depth.shape:
            raise ValueError(
                f"unaligned RGB/depth shapes in {capture}: "
                f"{image.shape[:2]} != {depth.shape}"
            )
        images.append(image)
        depths.append(depth)
    rgb = np.median(np.stack(images), axis=0).astype(np.uint8)
    stack = np.stack(depths)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        depth = np.nanmedian(np.where(stack > 0, stack, np.nan), axis=0)
    camera_matrix = np.asarray(
        records[0]["intrinsics"]["K_raw_rgb"],
        dtype=float,
    )
    return rgb, depth, camera_matrix


def _cross_score(
    labels: np.ndarray,
    label: int,
    stats: np.ndarray,
    center: np.ndarray,
) -> tuple[float, float, float]:
    x, y, width, height, area = (
        int(value) for value in stats[label]
    )
    patch = labels[y:y + height, x:x + width] == label
    center_x = int(np.clip(round(center[0] - x), 0, width - 1))
    center_y = int(np.clip(round(center[1] - y), 0, height - 1))
    band_x = max(1, int(round(width * 0.20)))
    band_y = max(1, int(round(height * 0.20)))
    horizontal = patch[
        max(0, center_y - band_y):
        min(height, center_y + band_y + 1),
        :,
    ]
    vertical = patch[
        :,
        max(0, center_x - band_x):
        min(width, center_x + band_x + 1),
    ]
    horizontal_span = (
        float(np.count_nonzero(np.any(horizontal, axis=0))) / width
    )
    vertical_span = (
        float(np.count_nonzero(np.any(vertical, axis=1))) / height
    )
    return (
        min(horizontal_span, vertical_span),
        width / max(height, 1),
        area / max(width * height, 1),
    )


def detect_tool_relative_blue_cross(
    image_bgr: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    profile: dict,
) -> tuple[dict, np.ndarray]:
    """Detect a target marker relative to the visible tool, without an ROI."""

    image = np.asarray(image_bgr)
    depth = np.asarray(depth_m, dtype=float)
    if image.shape[:2] != depth.shape:
        raise ValueError("RGB and depth must share one sensor coordinate system")
    height, width = depth.shape
    image_area = height * width
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = np.asarray(profile.get("hsv_low", [90, 55, 40]), dtype=np.uint8)
    high = np.asarray(
        profile.get("hsv_high", [135, 255, 255]),
        dtype=np.uint8,
    )
    mask = cv2.inRange(hsv, low, high)
    kernel_width = max(
        1,
        int(round(min(height, width) * float(
            profile.get("morphology_fraction", 0.005)
        ))),
    )
    if kernel_width % 2 == 0:
        kernel_width += 1
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((kernel_width, kernel_width), dtype=np.uint8),
    )
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    if count < 3:
        raise ValueError("tool and target marker were not both detected")
    tool_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    tool_fraction = (
        float(stats[tool_label, cv2.CC_STAT_AREA]) / image_area
    )
    if tool_fraction < float(profile.get("minimum_tool_area_fraction", 0.02)):
        raise ValueError("largest blue component is too small to be the tool")
    distance_from_tool = cv2.distanceTransform(
        (labels != tool_label).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    diagonal = float(np.hypot(height, width))
    minimum_area_fraction = float(
        profile.get("minimum_marker_area_fraction", 0.00008)
    )
    maximum_area_fraction = float(
        profile.get("maximum_marker_area_fraction", 0.015)
    )
    minimum_cross_score = float(profile.get("minimum_cross_score", 0.72))
    minimum_depth_samples = int(
        np.ceil(
            image_area
            * float(profile.get("minimum_depth_sample_fraction", 0.00006))
        )
    )
    candidates = []
    for label in range(1, count):
        if label == tool_label:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_fraction = area / image_area
        if not minimum_area_fraction <= area_fraction <= maximum_area_fraction:
            continue
        cross_score, aspect, fill = _cross_score(
            labels,
            label,
            stats,
            centers[label],
        )
        if not (
            float(profile.get("minimum_aspect", 0.30))
            <= aspect
            <= float(profile.get("maximum_aspect", 1.80))
        ):
            continue
        if not (
            float(profile.get("minimum_fill", 0.20))
            <= fill
            <= float(profile.get("maximum_fill", 0.90))
        ):
            continue
        if cross_score < minimum_cross_score:
            continue
        selected = labels == label
        rows, columns = np.nonzero(selected)
        values = depth[rows, columns]
        valid = np.isfinite(values) & (values > 0)
        if int(np.count_nonzero(valid)) < minimum_depth_samples:
            continue
        rows = rows[valid]
        columns = columns[valid]
        values = values[valid]
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        points = np.c_[
            (columns - cx) * values / fx,
            (rows - cy) * values / fy,
            values,
        ]
        point = np.median(points, axis=0)
        normalized_distance = (
            float(np.min(distance_from_tool[selected])) / diagonal
        )
        candidates.append(
            {
                "label": label,
                "center_px": centers[label].astype(float),
                "point_camera_m": point,
                "area_fraction": area_fraction,
                "cross_score": cross_score,
                "aspect": aspect,
                "fill": fill,
                "normalized_tool_distance": normalized_distance,
                "depth_samples": int(len(points)),
            }
        )
    if not candidates:
        raise ValueError("no depth-valid cross-shaped target candidate")
    selected = min(
        candidates,
        key=lambda item: (
            item["normalized_tool_distance"],
            -item["cross_score"],
            -item["depth_samples"],
        ),
    )
    maximum_distance = float(
        profile.get("maximum_normalized_tool_distance", 0.20)
    )
    if selected["normalized_tool_distance"] > maximum_distance:
        raise ValueError(
            "nearest target marker is too far from the visible tool: "
            f"{selected['normalized_tool_distance']:.4f}"
        )

    overlay = image.copy()
    tool_mask = labels == tool_label
    overlay[tool_mask] = (
        0.65 * overlay[tool_mask]
        + 0.35 * np.array([255, 255, 0])
    ).astype(np.uint8)
    target_mask = labels == selected["label"]
    overlay[target_mask] = (
        0.35 * overlay[target_mask]
        + 0.65 * np.array([0, 255, 255])
    ).astype(np.uint8)
    center = tuple(np.rint(selected["center_px"]).astype(int))
    marker_size = max(12, int(round(min(height, width) * 0.04)))
    cv2.drawMarker(
        overlay,
        center,
        (0, 0, 255),
        cv2.MARKER_CROSS,
        marker_size,
        max(2, marker_size // 10),
    )
    return selected, overlay


def observe_capture(
    capture: str | Path,
    *,
    episode: str,
    split: str,
    marker_profile: dict,
    overlay_path: str | Path | None = None,
) -> TargetObservation:
    capture = Path(capture).resolve()
    manifest = _validate_capture_manifest(capture)
    image, depth, camera_matrix = median_rgbd(capture)
    selected, overlay = detect_tool_relative_blue_cross(
        image,
        depth,
        camera_matrix,
        marker_profile,
    )
    if overlay_path is not None:
        overlay_path = Path(overlay_path)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), overlay)
    state = manifest["robot_state"]["after"]["right_ee_pose"]
    return TargetObservation(
        capture=str(capture),
        session_id=str(manifest["session_id"]),
        episode=str(episode),
        split=str(split),
        center_px=tuple(float(value) for value in selected["center_px"]),
        point_camera_m=tuple(
            float(value) for value in selected["point_camera_m"]
        ),
        ee_quaternion_wxyz=tuple(
            float(value) for value in state["quaternion_wxyz"]
        ),
        ee_translation_robot_m=tuple(
            float(value) for value in state["translation_xyz_m"]
        ),
        right_q_physical_rad=tuple(
            float(value)
            for value in manifest["robot_state"]["after"][
                "right_joint_positions_rad"
            ]
        ),
        candidate_area_fraction=float(selected["area_fraction"]),
        candidate_cross_score=float(selected["cross_score"]),
        normalized_tool_distance=float(
            selected["normalized_tool_distance"]
        ),
        depth_samples=int(selected["depth_samples"]),
    )


def _unpack_fit(
    values: np.ndarray,
    episodes: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rotation = Rotation.from_rotvec(values[:3]).as_matrix()
    translation = values[3:6]
    targets = {
        episode: values[6 + 3 * index:9 + 3 * index]
        for index, episode in enumerate(episodes)
    }
    return rotation, translation, targets


def _world_point(
    observation: TargetObservation,
    ee_from_camera_rotation: np.ndarray,
    ee_from_camera_translation: np.ndarray,
) -> np.ndarray:
    return (
        observation.robot_from_ee_rotation
        @ (
            ee_from_camera_rotation @ observation.camera_point
            + ee_from_camera_translation
        )
        + observation.robot_from_ee_translation
    )


def fit_static_target_hand_eye(
    observations: Iterable[TargetObservation],
    profile: dict,
) -> dict:
    """Fit one rigid camera mount and one target point per static episode."""

    observations = list(observations)
    train = [item for item in observations if item.split == "fit"]
    holdout = [item for item in observations if item.split == "holdout"]
    episodes = sorted({item.episode for item in train})
    if not episodes:
        raise ValueError("at least one fit episode is required")
    parameter_count = 6 + 3 * len(episodes)
    if 3 * len(train) < parameter_count:
        raise ValueError(
            f"insufficient observations: {len(train)} for "
            f"{parameter_count} parameters"
        )
    for episode in episodes:
        if sum(item.episode == episode for item in train) < 2:
            raise ValueError(
                f"episode {episode} needs at least two stopped poses"
            )
    residual_scale_m = float(profile.get("residual_scale_m", 0.005))
    maximum_mount_translation_m = float(
        profile.get("maximum_mount_translation_m", 0.35)
    )

    def residual(values: np.ndarray) -> np.ndarray:
        rotation, translation, targets = _unpack_fit(values, episodes)
        result = []
        for item in train:
            result.extend(
                (
                    _world_point(item, rotation, translation)
                    - targets[item.episode]
                )
                / residual_scale_m
            )
        excess = max(
            0.0,
            float(np.linalg.norm(translation))
            - maximum_mount_translation_m,
        )
        result.append(excess / residual_scale_m)
        return np.asarray(result)

    seed = int(profile.get("optimization_seed", 7))
    starts = int(profile.get("optimization_starts", 32))
    generator = np.random.default_rng(seed)
    best = None
    for index in range(starts):
        rotation_vector = (
            np.zeros(3)
            if index == 0
            else Rotation.random(random_state=generator).as_rotvec()
        )
        rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
        translation = np.zeros(3)
        targets = []
        for episode in episodes:
            points = [
                _world_point(item, rotation, translation)
                for item in train
                if item.episode == episode
            ]
            targets.extend(np.mean(points, axis=0))
        initial = np.r_[rotation_vector, translation, targets]
        result = least_squares(
            residual,
            initial,
            loss="soft_l1",
            max_nfev=int(profile.get("maximum_function_evaluations", 5000)),
        )
        score = float(np.mean(residual(result.x) ** 2))
        if best is None or score < best[0]:
            best = (score, result)
    assert best is not None
    result = best[1]
    rotation, translation, targets = _unpack_fit(result.x, episodes)
    per_observation = []
    train_errors = []
    holdout_errors = []
    for item in observations:
        if item.episode not in targets:
            continue
        point = _world_point(item, rotation, translation)
        error = float(np.linalg.norm(point - targets[item.episode]))
        record = {
            "session_id": item.session_id,
            "episode": item.episode,
            "split": item.split,
            "target_robot_xyz_m": point.tolist(),
            "residual_m": error,
        }
        per_observation.append(record)
        (train_errors if item.split == "fit" else holdout_errors).append(error)
    singular_values = np.linalg.svd(result.jac, compute_uv=False)
    jacobian_rank = int(np.linalg.matrix_rank(result.jac))
    data_parameter_count = len(result.x)
    maximum_train = max(train_errors, default=float("inf"))
    maximum_holdout = max(holdout_errors, default=0.0)
    rms_train = float(
        np.sqrt(
            np.mean(
                [
                    value * value
                    for item in train
                    for value in (
                        _world_point(item, rotation, translation)
                        - targets[item.episode]
                    )
                ]
            )
        )
    )
    accepted = bool(
        jacobian_rank == data_parameter_count
        and rms_train <= float(profile.get("maximum_train_rms_m", 0.008))
        and maximum_train
        <= float(profile.get("maximum_train_point_error_m", 0.018))
        and maximum_holdout
        <= float(profile.get("maximum_holdout_point_error_m", 0.008))
    )
    reasons = []
    if jacobian_rank != data_parameter_count:
        reasons.append("hand_eye_jacobian_rank_deficient")
    if rms_train > float(profile.get("maximum_train_rms_m", 0.008)):
        reasons.append("train_rms_too_large")
    if maximum_train > float(
        profile.get("maximum_train_point_error_m", 0.018)
    ):
        reasons.append("train_point_error_too_large")
    if maximum_holdout > float(
        profile.get("maximum_holdout_point_error_m", 0.008)
    ):
        reasons.append("holdout_point_error_too_large")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return {
        "accepted": accepted,
        "reasons": reasons,
        "ee_from_camera": transform.tolist(),
        "target_robot_xyz_m_by_episode": {
            key: value.tolist() for key, value in targets.items()
        },
        "train_rms_m": rms_train,
        "maximum_train_point_error_m": maximum_train,
        "maximum_holdout_point_error_m": maximum_holdout,
        "jacobian_rank": jacobian_rank,
        "parameter_count": data_parameter_count,
        "jacobian_singular_values": singular_values.tolist(),
        "observations": per_observation,
    }


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _cad_poses(
    observations: list[TargetObservation],
    model_path: str | Path,
    model_branch: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
    data = mujoco.MjData(model)
    joint_ids = [
        int(model.joint(f"{model_branch}/joint{index}").qposadr[0])
        for index in range(1, 7)
    ]
    base_id = int(model.body(f"{model_branch}/base_link").id)
    site_id = int(model.site(f"{model_branch}/ee").id)
    base_from_ee = []
    scene_from_base = None
    for item in observations:
        data.qpos[joint_ids] = np.asarray(
            item.right_q_physical_rad,
            dtype=float,
        )
        mujoco.mj_forward(model, data)
        current_scene_from_base = _transform(
            data.xmat[base_id].reshape(3, 3),
            data.xpos[base_id],
        )
        scene_from_ee = _transform(
            data.site_xmat[site_id].reshape(3, 3),
            data.site_xpos[site_id],
        )
        base_from_ee.append(
            np.linalg.inv(current_scene_from_base) @ scene_from_ee
        )
        if scene_from_base is None:
            scene_from_base = current_scene_from_base
    assert scene_from_base is not None
    return base_from_ee, scene_from_base


def fit_controller_cad_bridge(
    observations: Iterable[TargetObservation],
    *,
    model_path: str | Path,
    model_branch: str,
    profile: dict,
) -> dict:
    """Fit controller-from-CAD-base and the constant EE site convention."""

    observations = list(observations)
    if len(observations) < 3:
        raise ValueError("kinematic bridge needs at least three stopped poses")
    base_from_ee, scene_from_base = _cad_poses(
        observations,
        model_path,
        model_branch,
    )
    controller_from_ee = []
    for item in observations:
        quaternion = np.asarray(item.ee_quaternion_wxyz, dtype=float)
        controller_from_ee.append(
            _transform(
                Rotation.from_quat(
                    np.r_[quaternion[1:], quaternion[0]]
                ).as_matrix(),
                np.asarray(item.ee_translation_robot_m, dtype=float),
            )
        )

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            _transform(
                Rotation.from_rotvec(values[:3]).as_matrix(),
                values[3:6],
            ),
            _transform(
                Rotation.from_rotvec(values[6:9]).as_matrix(),
                values[9:12],
            ),
        )

    translation_scale = float(profile.get("translation_scale_m", 0.005))
    rotation_scale = np.deg2rad(
        float(profile.get("rotation_scale_deg", 1.0))
    )

    def residual(values: np.ndarray) -> np.ndarray:
        controller_from_base, cad_ee_from_controller_ee = unpack(values)
        result = []
        for cad_pose, observed_pose in zip(
            base_from_ee,
            controller_from_ee,
        ):
            error = (
                np.linalg.inv(observed_pose)
                @ controller_from_base
                @ cad_pose
                @ cad_ee_from_controller_ee
            )
            result.extend(error[:3, 3] / translation_scale)
            result.extend(
                Rotation.from_matrix(error[:3, :3]).as_rotvec()
                / rotation_scale
            )
        return np.asarray(result)

    # The first pose provides a deterministic base initialization when the two
    # EE conventions are initially assumed equal.  Joint excitation then
    # separates the base transform from the constant tool/site transform.
    initial_controller_from_base = (
        controller_from_ee[0] @ np.linalg.inv(base_from_ee[0])
    )
    initial = np.r_[
        Rotation.from_matrix(
            initial_controller_from_base[:3, :3]
        ).as_rotvec(),
        initial_controller_from_base[:3, 3],
        np.zeros(6),
    ]
    result = least_squares(
        residual,
        initial,
        loss="soft_l1",
        max_nfev=int(profile.get("maximum_function_evaluations", 5000)),
    )
    controller_from_base, cad_ee_from_controller_ee = unpack(result.x)
    translation_errors = []
    rotation_errors = []
    per_pose = []
    for item, cad_pose, observed_pose in zip(
        observations,
        base_from_ee,
        controller_from_ee,
    ):
        predicted = (
            controller_from_base
            @ cad_pose
            @ cad_ee_from_controller_ee
        )
        error = np.linalg.inv(observed_pose) @ predicted
        translation_error = float(np.linalg.norm(error[:3, 3]))
        rotation_error = float(
            np.degrees(
                Rotation.from_matrix(error[:3, :3]).magnitude()
            )
        )
        translation_errors.append(translation_error)
        rotation_errors.append(rotation_error)
        per_pose.append(
            {
                "session_id": item.session_id,
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
            }
        )
    rank = int(np.linalg.matrix_rank(result.jac))
    accepted = bool(
        rank == 12
        and max(translation_errors)
        <= float(profile.get("maximum_translation_error_m", 0.003))
        and max(rotation_errors)
        <= float(profile.get("maximum_rotation_error_deg", 0.5))
    )
    reasons = []
    if rank != 12:
        reasons.append("kinematic_bridge_jacobian_rank_deficient")
    if max(translation_errors) > float(
        profile.get("maximum_translation_error_m", 0.003)
    ):
        reasons.append("kinematic_bridge_translation_error")
    if max(rotation_errors) > float(
        profile.get("maximum_rotation_error_deg", 0.5)
    ):
        reasons.append("kinematic_bridge_rotation_error")
    return {
        "accepted": accepted,
        "reasons": reasons,
        "model_path": str(Path(model_path).resolve()),
        "model_branch": model_branch,
        "physical_arm": "right",
        "controller_from_cad_base": controller_from_base.tolist(),
        "cad_ee_from_controller_ee": (
            cad_ee_from_controller_ee.tolist()
        ),
        "scene_from_cad_base": scene_from_base.tolist(),
        "maximum_translation_error_m": max(translation_errors),
        "maximum_rotation_error_deg": max(rotation_errors),
        "jacobian_rank": rank,
        "parameter_count": 12,
        "poses": per_pose,
    }


def make_model_registered_object_scene(
    calibration: dict,
    kinematic_bridge: dict,
    geometry_gate: dict,
    *,
    current_episode: str,
    support_plane_z_m: float,
    radius_m: float,
    height_m: float,
) -> dict:
    """Map a wrist-derived controller point through exact CAD into the scene."""

    targets = calibration["target_robot_xyz_m_by_episode"]
    if current_episode not in targets:
        raise ValueError(f"current episode {current_episode} was not fitted")
    controller_from_base = np.asarray(
        kinematic_bridge["controller_from_cad_base"],
        dtype=float,
    )
    scene_from_base = np.asarray(
        kinematic_bridge["scene_from_cad_base"],
        dtype=float,
    )
    scene_from_controller = scene_from_base @ np.linalg.inv(
        controller_from_base
    )
    episode_targets_scene = {
        episode: (
            scene_from_controller
            @ np.r_[np.asarray(target, dtype=float), 1.0]
        )[:3]
        for episode, target in targets.items()
    }
    target_scene_measured = episode_targets_scene[current_episode]
    target_scene = target_scene_measured.copy()
    target_scene[2] = float(support_plane_z_m) + float(height_m) / 2.0
    pose = np.eye(4)
    pose[:3, 3] = target_scene
    return {
        "schema": OBJECT_SCENE_SCHEMA,
        "source": {
            "kind": (
                "wrist_rgbd_hand_eye_plus_exact_cad_kinematic_bridge"
            ),
            "current_episode": current_episode,
            "wrist_calibration_accepted": calibration["accepted"],
            "kinematic_bridge_accepted": kinematic_bridge["accepted"],
            "target_geometry_gate_accepted": geometry_gate["accepted"],
            "episode_targets_scene_xyz_m": {
                key: value.tolist()
                for key, value in episode_targets_scene.items()
            },
        },
        "camera_to_scene_accepted": bool(
            calibration["accepted"]
            and kinematic_bridge["accepted"]
            and geometry_gate["accepted"]
        ),
        "operator_confirmed": True,
        "objects": [
            {
                "instance_id": "petri-lid-target",
                "semantic_name": "petri dish lid",
                "role": "target_lid",
                "status": "confirmed",
                "pose_scene": pose.tolist(),
                "geometry": {
                    "type": "cylinder",
                    "radius_m": float(radius_m),
                    "height_m": float(height_m),
                    "pose_anchor": "center",
                },
                "perception": {
                    "semantic_identity": "SAM scene catalog plus reviewed labels",
                    "metric_marker": "tool-relative blue cross with TrueDepth",
                    "fixed_pixel_roi": False,
                    "measured_marker_scene_xyz_m": (
                        target_scene_measured.tolist()
                    ),
                    "support_constrained_center_scene_xyz_m": (
                        target_scene.tolist()
                    ),
                    "support_plane_z_m": float(support_plane_z_m),
                    "z_policy": (
                        "marker XY from wrist RGB-D; transparent-object center "
                        "Z constrained to measured support plus half thickness"
                    ),
                    "wrist_hand_eye_accepted": calibration["accepted"],
                    "kinematic_bridge_accepted": (
                        kinematic_bridge["accepted"]
                    ),
                    "target_geometry_gate": geometry_gate,
                    "wrist_train_rms_m": calibration["train_rms_m"],
                    "wrist_maximum_holdout_point_error_m": calibration[
                        "maximum_holdout_point_error_m"
                    ],
                },
            }
        ],
    }


def evaluate_target_geometry_gate(
    observations: Iterable[TargetObservation],
    calibration: dict,
    kinematic_bridge: dict,
    *,
    model_path: str | Path,
    model_branch: str,
    radius_m: float,
    profile: dict,
) -> dict:
    """Reject target mappings that are implausibly far from the measured EE.

    Distances are metric and normalized by object radius.  This deliberately
    avoids camera pixels and prevents a high-confidence but grossly misplaced
    target from reaching trajectory planning.
    """

    observations = list(observations)
    base_from_ee, scene_from_base = _cad_poses(
        observations,
        model_path,
        model_branch,
    )
    controller_from_base = np.asarray(
        kinematic_bridge["controller_from_cad_base"],
        dtype=float,
    )
    scene_from_controller = scene_from_base @ np.linalg.inv(
        controller_from_base
    )
    target_scene_by_episode = {
        episode: (
            scene_from_controller
            @ np.r_[np.asarray(point, dtype=float), 1.0]
        )[:3]
        for episode, point in calibration[
            "target_robot_xyz_m_by_episode"
        ].items()
    }
    records = []
    by_episode: dict[str, list[dict]] = {}
    for observation, cad_pose in zip(observations, base_from_ee):
        target = target_scene_by_episode.get(observation.episode)
        if target is None:
            continue
        ee_scene = (scene_from_base @ cad_pose)[:3, 3]
        distance = float(np.linalg.norm(target - ee_scene))
        record = {
            "session_id": observation.session_id,
            "episode": observation.episode,
            "split": observation.split,
            "ee_site_scene_xyz_m": ee_scene.tolist(),
            "target_scene_xyz_m": target.tolist(),
            "ee_to_target_distance_m": distance,
            "distance_in_object_radii": distance / float(radius_m),
        }
        records.append(record)
        by_episode.setdefault(observation.episode, []).append(record)
    required = list(
        profile.get(
            "required_episodes",
            ["successful_grasp_before_lift"],
        )
    )
    minimum_radii = float(
        profile.get("minimum_nearest_distance_object_radii", 0.25)
    )
    maximum_radii = float(
        profile.get("maximum_nearest_distance_object_radii", 3.0)
    )
    episode_results = {}
    reasons = []
    for episode in required:
        candidates = by_episode.get(episode, [])
        if not candidates:
            reasons.append(f"missing_geometry_gate_episode:{episode}")
            continue
        nearest = min(
            candidates,
            key=lambda item: item["ee_to_target_distance_m"],
        )
        ratio = nearest["distance_in_object_radii"]
        accepted = minimum_radii <= ratio <= maximum_radii
        episode_results[episode] = {
            "accepted": accepted,
            "nearest_observation": nearest,
            "minimum_allowed_distance_m": minimum_radii * float(radius_m),
            "maximum_allowed_distance_m": maximum_radii * float(radius_m),
            "physical_scale_source": (
                "configured_object_radius_not_image_pixels"
            ),
        }
        if not accepted:
            reasons.append(f"implausible_ee_target_distance:{episode}")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "required_episodes": required,
        "object_radius_m": float(radius_m),
        "thresholds_in_object_radii": {
            "minimum": minimum_radii,
            "maximum": maximum_radii,
        },
        "episodes": episode_results,
        "observations": records,
    }


def calibrate_from_config(
    config: dict,
    output_dir: str | Path,
    *,
    scene_model: str | Path | None = None,
) -> dict:
    output = Path(output_dir).resolve()
    overlay_dir = output / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    marker_profile = dict(config["marker_profile"])
    observations = []
    for index, item in enumerate(config["captures"]):
        observations.append(
            observe_capture(
                item["path"],
                episode=item["episode"],
                split=item.get("split", "fit"),
                marker_profile=marker_profile,
                overlay_path=overlay_dir / f"{index:02d}_{item['episode']}.png",
            )
        )
    calibration = fit_static_target_hand_eye(
        observations,
        config.get("fit", {}),
    )
    bridge_config = config["kinematic_bridge"]
    bridge_model = (
        Path(scene_model)
        if scene_model is not None
        else Path(bridge_config["model"])
    )
    kinematic_bridge = fit_controller_cad_bridge(
        observations,
        model_path=bridge_model,
        model_branch=bridge_config["model_branch"],
        profile=bridge_config,
    )
    object_config = config["object_scene"]
    geometry_gate = evaluate_target_geometry_gate(
        observations,
        calibration,
        kinematic_bridge,
        model_path=bridge_model,
        model_branch=bridge_config["model_branch"],
        radius_m=float(object_config["radius_m"]),
        profile=config.get("target_geometry_gate", {}),
    )
    report = {
        "schema": SCHEMA,
        "accepted": bool(
            calibration["accepted"]
            and kinematic_bridge["accepted"]
            and geometry_gate["accepted"]
        ),
        "commands_sent": False,
        "observation_only": True,
        "marker_profile": marker_profile,
        "capture_observations": [asdict(item) for item in observations],
        "fit": calibration,
        "kinematic_bridge": kinematic_bridge,
        "target_geometry_gate": geometry_gate,
    }
    object_scene = make_model_registered_object_scene(
        calibration,
        kinematic_bridge,
        geometry_gate,
        current_episode=object_config["current_episode"],
        support_plane_z_m=object_config["support_plane_z_m"],
        radius_m=object_config["radius_m"],
        height_m=object_config["height_m"],
    )
    report_path = output / "wrist_target_report.json"
    object_path = output / "latest_target_scene.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    object_path.write_text(
        json.dumps(object_scene, indent=2, ensure_ascii=False) + "\n"
    )
    return {
        "report": report,
        "object_scene": object_scene,
        "report_path": str(report_path),
        "object_scene_path": str(object_path),
    }
