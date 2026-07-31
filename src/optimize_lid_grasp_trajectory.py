#!/usr/bin/env python3
"""Search and physically verify a lid grasp in the accepted semantic MuJoCo.

This module is simulation-only.  It derives an articulated contact model from
the reviewed fixed NYU gripper visual, makes only the configured lid dynamic,
solves several rim-grasp candidates, and accepts a candidate only when the lid
is contacted from both sides and follows a verification lift.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.arm.home import (
    physical_to_semantic_model_q_offset,
    semantic_model_home_q,
)


SCHEMA = "piper_robot.simulated_lid_grasp_search/v1"
TRAJECTORY_SCHEMA = "piper_robot.simulated_lid_grasp_trajectory/v1"
GRASP_SITE_LOCAL = np.asarray([-0.0288, -0.0183, 0.058], dtype=float)
OPEN_HALF_GAP_M = 0.014
CLOSED_TARGET_HALF_GAP_M = 0.0024
JOINT_LIMIT_MARGIN_RAD = 0.02
VERTICAL_LIFT_HEIGHT_M = 0.040
VERTICAL_LIFT_WAYPOINT_COUNT = 8


def _numbers(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _object_by_role(scene: dict, role: str) -> dict:
    matches = [
        item for item in scene.get("objects", ()) if item.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {role}, got {len(matches)}")
    return matches[0]


def _object_position(item: dict) -> np.ndarray:
    pose = np.asarray(item["pose_scene"], dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("object pose_scene must be a finite 4x4 transform")
    return pose[:3, 3].copy()


def _validate_open_ratio(open_ratio: float) -> float:
    open_ratio = float(open_ratio)
    if not math.isfinite(open_ratio) or not 0.0 <= open_ratio <= 1.0:
        raise ValueError("demonstrated gripper open ratio must be in [0, 1]")
    return open_ratio


def load_demonstrated_closure(
    replay_config_path: str | Path,
) -> dict:
    """Load the stationary, measured closed grasp used as jaw provenance."""

    replay_config_path = Path(replay_config_path).resolve()
    profile = json.loads(replay_config_path.read_text())
    matches = [
        item
        for item in profile.get("measured_keyframes", ())
        if item.get("stage") == "closed_nonempty"
    ]
    if len(matches) != 1:
        raise ValueError(
            "replay profile must contain exactly one closed_nonempty keyframe"
        )
    capture = Path(matches[0]["capture"])
    if not capture.is_absolute():
        capture = Path(__file__).resolve().parents[1] / capture
    manifest_path = (capture / "manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text())
    robot_state = manifest.get("robot_state", {})
    if robot_state.get("commands_sent") is not False:
        raise ValueError("demonstrated closure must be observation-only")
    if robot_state.get("stability", {}).get("stationary") is not True:
        raise ValueError("demonstrated closure must be stationary")
    ratio = _validate_open_ratio(
        robot_state["after"]["right_gripper_open_ratio"]
    )
    return {
        "replay_config": str(replay_config_path),
        "keyframe_name": matches[0]["name"],
        "capture": str(capture.resolve()),
        "manifest": str(manifest_path),
        "session_id": manifest.get("session_id"),
        "right_gripper_open_ratio": ratio,
        "proxy_target_half_gap_m": CLOSED_TARGET_HALF_GAP_M,
        "mapping": (
            "measured ratio is preserved for physical replay; proxy jaws use "
            "a separate contact-seeking target because linkage width is not "
            "calibrated"
        ),
    }


def _vertical_lift_targets(
    grasp_position: np.ndarray,
    *,
    height_m: float = VERTICAL_LIFT_HEIGHT_M,
    count: int = VERTICAL_LIFT_WAYPOINT_COUNT,
) -> list[np.ndarray]:
    if height_m <= 0 or count < 2:
        raise ValueError("vertical lift needs positive height and >=2 waypoints")
    grasp_position = np.asarray(grasp_position, dtype=float)
    return [
        grasp_position + np.asarray([0.0, 0.0, height_m * index / count])
        for index in range(1, count + 1)
    ]


def build_articulated_grasp_model(
    *,
    model_path: str | Path,
    object_scene: dict,
    output_path: str | Path,
) -> Path:
    """Preserve the reviewed model and add simulation-only contact dynamics."""

    model_path = Path(model_path).resolve()
    root = ET.parse(model_path).getroot()
    # Semantic scene MJCFs intentionally include the reviewed robot as a
    # separate file.  Inline that one top-level model in the derived artifact
    # so the simulation-only jaw joints can be added without changing either
    # reviewed source.
    includes = list(root.findall("include"))
    if len(includes) != 1:
        raise ValueError(
            f"expected one reviewed robot include, got {len(includes)}"
        )
    included_path = Path(includes[0].get("file")).resolve()
    included = ET.parse(included_path).getroot()
    root.remove(includes[0])
    for tag in ("default", "option"):
        child = included.find(tag)
        if child is not None and root.find(tag) is None:
            root.insert(0, child)
    for tag in ("asset", "worldbody", "contact", "equality", "actuator"):
        source = included.find(tag)
        if source is None:
            continue
        destination = root.find(tag)
        if destination is None:
            destination = ET.SubElement(root, tag)
        children = list(source)
        if tag == "worldbody":
            for child in reversed(children):
                destination.insert(0, child)
        else:
            destination.extend(children)
    compiler = root.find("compiler")
    meshdir = (
        (model_path.parent / compiler.get("meshdir")).resolve()
        if compiler is not None and compiler.get("meshdir")
        else model_path.parent
    )
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for name, filename in (
        ("grasp_search_housing", "gripper_housing.stl"),
        ("grasp_search_upper", "gripper_upper.stl"),
        ("grasp_search_lower", "gripper_lower.stl"),
    ):
        old = asset.find(f"mesh[@name='{name}']")
        if old is not None:
            asset.remove(old)
        source = (
            Path(__file__).resolve().parents[1]
            / "robot/piper-mujoco/mjcf/meshes/piper"
            / filename
        )
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": name,
                "file": str(source.resolve()),
                "scale": "0.001 0.001 0.001",
            },
        )
    gripper = root.find(".//body[@name='right/gripper_base']")
    if gripper is None:
        raise ValueError("reviewed model lacks right/gripper_base")
    fixed_collision = gripper.find(
        "geom[@name='right/nyu_gripper_collision']"
    )
    if fixed_collision is None:
        raise ValueError("reviewed NYU gripper collision disappeared")
    fixed_collision.set("contype", "0")
    fixed_collision.set("conaffinity", "0")
    fixed_collision.set("group", "2")
    for name in ("right/grasp", "right/grasp_search_center"):
        old = gripper.find(f"site[@name='{name}']")
        if old is not None:
            gripper.remove(old)
    ET.SubElement(
        gripper,
        "site",
        {
            "name": "right/grasp",
            "pos": _numbers(GRASP_SITE_LOCAL),
            "size": "0.003",
            "rgba": "1 0.3 0 1",
        },
    )
    for name in ("right/grasp_search_upper", "right/grasp_search_lower"):
        old = gripper.find(f"body[@name='{name}']")
        if old is not None:
            gripper.remove(old)
    common = {
        "pos": _numbers(GRASP_SITE_LOCAL),
        "gravcomp": "1",
    }
    upper = ET.SubElement(
        gripper, "body", {"name": "right/grasp_search_upper", **common}
    )
    ET.SubElement(
        upper,
        "joint",
        {
            "name": "right/grasp_search_upper_joint",
            "type": "slide",
            "axis": "0 1 0",
            "range": f"{CLOSED_TARGET_HALF_GAP_M} {OPEN_HALF_GAP_M}",
            "damping": "2",
            "armature": "0.002",
        },
    )
    ET.SubElement(
        upper,
        "geom",
        {
            "name": "right/grasp_search_upper_pad",
            "type": "box",
            "size": "0.020 0.0025 0.010",
            "rgba": "1 0.35 0.05 0.90",
            "friction": "3 0.01 0.001",
            "solref": "0.004 1",
            "solimp": "0.95 0.99 0.001",
            "density": "800",
        },
    )
    lower = ET.SubElement(
        gripper, "body", {"name": "right/grasp_search_lower", **common}
    )
    ET.SubElement(
        lower,
        "joint",
        {
            "name": "right/grasp_search_lower_joint",
            "type": "slide",
            "axis": "0 1 0",
            "range": f"{-OPEN_HALF_GAP_M} {-CLOSED_TARGET_HALF_GAP_M}",
            "damping": "2",
            "armature": "0.002",
        },
    )
    ET.SubElement(
        lower,
        "geom",
        {
            "name": "right/grasp_search_lower_pad",
            "type": "box",
            "size": "0.020 0.0025 0.010",
            "rgba": "1 0.35 0.05 0.90",
            "friction": "3 0.01 0.001",
            "solref": "0.004 1",
            "solimp": "0.95 0.99 0.001",
            "density": "800",
        },
    )
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for old in list(equality):
        if old.get("name") == "right/grasp_search_coupling":
            equality.remove(old)
    ET.SubElement(
        equality,
        "joint",
        {
            "name": "right/grasp_search_coupling",
            "joint1": "right/grasp_search_upper_joint",
            "joint2": "right/grasp_search_lower_joint",
            "polycoef": "0 -1 0 0 0",
        },
    )
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for old in list(actuator):
        if old.get("name") == "right/grasp_search_close":
            actuator.remove(old)
    ET.SubElement(
        actuator,
        "position",
        {
            "name": "right/grasp_search_close",
            "joint": "right/grasp_search_upper_joint",
            "kp": "180",
            "kv": "8",
            "ctrlrange": (
                f"{CLOSED_TARGET_HALF_GAP_M} {OPEN_HALF_GAP_M}"
            ),
            "forcerange": "-20 20",
        },
    )
    lid = _object_by_role(object_scene, "target_lid")
    body_name = str(lid.get("body_name", "petri_lid-1"))
    lid_body = root.find(f".//body[@name='{body_name}']")
    if lid_body is None:
        # Current scene files keep the body name in the refresh report, while
        # the portable object scene uses semantic identity.
        lid_body = root.find(".//body[@name='petri_lid-1']")
    if lid_body is None:
        raise ValueError("reviewed model lacks the target lid body")
    for joint in list(lid_body.findall("joint")):
        lid_body.remove(joint)
    ET.SubElement(lid_body, "freejoint", {"name": "grasp_search_lid_free"})
    lid_geom = lid_body.find("geom")
    if lid_geom is None:
        raise ValueError("target lid lacks collision geometry")
    lid_geom.set("name", "grasp_search_lid")
    lid_geom.set("density", "550")
    lid_geom.set("friction", "2.5 0.01 0.001")
    lid_geom.set("solref", "0.004 1")
    lid_geom.set("solimp", "0.95 0.99 0.001")
    lid_geom.set("contype", "1")
    lid_geom.set("conaffinity", "1")
    lid_position = _object_position(lid)
    lid_height = float(lid["geometry"]["height_m"])
    worldbody = root.find("worldbody")
    old_support = worldbody.find(
        "body[@name='grasp_search_local_support']"
    )
    if old_support is not None:
        worldbody.remove(old_support)
    support_half_height = 0.005
    support = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "grasp_search_local_support",
            "pos": _numbers(
                [
                    lid_position[0],
                    lid_position[1],
                    lid_position[2]
                    - 0.5 * lid_height
                    - support_half_height,
                ]
            ),
        },
    )
    ET.SubElement(
        support,
        "geom",
        {
            "name": "grasp_search_local_support_surface",
            "type": "box",
            "size": f"0.11 0.11 {support_half_height}",
            "rgba": "0.25 0.25 0.28 0.25",
            "friction": "0.8 0.01 0.001",
        },
    )
    # The full reviewed scene keeps hundreds of conservative semantic cells.
    # They remain authoritative in the source collision audit, but are not
    # needed to evaluate local pad/lid contact and make a candidate sweep
    # unnecessarily slow.  The fixed dish is the local support authority.
    # Semantic support cells overlap that completed object and otherwise snag
    # the dynamic lid throughout a vertical lift, so they remain visible but
    # do not participate in this local contact test.
    for body in root.findall(".//body"):
        name = str(body.get("name", ""))
        keep_collision = name in {
            "petri_lid-1",
            "petri_dish-1",
            "grasp_search_local_support",
        }
        if name.startswith(("left/", "right/")):
            keep_collision = False
        if not keep_collision:
            for geom in body.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    # Re-enable the two local pads added above.
    for pad_name in (
        "right/grasp_search_upper_pad",
        "right/grasp_search_lower_pad",
    ):
        pad = root.find(f".//geom[@name='{pad_name}']")
        pad.set("contype", "1")
        pad.set("conaffinity", "1")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.005")
    # Existing keyframes have a fixed nq.  The derived free lid and two jaw
    # joints make them stale, so the simulator initializes explicitly.
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(
        output_path, encoding="unicode", xml_declaration=False
    )
    mujoco.MjModel.from_xml_path(str(output_path))
    return output_path


class GraspKinematics:
    def __init__(self, model_path: str | Path):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.right_ids = np.asarray(
            [
                self.model.joint(f"right/joint{index}").qposadr[0]
                for index in range(1, 7)
            ],
            dtype=int,
        )
        self.left_ids = np.asarray(
            [
                self.model.joint(f"left/joint{index}").qposadr[0]
                for index in range(1, 7)
            ],
            dtype=int,
        )
        self.site_id = int(self.model.site("right/grasp").id)
        joint_ids = [
            int(self.model.joint(f"right/joint{index}").id)
            for index in range(1, 7)
        ]
        self.lower = np.asarray(
            [self.model.jnt_range[index, 0] for index in joint_ids]
        )
        self.upper = np.asarray(
            [self.model.jnt_range[index, 1] for index in joint_ids]
        )
        # The semantic jaw-roll calibration is a coordinate bridge, not an
        # extension of the physical Piper range.  Optimize only in the
        # intersection of model and physical limits.
        self.right_offset = physical_to_semantic_model_q_offset("right")
        physical_lower = np.asarray(
            [self.model.jnt_range[index, 0] for index in joint_ids]
        )
        physical_upper = np.asarray(
            [self.model.jnt_range[index, 1] for index in joint_ids]
        )
        self.lower = np.maximum(
            self.lower, physical_lower + self.right_offset
        ) + JOINT_LIMIT_MARGIN_RAD
        self.upper = np.minimum(
            self.upper, physical_upper + self.right_offset
        ) - JOINT_LIMIT_MARGIN_RAD
        if np.any(self.lower >= self.upper):
            raise ValueError("physical/model joint-limit intersection is empty")
        self.left_home = semantic_model_home_q("left")

    def pose(self, q_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.right_ids] = q_model
        self.data.qpos[self.left_ids] = self.left_home
        mujoco.mj_forward(self.model, self.data)
        return (
            self.data.site_xpos[self.site_id].copy(),
            self.data.site_xmat[self.site_id].reshape(3, 3).copy(),
        )

    def solve(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        seed: np.ndarray,
        *,
        maximum_position_error_m: float = 0.004,
        maximum_rotation_error_deg: float = 12.0,
    ) -> tuple[np.ndarray, dict]:
        seed = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)

        def residual(q):
            position, rotation = self.pose(q)
            orientation = Rotation.from_matrix(
                target_rotation.T @ rotation
            ).as_rotvec()
            return np.concatenate(
                [
                    (position - target_position) / 0.002,
                    orientation / math.radians(2.0),
                    0.025 * (q - seed),
                ]
            )

        result = least_squares(
            residual,
            seed,
            bounds=(self.lower + 1e-6, self.upper - 1e-6),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=300,
        )
        position, rotation = self.pose(result.x)
        position_error = float(np.linalg.norm(position - target_position))
        rotation_error = float(
            np.linalg.norm(
                Rotation.from_matrix(target_rotation.T @ rotation).as_rotvec()
            )
        )
        accepted = bool(
            result.success
            and position_error <= maximum_position_error_m
            and rotation_error <= math.radians(maximum_rotation_error_deg)
        )
        return result.x.copy(), {
            "accepted": accepted,
            "position_error_m": position_error,
            "rotation_error_deg": math.degrees(rotation_error),
            "optimizer_status": int(result.status),
            "optimizer_cost": float(result.cost),
        }


def _rotation_for_rim(outward_xy: np.ndarray, pitch_deg: float) -> np.ndarray:
    x_axis = np.asarray([outward_xy[0], outward_xy[1], 0.0], dtype=float)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.asarray([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    pitch = Rotation.from_rotvec(
        math.radians(pitch_deg) * y_axis
    ).as_matrix()
    return pitch @ rotation


def _minimum_jerk(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def _trajectory_samples(knots: list[dict], hz: float = 100.0) -> list[dict]:
    samples = []
    cursor = 0.0
    for first, second in zip(knots, knots[1:]):
        delta = float(np.max(np.abs(second["q_model"] - first["q_model"])))
        duration = max(float(second["minimum_duration_s"]), 2.5 * delta)
        count = max(2, int(math.ceil(duration * hz)))
        for index in range(1, count + 1):
            fraction = index / count
            blend = _minimum_jerk(fraction)
            samples.append(
                {
                    "t_s": cursor + fraction * duration,
                    "stage": second["stage"],
                    "q_model": (
                        first["q_model"]
                        + blend * (second["q_model"] - first["q_model"])
                    ),
                    "jaw_target_m": (
                        first["jaw_target_m"]
                        + blend
                        * (
                            second["jaw_target_m"]
                            - first["jaw_target_m"]
                        )
                    ),
                    **(
                        {
                            "right_gripper_open_ratio": (
                                first["right_gripper_open_ratio"]
                                + blend
                                * (
                                    second["right_gripper_open_ratio"]
                                    - first["right_gripper_open_ratio"]
                                )
                            )
                        }
                        if "right_gripper_open_ratio" in first
                        and "right_gripper_open_ratio" in second
                        else {}
                    ),
                }
            )
        cursor += duration
    return samples


def _contact_bodies(model, data, lid_body_id: int) -> set[str]:
    names = set()
    for contact in data.contact:
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if first == lid_body_id:
            names.add(model.body(second).name)
        elif second == lid_body_id:
            names.add(model.body(first).name)
    return names


def simulate_candidate(
    model_path: str | Path,
    samples: list[dict],
    *,
    closed_target_half_gap_m: float = CLOSED_TARGET_HALF_GAP_M,
    render: bool = False,
    video_path: str | Path | None = None,
    width: int = 720,
    height: int = 540,
    fps: int = 30,
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    right_ids = np.asarray(
        [
            model.joint(f"right/joint{index}").qposadr[0]
            for index in range(1, 7)
        ],
        dtype=int,
    )
    left_ids = np.asarray(
        [
            model.joint(f"left/joint{index}").qposadr[0]
            for index in range(1, 7)
        ],
        dtype=int,
    )
    upper_q = int(
        model.joint("right/grasp_search_upper_joint").qposadr[0]
    )
    lower_q = int(
        model.joint("right/grasp_search_lower_joint").qposadr[0]
    )
    jaw_actuator = int(model.actuator("right/grasp_search_close").id)
    lid_joint = model.joint("grasp_search_lid_free")
    lid_q = int(lid_joint.qposadr[0])
    lid_body = int(model.body("petri_lid-1").id)
    grasp_site = int(model.site("right/grasp").id)
    lid_radius = float(model.geom("grasp_search_lid").size[0])
    data.qpos[right_ids] = samples[0]["q_model"]
    data.qpos[left_ids] = semantic_model_home_q("left")
    data.qpos[upper_q] = OPEN_HALF_GAP_M
    data.qpos[lower_q] = -OPEN_HALF_GAP_M
    data.ctrl[:12] = np.concatenate(
        [semantic_model_home_q("left"), samples[0]["q_model"]]
    )
    data.ctrl[jaw_actuator] = OPEN_HALF_GAP_M
    mujoco.mj_forward(model, data)
    initial_lid = data.qpos[lid_q : lid_q + 3].copy()
    stage_contacts: dict[str, set[str]] = {}
    maximum_relative_distance = 0.0
    lift_relative_distances = []
    lift_start_grasp_xy = None
    lift_start_relative_xy = None
    lift_start_grasp_in_lid = None
    maximum_grasp_xy_deviation_during_lift = 0.0
    maximum_lid_relative_xy_slip_during_lift = 0.0
    maximum_grasp_point_slip_in_lid_frame = 0.0
    frames = []
    renderer = None
    option = None
    camera = None
    next_frame_t = 0.0
    if render:
        renderer = mujoco.Renderer(model, height=height, width=width)
        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)
        option.geomgroup[:] = 1
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.0, 0.86, -0.49]
        camera.distance = 0.88
        camera.azimuth = 28.0
        camera.elevation = -20.0
    dt = float(model.opt.timestep)
    for sample in samples:
        if (
            sample["stage"] == "verification_lift"
            and lift_start_grasp_xy is None
        ):
            lift_start_grasp_xy = data.site_xpos[grasp_site, :2].copy()
            lift_start_relative_xy = (
                data.qpos[lid_q : lid_q + 2]
                - data.site_xpos[grasp_site, :2]
            ).copy()
            lid_rotation = data.xmat[lid_body].reshape(3, 3)
            lift_start_grasp_in_lid = lid_rotation.T @ (
                data.site_xpos[grasp_site] - data.xpos[lid_body]
            )
        target_t = float(sample["t_s"])
        while data.time + 1e-12 < target_t:
            data.qpos[right_ids] = sample["q_model"]
            data.qvel[right_ids] = 0.0
            data.qpos[left_ids] = semantic_model_home_q("left")
            data.qvel[left_ids] = 0.0
            data.ctrl[:12] = np.concatenate(
                [semantic_model_home_q("left"), sample["q_model"]]
            )
            data.ctrl[jaw_actuator] = float(sample["jaw_target_m"])
            mujoco.mj_step(model, data)
            contacts = _contact_bodies(model, data, lid_body)
            stage_contacts.setdefault(sample["stage"], set()).update(contacts)
            lid_position = data.qpos[lid_q : lid_q + 3].copy()
            grasp_position = data.site_xpos[grasp_site].copy()
            relative = float(np.linalg.norm(lid_position - grasp_position))
            maximum_relative_distance = max(
                maximum_relative_distance, relative
            )
            if sample["stage"] in {"verification_lift", "hold"}:
                lift_relative_distances.append(relative)
            if sample["stage"] == "verification_lift":
                grasp_xy_deviation = float(
                    np.linalg.norm(
                        grasp_position[:2] - lift_start_grasp_xy
                    )
                )
                relative_xy = lid_position[:2] - grasp_position[:2]
                relative_xy_slip = float(
                    np.linalg.norm(relative_xy - lift_start_relative_xy)
                )
                maximum_grasp_xy_deviation_during_lift = max(
                    maximum_grasp_xy_deviation_during_lift,
                    grasp_xy_deviation,
                )
                maximum_lid_relative_xy_slip_during_lift = max(
                    maximum_lid_relative_xy_slip_during_lift,
                    relative_xy_slip,
                )
                lid_rotation = data.xmat[lid_body].reshape(3, 3)
                grasp_in_lid = lid_rotation.T @ (
                    grasp_position - data.xpos[lid_body]
                )
                grasp_point_slip = float(
                    np.linalg.norm(
                        grasp_in_lid - lift_start_grasp_in_lid
                    )
                )
                maximum_grasp_point_slip_in_lid_frame = max(
                    maximum_grasp_point_slip_in_lid_frame,
                    grasp_point_slip,
                )
            if render and data.time + 1e-12 >= next_frame_t:
                renderer.update_scene(
                    data, camera=camera, scene_option=option
                )
                frame = renderer.render().copy()
                cv2.rectangle(frame, (10, 10), (710, 58), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"PHYSICS GRASP | {sample['stage'].upper()}",
                    (22, 43),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                frames.append(frame)
                next_frame_t += 1.0 / fps
            if not np.all(np.isfinite(data.qpos)):
                break
        if not np.all(np.isfinite(data.qpos)):
            break
    final_lid = data.qpos[lid_q : lid_q + 3].copy()
    upper_position = float(data.qpos[upper_q])
    upper_contact = "right/grasp_search_upper" in set().union(
        *stage_contacts.values()
    )
    lower_contact = "right/grasp_search_lower" in set().union(
        *stage_contacts.values()
    )
    hold_contacts = stage_contacts.get("hold", set())
    hold_bilateral_contact = bool(
        "right/grasp_search_upper" in hold_contacts
        and "right/grasp_search_lower" in hold_contacts
    )
    lift_m = float(final_lid[2] - initial_lid[2])
    final_relative = float(
        np.linalg.norm(final_lid - data.site_xpos[grasp_site])
    )
    lift_relative_max = (
        max(lift_relative_distances) if lift_relative_distances else math.inf
    )
    closure_obstructed = (
        upper_position > closed_target_half_gap_m + 0.0006
    )
    maximum_lid_to_grasp_m = 1.5 * lid_radius
    maximum_grasp_xy_deviation_m = 0.075 * lid_radius
    maximum_grasp_point_slip_in_lid_frame_m = 0.10 * lid_radius
    success = bool(
        np.all(np.isfinite(data.qpos))
        and upper_contact
        and lower_contact
        and hold_bilateral_contact
        and closure_obstructed
        and lift_m >= 0.020
        and final_relative <= maximum_lid_to_grasp_m
        and lift_relative_max <= maximum_lid_to_grasp_m
        and maximum_grasp_xy_deviation_during_lift
        <= maximum_grasp_xy_deviation_m
        and maximum_grasp_point_slip_in_lid_frame
        <= maximum_grasp_point_slip_in_lid_frame_m
    )
    if renderer is not None:
        renderer.close()
    if render and video_path is not None and frames:
        video_path = Path(video_path).resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video_path),
            ],
            stdin=subprocess.PIPE,
        )
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError("ffmpeg failed")
        cv2.imwrite(
            str(video_path.with_name(video_path.stem + "_final.png")),
            cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR),
        )
    return {
        "success": success,
        "initial_lid_xyz_m": initial_lid.tolist(),
        "final_lid_xyz_m": final_lid.tolist(),
        "lid_lift_m": lift_m,
        "final_lid_to_grasp_m": final_relative,
        "maximum_lift_lid_to_grasp_m": lift_relative_max,
        "maximum_all_stage_lid_to_grasp_m": maximum_relative_distance,
        "maximum_grasp_xy_deviation_during_lift_m": (
            maximum_grasp_xy_deviation_during_lift
        ),
        "maximum_lid_relative_xy_slip_during_lift_m": (
            maximum_lid_relative_xy_slip_during_lift
        ),
        "maximum_grasp_point_slip_in_lid_frame_m": (
            maximum_grasp_point_slip_in_lid_frame
        ),
        "maximum_grasp_xy_deviation_m": maximum_grasp_xy_deviation_m,
        "maximum_grasp_point_slip_in_lid_frame_limit_m": (
            maximum_grasp_point_slip_in_lid_frame_m
        ),
        "upper_pad_contact": upper_contact,
        "lower_pad_contact": lower_contact,
        "hold_bilateral_pad_contact": hold_bilateral_contact,
        "closure_obstructed": closure_obstructed,
        "maximum_lid_to_grasp_m": maximum_lid_to_grasp_m,
        "commanded_half_gap_m": closed_target_half_gap_m,
        "measured_half_gap_m": upper_position,
        "contact_bodies_by_stage": {
            stage: sorted(values) for stage, values in stage_contacts.items()
        },
    }


def search(args) -> dict:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = json.loads(Path(args.object_scene).read_text())
    lid = _object_by_role(scene, "target_lid")
    lid_position = _object_position(lid)
    radius = float(lid["geometry"]["radius_m"])
    closure_demonstration = load_demonstrated_closure(
        args.demonstration_config
    )
    closed_target_half_gap_m = CLOSED_TARGET_HALF_GAP_M
    demonstrated_closed_open_ratio = float(
        closure_demonstration["right_gripper_open_ratio"]
    )
    derived_model = build_articulated_grasp_model(
        model_path=args.model,
        object_scene=scene,
        output_path=output / "grasp_physics_scene.mjcf",
    )
    kinematics = GraspKinematics(derived_model)
    home = semantic_model_home_q("right")
    _, home_rotation = kinematics.pose(home)
    base_position = kinematics.model.body("right/base_link").pos.copy()
    outward = lid_position[:2] - base_position[:2]
    outward = -outward / np.linalg.norm(outward)
    base_angle = math.atan2(outward[1], outward[0])
    candidates = []
    candidate_id = 0
    # Search both the demonstrated near-radial approach and deeper insertions.
    # A rim-only pinch can lift while the transparent lid rotates and slides;
    # the positive insets reduce that moment arm without hard-coding pixels.
    for angle_offset_deg in (-6, 0, 6):
        angle = base_angle + math.radians(angle_offset_deg)
        direction = np.asarray([math.cos(angle), math.sin(angle)])
        for radial_inset_m in (
            -0.004,
            0.004,
            0.012,
            0.016,
            0.020,
            0.024,
            0.028,
        ):
            for z_offset_m in (-0.004, -0.002, 0.0, 0.002, 0.004):
                candidate_id += 1
                rim = lid_position.copy()
                rim[:2] += (radius - radial_inset_m) * direction
                rim[2] += z_offset_m
                rotation = _rotation_for_rim(direction, 0.0)
                targets = [
                    (
                        "hover_xy",
                        rim
                        + np.asarray(
                            [0.055 * direction[0], 0.055 * direction[1], 0.050]
                        ),
                        1.2,
                        home_rotation,
                    ),
                    (
                        "descend",
                        rim
                        + np.asarray(
                            [0.055 * direction[0], 0.055 * direction[1], 0.0]
                        ),
                        0.8,
                        home_rotation,
                    ),
                    ("insert", rim, 0.8, rotation),
                ]
                q = home.copy()
                knots = [
                    {
                        "stage": "home",
                        "q_model": q.copy(),
                        "jaw_target_m": OPEN_HALF_GAP_M,
                        "right_gripper_open_ratio": 1.0,
                        "minimum_duration_s": 0.5,
                    }
                ]
                ik_reports = []
                accepted = True
                for stage, target, duration, stage_rotation in targets:
                    q, ik = kinematics.solve(
                        target,
                        stage_rotation,
                        q,
                        maximum_position_error_m=(
                            0.015 if stage == "hover_xy" else (
                                0.012 if stage == "descend" else 0.004
                            )
                        ),
                        maximum_rotation_error_deg=(
                            30.0 if stage in {"hover_xy", "descend"} else 12.0
                        ),
                    )
                    ik_reports.append({"stage": stage, **ik})
                    if not ik["accepted"]:
                        accepted = False
                        break
                    knots.append(
                        {
                            "stage": stage,
                            "q_model": q.copy(),
                            "jaw_target_m": OPEN_HALF_GAP_M,
                            "right_gripper_open_ratio": 1.0,
                            "minimum_duration_s": duration,
                        }
                    )
                simulation = None
                if accepted:
                    grasp_q = q.copy()
                    lift_knots = []
                    lift_q = grasp_q.copy()
                    for lift_index, lift_target in enumerate(
                        _vertical_lift_targets(rim), start=1
                    ):
                        lift_q, lift_ik = kinematics.solve(
                            lift_target,
                            rotation,
                            lift_q,
                            maximum_position_error_m=0.003,
                            maximum_rotation_error_deg=10.0,
                        )
                        ik_reports.append(
                            {
                                "stage": "verification_lift",
                                "waypoint": lift_index,
                                "target_xyz_m": lift_target.tolist(),
                                **lift_ik,
                            }
                        )
                        if not lift_ik["accepted"]:
                            accepted = False
                            break
                        lift_knots.append(
                            {
                                "stage": "verification_lift",
                                "q_model": lift_q.copy(),
                                "jaw_target_m": closed_target_half_gap_m,
                                "right_gripper_open_ratio": (
                                    demonstrated_closed_open_ratio
                                ),
                                "minimum_duration_s": 0.20,
                            }
                        )
                if accepted:
                    knots.append(
                        {
                            "stage": "close",
                            "q_model": grasp_q.copy(),
                            "jaw_target_m": closed_target_half_gap_m,
                            "right_gripper_open_ratio": (
                                demonstrated_closed_open_ratio
                            ),
                            "minimum_duration_s": 1.2,
                        }
                    )
                    knots.extend(lift_knots)
                    knots.append(
                        {
                            "stage": "hold",
                            "q_model": lift_q.copy(),
                            "jaw_target_m": closed_target_half_gap_m,
                            "right_gripper_open_ratio": (
                                demonstrated_closed_open_ratio
                            ),
                            "minimum_duration_s": 1.0,
                        }
                    )
                    samples = _trajectory_samples(knots)
                    simulation = simulate_candidate(
                        derived_model,
                        samples,
                        closed_target_half_gap_m=closed_target_half_gap_m,
                    )
                score = (
                    (
                        100.0
                        + 1000.0 * simulation["lid_lift_m"]
                        - 100.0 * simulation["final_lid_to_grasp_m"]
                        - 5000.0
                        * simulation[
                            "maximum_grasp_point_slip_in_lid_frame_m"
                        ]
                        - 2000.0
                        * simulation[
                            "maximum_grasp_xy_deviation_during_lift_m"
                        ]
                    )
                    if simulation and simulation["success"]
                    else (
                        -1000.0
                        if simulation is None
                        else (
                            20.0 * int(simulation["upper_pad_contact"])
                            + 20.0 * int(simulation["lower_pad_contact"])
                            + 500.0 * simulation["lid_lift_m"]
                            - 100.0
                            * simulation["final_lid_to_grasp_m"]
                        )
                    )
                )
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "angle_offset_deg": angle_offset_deg,
                        "radial_inset_m": radial_inset_m,
                        "z_offset_m": z_offset_m,
                        "rim_target_xyz_m": rim.tolist(),
                        "ik_accepted": accepted,
                        "ik": ik_reports,
                        "simulation": simulation,
                        "score": score,
                        "_knots": knots if accepted else None,
                    }
                )
    ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
    successful = [
        item
        for item in ranked
        if item["simulation"] is not None
        and item["simulation"]["success"]
    ]
    best = successful[0] if successful else ranked[0]
    if best["_knots"] is not None:
        best_samples = _trajectory_samples(best["_knots"])
        best["simulation"] = simulate_candidate(
            derived_model,
            best_samples,
            closed_target_half_gap_m=closed_target_half_gap_m,
            render=True,
            video_path=output / "best_lid_grasp.mp4",
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        trajectory = {
            "schema": TRAJECTORY_SCHEMA,
            "commands_sent": False,
            "simulation_only": True,
            "model_path": str(derived_model),
            "source_model_path": str(Path(args.model).resolve()),
            "object_scene_path": str(Path(args.object_scene).resolve()),
            "candidate_id": best["candidate_id"],
            "joint_limit_margin_rad": JOINT_LIMIT_MARGIN_RAD,
            "closure_demonstration": closure_demonstration,
            "right_physical_to_model_q_offset_rad": (
                physical_to_semantic_model_q_offset("right").tolist()
            ),
            "knots": [
                {
                    **{key: value for key, value in knot.items() if key != "q_model"},
                    "right_q_model_rad": knot["q_model"].tolist(),
                    "right_q_physical_rad": (
                        knot["q_model"]
                        - physical_to_semantic_model_q_offset("right")
                    ).tolist(),
                }
                for knot in best["_knots"]
            ],
            "simulation_validation": best["simulation"],
        }
        (output / "best_lid_grasp_trajectory.json").write_text(
            json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n"
        )
    for item in candidates:
        item.pop("_knots", None)
    best.pop("_knots", None)
    report = {
        "schema": SCHEMA,
        "commands_sent": False,
        "simulation_only": True,
        "source_model": str(Path(args.model).resolve()),
        "derived_model": str(derived_model),
        "object_scene": str(Path(args.object_scene).resolve()),
        "lid_position_scene_m": lid_position.tolist(),
        "lid_radius_m": radius,
        "joint_limit_margin_rad": JOINT_LIMIT_MARGIN_RAD,
        "closure_demonstration": closure_demonstration,
        "vertical_lift": {
            "height_m": VERTICAL_LIFT_HEIGHT_M,
            "waypoint_count": VERTICAL_LIFT_WAYPOINT_COUNT,
            "fixed_xy_and_orientation": True,
        },
        "candidate_count": len(candidates),
        "ik_accepted_count": sum(item["ik_accepted"] for item in candidates),
        "successful_candidate_count": len(successful),
        "accepted": bool(
            best["simulation"] is not None and best["simulation"]["success"]
        ),
        "best_candidate": best,
        "candidates": sorted(candidates, key=lambda item: item["candidate_id"]),
        "artifacts": {
            "trajectory": str(
                (output / "best_lid_grasp_trajectory.json").resolve()
            ),
            "video": str((output / "best_lid_grasp.mp4").resolve()),
            "final_image": str(
                (output / "best_lid_grasp_final.png").resolve()
            ),
        },
        "hardware_motion_authorized": False,
    }
    (output / "grasp_search_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--object-scene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--demonstration-config", required=True)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("render dimensions and fps must be positive")
    report = search(args)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
