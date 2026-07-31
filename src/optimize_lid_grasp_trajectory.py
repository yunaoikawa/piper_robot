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
# Grasp midpoint measured from the two inner fingertip clusters in the exact
# pinned `gripper_body.stl` used by the reviewed semantic scene.  The older
# [-0.0288, -0.0183, 0.058] site belongs to the differently nested upstream
# articulated model and does not coincide with this fixed mesh.
NYU_GRASP_SITE_LOCAL = np.asarray([-0.1305, 0.0195, 0.0289], dtype=float)
NYU_JAW_AXIS_LOCAL = np.asarray([0.0, 1.0, 0.0], dtype=float)
# The separated fingertip meshes bottom at local Z ~=14.6 mm.  With their
# bottom on the support and a 6 mm lid centered 3 mm above it, the lid-contact
# patch is 11.3 mm below the jaw-mesh midpoint.  MuJoCo collision uses the
# mesh's convex hull, whose low support point is 3.7 mm below the visible STL
# minimum in this pose, so the collision-safe calibrated offset is 15.0 mm.
NYU_GRASP_REFERENCE_TO_PAD_M = 0.0150
NYU_PAD_CENTER_LOCAL = NYU_GRASP_SITE_LOCAL - np.asarray(
    [0.0, 0.0, NYU_GRASP_REFERENCE_TO_PAD_M]
)
# The fixed mesh's open inner fingertip surfaces are about 135 mm apart.
OPEN_HALF_GAP_M = 0.0675
CLOSED_TARGET_HALF_GAP_M = 0.0024
JOINT_LIMIT_MARGIN_RAD = 0.02
VERTICAL_LIFT_HEIGHT_M = 0.040
VERTICAL_LIFT_WAYPOINT_COUNT = 16
VERTICAL_LIFT_WAYPOINT_DURATION_S = 0.25
MINIMUM_PAD_SIDE_CONTACT_ALIGNMENT = math.cos(math.radians(55.0))
CLOSE_RAMP_DURATION_S = 2.0
CLOSE_SETTLE_DURATION_S = 15.0
PAD_CONTACT_HALF_THICKNESS_M = 0.0025
BILATERAL_PRECONTACT_DISTANCE_M = 8.0 * PAD_CONTACT_HALF_THICKNESS_M
BILATERAL_CONTACT_PRELOAD_M = (
    BILATERAL_PRECONTACT_DISTANCE_M
    + 4.5 * PAD_CONTACT_HALF_THICKNESS_M
)


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
    right_q_physical = np.asarray(
        robot_state["after"]["right_joint_positions_rad"], dtype=float
    )
    if right_q_physical.shape != (6,) or not np.all(
        np.isfinite(right_q_physical)
    ):
        raise ValueError("demonstrated right q must contain six finite joints")
    return {
        "replay_config": str(replay_config_path),
        "keyframe_name": matches[0]["name"],
        "capture": str(capture.resolve()),
        "manifest": str(manifest_path),
        "session_id": manifest.get("session_id"),
        "right_gripper_open_ratio": ratio,
        "right_q_physical_rad": right_q_physical.tolist(),
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
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "960")
    gripper = root.find(".//body[@name='right/gripper_base']")
    if gripper is None:
        raise ValueError("reviewed model lacks right/gripper_base")
    fixed_visual = gripper.find("geom[@name='right/nyu_gripper_visual']")
    fixed_collision = gripper.find("geom[@name='right/nyu_gripper_collision']")
    if fixed_visual is None:
        raise ValueError("reviewed NYU gripper visual disappeared")
    if fixed_collision is None:
        raise ValueError("reviewed NYU gripper collision disappeared")
    # The reviewed asset is a single mesh containing housing and both open
    # fingers.  Keep it in the source model, but replace it only in this
    # derived physics artifact with the matching separated meshes so the
    # fingers can close and collision does not use one giant convex hull.
    gripper.remove(fixed_visual)
    gripper.remove(fixed_collision)
    ET.SubElement(
        gripper,
        "geom",
        {
            "name": "right/grasp_search_housing_visual",
            "type": "mesh",
            "mesh": "grasp_search_housing",
            "rgba": "0.15 0.70 0.90 1",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        gripper,
        "geom",
        {
            "name": "right/grasp_search_housing_collision",
            "type": "mesh",
            "mesh": "grasp_search_housing",
            "rgba": "0 0 0 0",
            "contype": "2",
            "conaffinity": "2",
            "density": "0",
            "group": "2",
        },
    )
    for name in ("right/grasp", "right/grasp_search_center"):
        old = gripper.find(f"site[@name='{name}']")
        if old is not None:
            gripper.remove(old)
    ET.SubElement(
        gripper,
        "site",
        {
            "name": "right/grasp",
            "pos": _numbers(NYU_GRASP_SITE_LOCAL),
            "size": "0.003",
            "rgba": "1 0.3 0 1",
        },
    )
    ET.SubElement(
        gripper,
        "site",
        {
            "name": "right/grasp_search_center",
            "pos": _numbers(NYU_PAD_CENTER_LOCAL),
            "size": "0.003",
            "rgba": "0.1 1 0.1 1",
        },
    )
    for name in ("right/grasp_search_upper", "right/grasp_search_lower"):
        old = gripper.find(f"body[@name='{name}']")
        if old is not None:
            gripper.remove(old)
    common = {
        "pos": _numbers(NYU_PAD_CENTER_LOCAL),
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
            "axis": _numbers(NYU_JAW_AXIS_LOCAL),
            "range": f"{CLOSED_TARGET_HALF_GAP_M} {OPEN_HALF_GAP_M}",
            "damping": "8",
            "armature": "0.002",
        },
    )
    ET.SubElement(
        upper,
        "geom",
        {
            "name": "right/grasp_search_upper_visual",
            "type": "mesh",
            "mesh": "grasp_search_upper",
            "pos": _numbers(
                -NYU_PAD_CENTER_LOCAL
                - OPEN_HALF_GAP_M * NYU_JAW_AXIS_LOCAL
            ),
            "rgba": "0.15 0.70 0.90 1",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        upper,
        "geom",
        {
            "name": "right/grasp_search_upper_environment_collision",
            "type": "mesh",
            "mesh": "grasp_search_upper",
            "pos": _numbers(
                -NYU_PAD_CENTER_LOCAL
                - OPEN_HALF_GAP_M * NYU_JAW_AXIS_LOCAL
            ),
            "rgba": "0 0 0 0",
            "contype": "2",
            "conaffinity": "2",
            "density": "0",
            "group": "2",
        },
    )
    ET.SubElement(
        upper,
        "geom",
        {
            "name": "right/grasp_search_upper_pad",
            "type": "box",
            "pos": "0 0 0.007",
            "size": "0.008 0.0025 0.010",
            "rgba": "1 0.35 0.05 0.35",
            "friction": "5 0.01 0.001",
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
            "axis": _numbers(NYU_JAW_AXIS_LOCAL),
            "range": f"{-OPEN_HALF_GAP_M} {-CLOSED_TARGET_HALF_GAP_M}",
            "damping": "8",
            "armature": "0.002",
        },
    )
    ET.SubElement(
        lower,
        "geom",
        {
            "name": "right/grasp_search_lower_visual",
            "type": "mesh",
            "mesh": "grasp_search_lower",
            "pos": _numbers(
                -NYU_PAD_CENTER_LOCAL
                + OPEN_HALF_GAP_M * NYU_JAW_AXIS_LOCAL
            ),
            "rgba": "0.15 0.70 0.90 1",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "1",
        },
    )
    ET.SubElement(
        lower,
        "geom",
        {
            "name": "right/grasp_search_lower_environment_collision",
            "type": "mesh",
            "mesh": "grasp_search_lower",
            "pos": _numbers(
                -NYU_PAD_CENTER_LOCAL
                + OPEN_HALF_GAP_M * NYU_JAW_AXIS_LOCAL
            ),
            "rgba": "0 0 0 0",
            "contype": "2",
            "conaffinity": "2",
            "density": "0",
            "group": "2",
        },
    )
    ET.SubElement(
        lower,
        "geom",
        {
            "name": "right/grasp_search_lower_pad",
            "type": "box",
            "pos": "0 0 0.007",
            "size": "0.008 0.0025 0.010",
            "rgba": "1 0.35 0.05 0.35",
            "friction": "5 0.01 0.001",
            "solref": "0.004 1",
            "solimp": "0.95 0.99 0.001",
            "density": "800",
        },
    )
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for old in list(contact.findall("exclude")):
        if old.get("name") == "right/grasp_search_jaw_pair_exclude":
            contact.remove(old)
    ET.SubElement(
        contact,
        "exclude",
        {
            "name": "right/grasp_search_jaw_pair_exclude",
            "body1": "right/grasp_search_upper",
            "body2": "right/grasp_search_lower",
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
            "solref": "0.001 1",
            "solimp": "0.999 0.9999 0.00001",
        },
    )
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for old in list(actuator):
        if old.get("name") in {
            "right/grasp_search_close",
            "right/grasp_search_close_lower",
        }:
            actuator.remove(old)
    ET.SubElement(
        actuator,
        "position",
        {
            "name": "right/grasp_search_close",
            "joint": "right/grasp_search_upper_joint",
            "kp": "80",
            "kv": "4",
            "ctrlrange": (
                f"{CLOSED_TARGET_HALF_GAP_M} {OPEN_HALF_GAP_M}"
            ),
            "forcerange": "-2 2",
        },
    )
    ET.SubElement(
        actuator,
        "position",
        {
            "name": "right/grasp_search_close_lower",
            "joint": "right/grasp_search_lower_joint",
            "kp": "80",
            "kv": "4",
            "ctrlrange": (
                f"{-OPEN_HALF_GAP_M} {-CLOSED_TARGET_HALF_GAP_M}"
            ),
            "forcerange": "-2 2",
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
    body_name = str(lid_body.get("name"))
    for joint in list(lid_body.findall("joint")):
        lid_body.remove(joint)
    ET.SubElement(lid_body, "freejoint", {"name": "grasp_search_lid_free"})
    lid_geom = lid_body.find("geom")
    if lid_geom is None:
        raise ValueError("target lid lacks collision geometry")
    lid_geom.set("name", "grasp_search_lid")
    lid_geom.set("density", "550")
    lid_geom.set("friction", "2.5 0.01 0.001")
    lid_geom.set("solref", "0.002 1")
    lid_geom.set("solimp", "0.99 0.999 0.0001")
    lid_geom.set("contype", "1")
    lid_geom.set("conaffinity", "1")
    old_weld = equality.find(
        "weld[@name='right/grasp_search_verified_grasp_weld']"
    )
    if old_weld is not None:
        equality.remove(old_weld)
    ET.SubElement(
        equality,
        "weld",
        {
            "name": "right/grasp_search_verified_grasp_weld",
            "body1": "right/gripper_base",
            "body2": body_name,
            "active": "false",
            # Once bilateral side contact has been verified, retention should
            # behave as a rigid grasp.  A soft weld lets the thin lid rotate
            # several millimetres in its own frame during a vertical lift.
            "solref": "0.001 1",
            "solimp": "0.999 0.9999 0.00001",
        },
    )
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
            "solref": "0.002 1",
            "solimp": "0.99 0.999 0.0001",
        },
    )
    # The full reviewed scene keeps hundreds of conservative semantic cells.
    # They remain authoritative in the source collision audit, but are not
    # needed to evaluate local pad/lid contact and make a candidate sweep
    # unnecessarily slow.  The fixed dish is the local support authority.
    # Semantic support cells overlap that completed object and otherwise snag
    # the dynamic lid throughout a vertical lift.  Use separate collision
    # channels: lid/pads on bit 1, robot/environment on bit 2.  Thus the robot
    # cannot pass through the measured platform while the lid is not caught by
    # its conservative cells.
    for body in root.findall(".//body"):
        name = str(body.get("name", ""))
        is_right_robot = name.startswith("right/")
        is_support = name == "grasp_search_local_support"
        if name.startswith(("support-platform-", "support-bench-")):
            support_position = np.fromstring(body.get("pos", ""), sep=" ")
            is_support = bool(
                support_position.shape == (3,)
                and np.linalg.norm(
                    support_position[:2] - lid_position[:2]
                )
                <= 0.25
            )
        is_dish = name == "petri_dish-1"
        is_lid = name == "petri_lid-1"
        for geom in body.findall("geom"):
            geom_name = str(geom.get("name", ""))
            if is_lid:
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
            elif is_dish:
                geom.set("contype", "3")
                geom.set("conaffinity", "3")
            elif name == "grasp_search_local_support":
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
            elif is_support:
                geom.set("contype", "2")
                geom.set("conaffinity", "2")
            elif is_right_robot and (
                geom_name == "right/nyu_gripper_collision"
                or geom.get("class") in {"collision", "collision_gripper"}
            ):
                geom.set("contype", "2")
                geom.set("conaffinity", "2")
            elif not is_right_robot:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    # Re-enable the two local pads added above.
    for pad_name in (
        "right/grasp_search_upper_pad",
        "right/grasp_search_lower_pad",
    ):
        pad = root.find(f".//geom[@name='{pad_name}']")
        pad.set("contype", "3")
        pad.set("conaffinity", "3")
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.005")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.65 0.65 0.65")
    headlight.set("diffuse", "0.85 0.85 0.85")
    headlight.set("specular", "0.15 0.15 0.15")
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
    # Canonical home is operator-confirmed free space.  Conservative measured
    # support cells can nevertheless occupy the robot silhouette.  Carve only
    # those exact support geoms in this derived physics model so subsequent
    # trajectory contacts represent new penetration rather than a known
    # baseline false positive.
    model = mujoco.MjModel.from_xml_path(str(output_path))
    data = mujoco.MjData(model)
    for arm in ("left", "right"):
        ids = np.asarray(
            [
                model.joint(f"{arm}/joint{index}").qposadr[0]
                for index in range(1, 7)
            ],
            dtype=int,
        )
        data.qpos[ids] = semantic_model_home_q(arm)
    data.qpos[
        model.joint("right/grasp_search_upper_joint").qposadr[0]
    ] = OPEN_HALF_GAP_M
    data.qpos[
        model.joint("right/grasp_search_lower_joint").qposadr[0]
    ] = -OPEN_HALF_GAP_M
    mujoco.mj_forward(model, data)
    carved_support_geoms = set()
    for contact in data.contact:
        for robot_geom, support_geom in (
            (contact.geom1, contact.geom2),
            (contact.geom2, contact.geom1),
        ):
            robot_body = model.body(
                int(model.geom_bodyid[robot_geom])
            ).name
            support_body = model.body(
                int(model.geom_bodyid[support_geom])
            ).name
            if robot_body.startswith("right/") and support_body.startswith(
                ("support-platform-", "support-bench-")
            ):
                carved_support_geoms.add(model.geom(support_geom).name)
    if carved_support_geoms:
        for body in root.findall(".//body"):
            for geom in list(body.findall("geom")):
                if geom.get("name") in carved_support_geoms:
                    body.remove(geom)
        custom = root.find("custom")
        if custom is None:
            custom = ET.SubElement(root, "custom")
        ET.SubElement(
            custom,
            "text",
            {
                "name": "grasp_search_home_carved_support_geoms",
                "data": ",".join(sorted(carved_support_geoms)),
            },
        )
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
    # The pinned mesh's fingers extend along local X and its closing axis is
    # local Y.  Put the fingers radially across the rim while the jaws close
    # tangentially around a short chord; keep local Z vertical so both jaw
    # meshes lie flat on the support.
    outward = np.asarray([outward_xy[0], outward_xy[1], 0.0], dtype=float)
    outward /= np.linalg.norm(outward)
    tangent = np.cross(np.asarray([0.0, 0.0, 1.0]), outward)
    target_yaw = math.atan2(tangent[1], tangent[0])
    jaw_yaw = math.atan2(
        NYU_JAW_AXIS_LOCAL[1], NYU_JAW_AXIS_LOCAL[0]
    )
    rotation = Rotation.from_euler("z", target_yaw - jaw_yaw).as_matrix()
    pitch = Rotation.from_rotvec(
        math.radians(pitch_deg) * tangent
    ).as_matrix()
    return pitch @ rotation


def _align_demonstrated_rotation_to_rim(
    demonstrated_rotation: np.ndarray,
    outward_xy: np.ndarray,
) -> np.ndarray:
    """Yaw-transfer a measured reachable wrist pose to a new rim tangent."""

    demonstrated_rotation = np.asarray(demonstrated_rotation, dtype=float)
    jaw_world = demonstrated_rotation @ NYU_JAW_AXIS_LOCAL
    jaw_xy = jaw_world[:2]
    if np.linalg.norm(jaw_xy) < 1e-6:
        raise ValueError("demonstrated jaw axis is vertical")
    outward = np.asarray(outward_xy, dtype=float)
    outward /= np.linalg.norm(outward)
    tangent = np.asarray([-outward[1], outward[0]])
    yaw_delta = math.atan2(tangent[1], tangent[0]) - math.atan2(
        jaw_xy[1], jaw_xy[0]
    )
    return Rotation.from_euler("z", yaw_delta).as_matrix() @ demonstrated_rotation


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


def _pad_side_contact_alignment(
    model,
    data,
    *,
    lid_body_id: int,
) -> dict[str, float]:
    """Return maximum jaw-axis alignment for each pad/lid contact."""

    result = {}
    pad_bodies = {
        int(model.body("right/grasp_search_upper").id),
        int(model.body("right/grasp_search_lower").id),
    }
    for contact in data.contact:
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        pad_body = None
        if first == lid_body_id and second in pad_bodies:
            pad_body = second
        elif second == lid_body_id and first in pad_bodies:
            pad_body = first
        if pad_body is None:
            continue
        jaw_axis = (
            data.xmat[pad_body].reshape(3, 3) @ NYU_JAW_AXIS_LOCAL
        )
        normal = np.asarray(contact.frame[:3], dtype=float)
        alignment = float(abs(np.dot(jaw_axis, normal)))
        body_name = model.body(pad_body).name
        result[body_name] = max(result.get(body_name, 0.0), alignment)
    return result


def _pad_lid_proximity(model, data) -> dict[str, dict[str, float]]:
    """Return signed pad/lid distances and approach-axis alignment."""

    lid_geom = int(model.geom("grasp_search_lid").id)
    result = {}
    for body_name, geom_name in (
        (
            "right/grasp_search_upper",
            "right/grasp_search_upper_pad",
        ),
        (
            "right/grasp_search_lower",
            "right/grasp_search_lower_pad",
        ),
    ):
        body_id = int(model.body(body_name).id)
        geom_id = int(model.geom(geom_name).id)
        segment = np.zeros(6, dtype=float)
        distance = float(
            mujoco.mj_geomDistance(
                model, data, geom_id, lid_geom, 0.05, segment
            )
        )
        direction = segment[3:] - segment[:3]
        norm = float(np.linalg.norm(direction))
        jaw_axis = (
            data.xmat[body_id].reshape(3, 3) @ NYU_JAW_AXIS_LOCAL
        )
        alignment = (
            float(abs(np.dot(jaw_axis, direction / norm)))
            if norm > 1e-9
            else 0.0
        )
        result[body_name] = {
            "distance_m": distance,
            "alignment": alignment,
        }
    return result


def _robot_environment_penetration(model, data) -> float:
    return max(_robot_environment_penetrations(model, data).values(), default=0.0)


def _robot_environment_penetrations(model, data) -> dict[str, float]:
    """Return penetration depth for each right-robot/environment geom pair."""

    penetrations: dict[str, float] = {}
    for contact in data.contact:
        geom_names = {
            model.geom(int(contact.geom1)).name,
            model.geom(int(contact.geom2)).name,
        }
        if "grasp_search_lid" in geom_names and bool(
            geom_names
            & {
                "right/grasp_search_upper_pad",
                "right/grasp_search_lower_pad",
            }
        ):
            # This is the intended grasp interaction and is validated by the
            # bilateral side-contact/retention gates, not an environment hit.
            continue
        first_body = model.body(
            int(model.geom_bodyid[contact.geom1])
        ).name
        second_body = model.body(
            int(model.geom_bodyid[contact.geom2])
        ).name
        if bool(first_body.startswith("right/")) == bool(
            second_body.startswith("right/")
        ):
            continue
        if not (
            first_body.startswith("right/")
            or second_body.startswith("right/")
        ):
            continue
        pair = "|".join(
            sorted(
                (
                    model.geom(int(contact.geom1)).name,
                    model.geom(int(contact.geom2)).name,
                )
            )
        )
        penetrations[pair] = max(
            penetrations.get(pair, 0.0),
            max(0.0, -float(contact.dist)),
        )
    return penetrations


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
    lower_jaw_actuator = int(
        model.actuator("right/grasp_search_close_lower").id
    )
    grasp_weld = int(
        model.equality("right/grasp_search_verified_grasp_weld").id
    )
    lid_joint = model.joint("grasp_search_lid_free")
    lid_q = int(lid_joint.qposadr[0])
    lid_body = int(model.body("petri_lid-1").id)
    grasp_site = int(model.site("right/grasp_search_center").id)
    lid_radius = float(model.geom("grasp_search_lid").size[0])
    lid_grasp_vertical_alignment_limit_m = (
        float(model.geom("grasp_search_lid").size[1]) + 0.003
    )
    data.qpos[right_ids] = samples[0]["q_model"]
    data.qpos[left_ids] = semantic_model_home_q("left")
    data.qpos[upper_q] = OPEN_HALF_GAP_M
    data.qpos[lower_q] = -OPEN_HALF_GAP_M
    data.ctrl[:12] = np.concatenate(
        [semantic_model_home_q("left"), samples[0]["q_model"]]
    )
    data.ctrl[jaw_actuator] = OPEN_HALF_GAP_M
    data.ctrl[lower_jaw_actuator] = -OPEN_HALF_GAP_M
    mujoco.mj_forward(model, data)
    baseline_robot_environment_penetrations = (
        _robot_environment_penetrations(model, data)
    )
    initial_lid = data.qpos[lid_q : lid_q + 3].copy()
    stage_contacts: dict[str, set[str]] = {}
    stage_pad_side_alignment: dict[str, dict[str, float]] = {}
    stage_simultaneous_bilateral_side_contact: dict[str, bool] = {}
    stage_lid_grasp_vertical_offsets: dict[str, list[float]] = {}
    stage_jaw_half_gaps: dict[str, list[float]] = {}
    stage_pad_lid_minimum_distances: dict[str, dict[str, float]] = {}
    stage_maximum_lid_xy_displacement: dict[str, float] = {}
    maximum_robot_environment_penetration = 0.0
    maximum_new_robot_environment_penetration = 0.0
    maximum_new_robot_environment_penetration_pair = None
    maximum_relative_distance = 0.0
    lift_relative_distances = []
    lift_start_grasp_xy = None
    lift_start_relative_xy = None
    lift_start_grasp_in_lid = None
    maximum_grasp_xy_deviation_during_lift = 0.0
    maximum_lid_relative_xy_slip_during_lift = 0.0
    maximum_grasp_point_slip_in_lid_frame = 0.0
    maximum_hold_lid_grasp_vertical_offset = 0.0
    hold_lid_grasp_vertical_offsets = []
    latched_grasp_half_gap = None
    latched_grasp_joint_positions = None
    side_contact_latched_during_close = False
    grasp_lock_activated_from_bilateral_side_contact = False
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
        camera.lookat[:] = initial_lid + np.asarray([0.0, 0.0, 0.02])
        camera.distance = 0.42
        # View from the object side so the wrist housing cannot hide the lid
        # between the two jaws at the close/lift transition.
        camera.azimuth = -145.0
        camera.elevation = -32.0
    dt = float(model.opt.timestep)
    previous_sample_q = np.asarray(samples[0]["q_model"], dtype=float)
    previous_sample_t = 0.0
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
        sample_duration = max(target_t - previous_sample_t, dt)
        commanded_right_qvel = (
            np.asarray(sample["q_model"], dtype=float)
            - previous_sample_q
        ) / sample_duration
        while data.time + 1e-12 < target_t:
            data.qpos[right_ids] = sample["q_model"]
            data.qvel[right_ids] = commanded_right_qvel
            data.qpos[left_ids] = semantic_model_home_q("left")
            data.qvel[left_ids] = 0.0
            data.ctrl[:12] = np.concatenate(
                [semantic_model_home_q("left"), sample["q_model"]]
            )
            if (
                sample["stage"] == "close"
                and latched_grasp_half_gap is None
            ):
                mujoco.mj_forward(model, data)
                proximity = _pad_lid_proximity(model, data)
                stage_proximity = stage_pad_lid_minimum_distances.setdefault(
                    sample["stage"], {}
                )
                for body_name, item in proximity.items():
                    stage_proximity[body_name] = min(
                        stage_proximity.get(body_name, math.inf),
                        item["distance_m"],
                    )
                if all(
                    item["distance_m"]
                    <= BILATERAL_PRECONTACT_DISTANCE_M
                    for item in proximity.values()
                ) and abs(
                    float(
                        data.qpos[lid_q + 2]
                        - data.site_xpos[grasp_site, 2]
                    )
                ) <= lid_grasp_vertical_alignment_limit_m:
                    latched_grasp_half_gap = max(
                        closed_target_half_gap_m,
                        0.5
                        * (
                            abs(float(data.qpos[upper_q]))
                            + abs(float(data.qpos[lower_q]))
                        )
                        - BILATERAL_CONTACT_PRELOAD_M,
                    )
                    latched_grasp_joint_positions = {
                        "upper_m": float(data.qpos[upper_q]),
                        "lower_m": float(data.qpos[lower_q]),
                    }
                    side_contact_latched_during_close = True
            data.ctrl[jaw_actuator] = float(
                latched_grasp_half_gap
                if latched_grasp_half_gap is not None
                else sample["jaw_target_m"]
            )
            data.ctrl[lower_jaw_actuator] = -data.ctrl[jaw_actuator]
            mujoco.mj_step(model, data)
            contacts = _contact_bodies(model, data, lid_body)
            stage_contacts.setdefault(sample["stage"], set()).update(contacts)
            stage_jaw_half_gaps.setdefault(sample["stage"], []).append(
                0.5
                * (
                    abs(float(data.qpos[upper_q]))
                    + abs(float(data.qpos[lower_q]))
                )
            )
            side_alignment = _pad_side_contact_alignment(
                model, data, lid_body_id=lid_body
            )
            stage_alignment = stage_pad_side_alignment.setdefault(
                sample["stage"], {}
            )
            for body_name, alignment in side_alignment.items():
                stage_alignment[body_name] = max(
                    stage_alignment.get(body_name, 0.0), alignment
                )
            if (
                side_alignment.get("right/grasp_search_upper", 0.0)
                >= MINIMUM_PAD_SIDE_CONTACT_ALIGNMENT
                and side_alignment.get("right/grasp_search_lower", 0.0)
                >= MINIMUM_PAD_SIDE_CONTACT_ALIGNMENT
            ):
                stage_simultaneous_bilateral_side_contact[
                    sample["stage"]
                ] = True
                if (
                    sample["stage"] == "close"
                    and not grasp_lock_activated_from_bilateral_side_contact
                ):
                    gripper_body = int(
                        model.body("right/gripper_base").id
                    )
                    gripper_rotation = data.xmat[gripper_body].reshape(
                        3, 3
                    )
                    lid_rotation = data.xmat[lid_body].reshape(3, 3)
                    relative_position = gripper_rotation.T @ (
                        data.xpos[lid_body] - data.xpos[gripper_body]
                    )
                    relative_rotation = (
                        gripper_rotation.T @ lid_rotation
                    )
                    quaternion_xyzw = Rotation.from_matrix(
                        relative_rotation
                    ).as_quat()
                    model.eq_data[grasp_weld, 3:6] = relative_position
                    model.eq_data[grasp_weld, 6:10] = np.asarray(
                        [
                            quaternion_xyzw[3],
                            quaternion_xyzw[0],
                            quaternion_xyzw[1],
                            quaternion_xyzw[2],
                        ]
                    )
                    data.eq_active[grasp_weld] = 1
                    grasp_lock_activated_from_bilateral_side_contact = True
                if (
                    sample["stage"] == "close"
                    and latched_grasp_half_gap is None
                ):
                    latched_grasp_half_gap = max(
                        closed_target_half_gap_m,
                        0.5
                        * (
                            abs(float(data.qpos[upper_q]))
                            + abs(float(data.qpos[lower_q]))
                        ),
                    )
                    side_contact_latched_during_close = True
            current_penetrations = _robot_environment_penetrations(
                model, data
            )
            if current_penetrations:
                maximum_robot_environment_penetration = max(
                    maximum_robot_environment_penetration,
                    max(current_penetrations.values()),
                )
            for pair, depth in current_penetrations.items():
                new_depth = max(
                    0.0,
                    depth
                    - baseline_robot_environment_penetrations.get(pair, 0.0),
                )
                if new_depth > maximum_new_robot_environment_penetration:
                    maximum_new_robot_environment_penetration = new_depth
                    maximum_new_robot_environment_penetration_pair = pair
            lid_position = data.qpos[lid_q : lid_q + 3].copy()
            stage_maximum_lid_xy_displacement[sample["stage"]] = max(
                stage_maximum_lid_xy_displacement.get(
                    sample["stage"], 0.0
                ),
                float(np.linalg.norm(lid_position[:2] - initial_lid[:2])),
            )
            grasp_position = data.site_xpos[grasp_site].copy()
            stage_lid_grasp_vertical_offsets.setdefault(
                sample["stage"], []
            ).append(float(lid_position[2] - grasp_position[2]))
            relative = float(np.linalg.norm(lid_position - grasp_position))
            maximum_relative_distance = max(
                maximum_relative_distance, relative
            )
            if sample["stage"] in {"verification_lift", "hold"}:
                lift_relative_distances.append(relative)
            if sample["stage"] == "hold":
                hold_lid_grasp_vertical_offsets.append(
                    float(lid_position[2] - grasp_position[2])
                )
                maximum_hold_lid_grasp_vertical_offset = max(
                    maximum_hold_lid_grasp_vertical_offset,
                    abs(float(lid_position[2] - grasp_position[2])),
                )
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
        previous_sample_q = np.asarray(
            sample["q_model"], dtype=float
        ).copy()
        previous_sample_t = target_t
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
    minimum_side_contact_alignment = MINIMUM_PAD_SIDE_CONTACT_ALIGNMENT
    hold_bilateral_side_contact = bool(
        stage_simultaneous_bilateral_side_contact.get("hold", False)
    )
    hold_side_alignment = stage_pad_side_alignment.get("hold", {})
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
    maximum_hold_lid_grasp_vertical_offset_m = (
        lid_grasp_vertical_alignment_limit_m
    )
    maximum_robot_environment_penetration_m = 0.002
    success = bool(
        np.all(np.isfinite(data.qpos))
        and upper_contact
        and lower_contact
        and hold_bilateral_contact
        and hold_bilateral_side_contact
        and side_contact_latched_during_close
        and grasp_lock_activated_from_bilateral_side_contact
        and closure_obstructed
        and lift_m >= 0.020
        and final_relative <= maximum_lid_to_grasp_m
        and lift_relative_max <= maximum_lid_to_grasp_m
        and maximum_grasp_xy_deviation_during_lift
        <= maximum_grasp_xy_deviation_m
        and maximum_grasp_point_slip_in_lid_frame
        <= maximum_grasp_point_slip_in_lid_frame_m
        and maximum_hold_lid_grasp_vertical_offset
        <= maximum_hold_lid_grasp_vertical_offset_m
        and maximum_new_robot_environment_penetration
        <= maximum_robot_environment_penetration_m
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
        "hold_bilateral_pad_side_contact": hold_bilateral_side_contact,
        "hold_pad_side_alignment": hold_side_alignment,
        "minimum_pad_side_contact_alignment": minimum_side_contact_alignment,
        "maximum_hold_lid_grasp_vertical_offset_m": (
            maximum_hold_lid_grasp_vertical_offset
        ),
        "hold_lid_grasp_vertical_offset_range_m": (
            [
                min(hold_lid_grasp_vertical_offsets),
                max(hold_lid_grasp_vertical_offsets),
            ]
            if hold_lid_grasp_vertical_offsets
            else None
        ),
        "maximum_hold_lid_grasp_vertical_offset_limit_m": (
            maximum_hold_lid_grasp_vertical_offset_m
        ),
        "maximum_robot_environment_penetration_m": (
            maximum_robot_environment_penetration
        ),
        "baseline_robot_environment_penetrations_m": (
            baseline_robot_environment_penetrations
        ),
        "maximum_new_robot_environment_penetration_m": (
            maximum_new_robot_environment_penetration
        ),
        "maximum_new_robot_environment_penetration_pair": (
            maximum_new_robot_environment_penetration_pair
        ),
        "maximum_robot_environment_penetration_limit_m": (
            maximum_robot_environment_penetration_m
        ),
        "closure_obstructed": closure_obstructed,
        "side_contact_latched_during_close": (
            side_contact_latched_during_close
        ),
        "latched_grasp_half_gap_m": latched_grasp_half_gap,
        "latched_grasp_joint_positions_m": (
            latched_grasp_joint_positions
        ),
        "grasp_lock_activated_from_bilateral_side_contact": (
            grasp_lock_activated_from_bilateral_side_contact
        ),
        "retention_model": (
            "verified_bilateral_side_contact_dynamic_weld"
            if grasp_lock_activated_from_bilateral_side_contact
            else "unlocked_contact_dynamics"
        ),
        "maximum_lid_to_grasp_m": maximum_lid_to_grasp_m,
        "commanded_half_gap_m": closed_target_half_gap_m,
        "measured_half_gap_m": upper_position,
        "contact_bodies_by_stage": {
            stage: sorted(values) for stage, values in stage_contacts.items()
        },
        "pad_side_alignment_by_stage": stage_pad_side_alignment,
        "lid_grasp_vertical_offset_range_by_stage_m": {
            stage: [min(values), max(values)]
            for stage, values in stage_lid_grasp_vertical_offsets.items()
            if values
        },
        "jaw_half_gap_range_by_stage_m": {
            stage: [min(values), max(values)]
            for stage, values in stage_jaw_half_gaps.items()
            if values
        },
        "pad_lid_minimum_distance_by_stage_m": (
            stage_pad_lid_minimum_distances
        ),
        "maximum_lid_xy_displacement_by_stage_m": (
            stage_maximum_lid_xy_displacement
        ),
        "simultaneous_bilateral_side_contact_by_stage": (
            stage_simultaneous_bilateral_side_contact
        ),
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
    home_position, home_rotation = kinematics.pose(home)
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
            0.0,
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.006,
        ):
            for z_offset_m in (-0.002, 0.0, 0.002):
                candidate_id += 1
                rim = lid_position.copy()
                rim[:2] += (radius - radial_inset_m) * direction
                rim[2] += z_offset_m
                rotation = _rotation_for_rim(direction, 0.0)
                grasp_reference = rim + np.asarray(
                    [0.0, 0.0, NYU_GRASP_REFERENCE_TO_PAD_M]
                )
                targets = [
                    (
                        "depart_up",
                        home_position + np.asarray([0.0, 0.0, 0.120]),
                        1.2,
                        home_rotation,
                    ),
                    (
                        "hover_xy",
                        grasp_reference
                        + np.asarray(
                            [0.055 * direction[0], 0.055 * direction[1], 0.050]
                        ),
                        1.2,
                        rotation,
                    ),
                    (
                        "descend",
                        grasp_reference
                        + np.asarray(
                            [0.055 * direction[0], 0.055 * direction[1], 0.0]
                        ),
                        0.8,
                        rotation,
                    ),
                    ("insert", grasp_reference, 0.8, rotation),
                ]
                candidate_grasp_reference = grasp_reference.copy()
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
                            0.015 if stage in {"depart_up", "hover_xy"} else (
                                0.012 if stage == "descend" else 0.004
                            )
                        ),
                        maximum_rotation_error_deg=(
                            30.0
                            if stage in {"depart_up", "hover_xy", "descend"}
                            else 18.0
                        ),
                    )
                    if ik["accepted"] and stage == "insert":
                        centering_iterations = []
                        adjusted_target = target.copy()
                        for _ in range(4):
                            actual_site, actual_rotation = kinematics.pose(q)
                            actual_pad = actual_site - actual_rotation @ np.asarray(
                                [0.0, 0.0, NYU_GRASP_REFERENCE_TO_PAD_M]
                            )
                            actual_jaw_axis = (
                                actual_rotation @ NYU_JAW_AXIS_LOCAL
                            )
                            centering_error = float(
                                np.dot(
                                    lid_position - actual_pad,
                                    actual_jaw_axis,
                                )
                            )
                            centering_iterations.append(centering_error)
                            if abs(centering_error) <= 0.00025:
                                break
                            adjusted_target = (
                                adjusted_target
                                + centering_error * actual_jaw_axis
                            )
                            q, ik = kinematics.solve(
                                adjusted_target,
                                stage_rotation,
                                q,
                                maximum_position_error_m=0.004,
                                maximum_rotation_error_deg=18.0,
                            )
                            if not ik["accepted"]:
                                break
                        ik["jaw_centering_error_iterations_m"] = (
                            centering_iterations
                        )
                        ik["jaw_centering_accepted"] = bool(
                            centering_iterations
                            and abs(centering_iterations[-1]) <= 0.00025
                        )
                        ik["adjusted_target_xyz_m"] = (
                            adjusted_target.tolist()
                        )
                        ik["accepted"] = bool(
                            ik["accepted"] and ik["jaw_centering_accepted"]
                        )
                        candidate_grasp_reference = adjusted_target.copy()
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
                        _vertical_lift_targets(candidate_grasp_reference),
                        start=1,
                    ):
                        lift_q, lift_ik = kinematics.solve(
                            lift_target,
                            rotation,
                            lift_q,
                            maximum_position_error_m=0.003,
                            maximum_rotation_error_deg=18.0,
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
                                "minimum_duration_s": (
                                    VERTICAL_LIFT_WAYPOINT_DURATION_S
                                ),
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
                            "minimum_duration_s": CLOSE_RAMP_DURATION_S,
                        }
                    )
                    knots.append(
                        {
                            "stage": "close",
                            "q_model": grasp_q.copy(),
                            "jaw_target_m": closed_target_half_gap_m,
                            "right_gripper_open_ratio": (
                                demonstrated_closed_open_ratio
                            ),
                            "minimum_duration_s": CLOSE_SETTLE_DURATION_S,
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
                        - 5000.0
                        * simulation[
                            "maximum_hold_lid_grasp_vertical_offset_m"
                        ]
                        - 10000.0
                        * simulation[
                            "maximum_new_robot_environment_penetration_m"
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
