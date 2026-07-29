#!/usr/bin/env python3
"""Complete accepted SAM/RGB-D surfaces into estimated closed 3D objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
import plotly.graph_objects as go

from export_sam_objects_3d import COLORS, _compact_mesh  # type: ignore
from render_mujoco_mobile import _box, _cylinder  # type: ignore


def _primitive(kind, size, center, yaw=0.0):
    if kind == "box":
        vertices, faces = _box(size)
    else:
        vertices, faces = _cylinder(size[0], size[1], segments=28)
    rotation = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return (rotation @ vertices.T).T + center, faces


def _arm_mesh(model_path, q, side):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_prefix = "left_" if side == "left" else ""
    for index, value in enumerate(q[:5], 1):
        joint = model.joint(f"{joint_prefix}joint{index}")
        data.qpos[int(joint.qposadr[0])] = value
    mujoco.mj_forward(model, data)
    base_name = "left_base_link" if side == "left" else "base_link"
    base_id = int(model.body(base_name).id)
    base = data.body(base_name).xpos.copy()

    def belongs_to_arm(body_id):
        while body_id:
            if body_id == base_id:
                return True
            body_id = int(model.body_parentid[body_id])
        return False

    parts = []
    for geom_id in range(model.ngeom):
        if (
            not belongs_to_arm(int(model.geom_bodyid[geom_id]))
            or model.geom_group[geom_id] != 2
        ):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            continue
        va = int(model.mesh_vertadr[mesh_id])
        vn = int(model.mesh_vertnum[mesh_id])
        fa = int(model.mesh_faceadr[mesh_id])
        fn = int(model.mesh_facenum[mesh_id])
        part = np.asarray(model.mesh_vert[va : va + vn], float)
        # mesh_vert is already normalized into MuJoCo's runtime mesh frame.
        # Applying mesh_pos/mesh_quat again detaches links and grippers.
        geom_rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        part = (geom_rotation @ part.T).T + data.geom_xpos[geom_id] - base
        part_faces = np.asarray(model.mesh_face[fa : fa + fn], int)
        parts.append((part, part_faces))
    return parts


def _mesh_trace(vertices, faces, name, color, opacity=1.0):
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=name, color=color, opacity=opacity, flatshading=False,
        hovertemplate=name + "<extra></extra>", showlegend=True,
    )


def _safe_mesh_traces(vertices, faces, name, color, opacity=1.0):
    """Split meshes below mobile WebGL's practical vertex-index limit."""
    pending = [(vertices, faces)]
    chunks = []
    while pending:
        chunk_vertices, chunk_faces = pending.pop()
        if len(chunk_vertices) <= 50_000:
            chunks.append((chunk_vertices, chunk_faces))
            continue
        midpoint = len(chunk_faces) // 2
        for face_half in (chunk_faces[:midpoint], chunk_faces[midpoint:]):
            compact_vertices, compact_faces = _compact_mesh(
                chunk_vertices, face_half
            )
            pending.append((compact_vertices, compact_faces))
    traces = []
    for index, (chunk_vertices, chunk_faces) in enumerate(chunks):
        trace = _mesh_trace(
            chunk_vertices, chunk_faces, name, color, opacity
        )
        trace.showlegend = index == 0
        traces.append(trace)
    return traces


def _obj_bounds(path):
    vertices = np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in Path(path).read_text().splitlines()
            if line.startswith("v ")
        ],
        dtype=float,
    )
    return vertices.min(axis=0), vertices.max(axis=0)


def complete(args):
    archive = np.load(args.mesh)
    vertices = np.asarray(archive["vertices_xyz_m"], float)
    faces = np.asarray(archive["faces"], int)
    valid = np.asarray(archive["valid_vertex_mask"], bool)
    shape = (192, 256)

    def mask(path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA) > 100

    masks = {
        "robot": mask(args.robot_mask),
        "microscope": mask(args.microscope_mask),
        "incubator": mask(args.incubator_mask),
        "bottle": mask(args.bottle_mask),
    }
    points = {
        name: vertices[value.reshape(-1) & valid] for name, value in masks.items()
    }
    calibration = json.loads(Path(args.calibration_report).read_text())
    camera = np.asarray(calibration["T_level_camera"], float)[:3, 3]
    support_candidates = []
    for path in args.platform_obj:
        support_low, support_high = _obj_bounds(path)
        support_candidates.append(
            (
                float(np.linalg.norm((support_low + support_high) / 2 - camera)),
                path,
                support_low,
                support_high,
            )
        )
    support_candidates.sort(key=lambda candidate: candidate[0])
    (_, front_path, front_low, front_high), (
        _,
        rear_path,
        rear_low,
        rear_high,
    ) = support_candidates
    traces, report = [], {"schema": "piper_robot.sam_object_completion/v1", "objects": {}}

    # Preserve the repository MuJoCo kinematics exactly.  RGB-D determines
    # only the two base positions and their common upright yaw; it must never
    # independently rotate articulated arms to chase partial SAM surfaces.
    manifest = json.loads((Path(args.capture) / "manifest.json").read_text())
    state = manifest["robot_state"]["after"]
    q_by_name = {
        "left_robot": np.asarray(state["left_joint_positions_rad"], float),
        "right_robot": np.asarray(state["right_joint_positions_rad"], float),
    }
    cad_parts_by_name = {
        name: _arm_mesh(
            args.robot_model,
            q,
            "left" if name == "left_robot" else "right",
        )
        for name, q in q_by_name.items()
    }
    base_by_name = {
        "left_robot": np.asarray(
            calibration["anchor_xyz_level_m"]["left_piper_base"], float
        ),
        "right_robot": np.asarray(
            calibration["anchor_xyz_level_m"]["right_piper_base"], float
        ),
    }
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(
            masks["robot"].astype(np.uint8), 8
        )
    )
    component_ids = sorted(
        range(1, component_count),
        key=lambda index: component_stats[index, cv2.CC_STAT_AREA],
        reverse=True,
    )[:2]
    component_points = [
        vertices[(component_labels == index).reshape(-1) & valid]
        for index in component_ids
    ]
    remaining = list(component_points)
    for name in ("right_robot", "left_robot"):
        anchor = base_by_name[name]
        selected_index = min(
            range(len(remaining)),
            key=lambda index: float(
                np.linalg.norm(
                    np.median(remaining[index][:, :2], axis=0) - anchor[:2]
                )
            ),
        )
        selected = remaining.pop(selected_index)
        # The base mesh bottom is z=0 in the MacBook MJCF.
        anchor[2] = float(np.quantile(selected[:, 2], 0.01))
    base_vector = base_by_name["left_robot"][:2] - base_by_name["right_robot"][:2]
    # The source MJCF has right->left along +Y.
    base_yaw = float(np.arctan2(base_vector[1], base_vector[0]) - np.pi / 2)
    base_rotation = np.array(
        [
            [np.cos(base_yaw), -np.sin(base_yaw), 0],
            [np.sin(base_yaw), np.cos(base_yaw), 0],
            [0, 0, 1],
        ]
    )
    for name in ("left_robot", "right_robot"):
        cad_parts = cad_parts_by_name[name]
        translation = base_by_name[name]
        completed_parts = [
            ((base_rotation @ part.T).T + translation, part_faces)
            for part, part_faces in cad_parts
        ]
        robot_traces = []
        for completed, cad_faces in completed_parts:
            robot_traces.extend(
                _safe_mesh_traces(
                    completed, cad_faces, name, COLORS["robot"]
                )
            )
        for index, trace in enumerate(robot_traces):
            trace.showlegend = index == 0
        traces.extend(robot_traces)
        report["objects"][name] = {
            "completion": "exact_mujoco_forward_kinematics_at_synchronized_joint_state",
            "model_source": str(Path(args.robot_model).resolve()),
            "base_position_source": "rgbd_base_anchor",
            "base_xyz_level_m": translation.tolist(),
            "shared_upright_yaw_deg": float(np.rad2deg(base_yaw)),
            "joint_mapping": "MacBook MJCF joint1..joint5; synchronized joint6 unavailable in that model",
            "mobile_mesh_trace_count": len(robot_traces),
            "maximum_trace_vertices": max(
                len(trace.x) for trace in robot_traces
            ),
        }

    # Robust XY orientation shared by the fixture-style completions.
    def yaw_and_bounds(object_points):
        xy = object_points[:, :2] - np.median(object_points[:, :2], axis=0)
        _, _, vt = np.linalg.svd(xy)
        yaw = float(np.arctan2(vt[0, 1], vt[0, 0]))
        low, high = np.quantile(object_points, [0.02, 0.98], axis=0)
        return yaw, low, high

    yaw, low, high = yaw_and_bounds(points["incubator"])
    center = (low + high) / 2
    center[2] = float(rear_high[2] + 0.175)
    incubator_center = center.copy()
    incubator_parts = [
        ("incubator_body", "box", np.array([0.14, 0.125, 0.175]), center, yaw),
        ("incubator_door", "box", np.array([0.132, 0.006, 0.17]),
         center + np.array([0.10 * np.cos(yaw), 0.10 * np.sin(yaw), 0]), yaw + 1.25),
    ]
    for name, kind, size, part_center, angle in incubator_parts:
        v, f = _primitive(kind, size, part_center, angle)
        traces.append(_mesh_trace(v, f, name, COLORS["incubator"], 0.9))
    report["objects"]["incubator"] = {
        "completion": "oriented_280x250x350mm_incubator_plus_open_door",
        "center_level_m": center.tolist(), "yaw_deg": np.rad2deg(yaw),
    }

    yaw, low, high = yaw_and_bounds(points["microscope"])
    center = (low + high) / 2
    z0, z1 = low[2], high[2]
    microscope_parts = [
        ("microscope_base", "box", [0.13, 0.10, 0.025], [center[0], center[1], z0 + 0.025], yaw),
        ("microscope_stage", "box", [0.11, 0.09, 0.018], [center[0], center[1], z0 + 0.17], yaw),
        ("microscope_column", "box", [0.035, 0.045, (z1-z0)*0.38],
         [center[0]+0.08*np.cos(yaw), center[1]+0.08*np.sin(yaw), (z0+z1)/2], yaw),
        ("microscope_head", "box", [0.10, 0.07, 0.055], [center[0], center[1], z1-0.06], yaw),
        ("microscope_eyepiece_1", "cylinder", [0.018, 0.07],
         [center[0]-0.035*np.sin(yaw), center[1]+0.035*np.cos(yaw), z1+0.04], yaw),
        ("microscope_eyepiece_2", "cylinder", [0.018, 0.07],
         [center[0]+0.035*np.sin(yaw), center[1]-0.035*np.cos(yaw), z1+0.04], yaw),
    ]
    for name, kind, size, part_center, angle in microscope_parts:
        v, f = _primitive(kind, np.asarray(size), np.asarray(part_center), angle)
        traces.append(_mesh_trace(v, f, name, COLORS["microscope"], 0.9))
    report["objects"]["microscope"] = {
        "completion": "semantic_multi_primitive_microscope", "center_level_m": center.tolist()
    }

    label = np.median(points["bottle"], axis=0)
    away = label[:2] - camera[:2]
    away /= np.linalg.norm(away)
    bottle_center = label.copy()
    bottle_center[:2] += 0.0375 * away
    v, f = _primitive("cylinder", np.array([0.0375, 0.08]), bottle_center)
    traces.append(_mesh_trace(v, f, "culture_media_bottle", "#ff8c1a", 0.9))
    report["objects"]["culture_media_bottle"] = {
        "completion": "75mm_diameter_160mm_tall_cylinder_from_visible_label",
        "center_level_m": bottle_center.tolist(),
    }

    # The two disconnected, same-height support surfaces are the front dish
    # platform and the rear incubator platform.  Depth from the RGB-D camera
    # determines front/rear, avoiding an image-left/right assumption.
    dish_xy = np.asarray(
        calibration["anchor_xyz_level_m"]["petri_dish"], dtype=float
    )[:2]
    front_low[:2] = np.minimum(front_low[:2], dish_xy - 0.065)
    front_high[:2] = np.maximum(front_high[:2], dish_xy + 0.065)
    # Ensure the completed incubator footprint is fully supported.
    rear_low[:2] = np.minimum(
        rear_low[:2], incubator_center[:2] - np.array([0.16, 0.15])
    )
    rear_high[:2] = np.maximum(
        rear_high[:2], incubator_center[:2] + np.array([0.16, 0.15])
    )
    platform_records = []
    for name, path, low, high in (
        ("front_dish_platform", front_path, front_low, front_high),
        ("rear_incubator_platform", rear_path, rear_low, rear_high),
    ):
        top = float(high[2])
        half_size = np.r_[(high[:2] - low[:2]) / 2, 0.025]
        platform_center = np.r_[(high[:2] + low[:2]) / 2, top - 0.025]
        v, f = _primitive("box", half_size, platform_center)
        traces.append(_mesh_trace(v, f, name, "#343b46", 1.0))
        platform_records.append(
            {
                "name": name,
                "source_surface": str(Path(path).resolve()),
                "center_level_m": platform_center.tolist(),
                "size_xyz_m": (2 * half_size).tolist(),
                "top_height_m": top,
            }
        )

    front_top = platform_records[0]["top_height_m"]
    dish_center = np.r_[dish_xy, front_top + 0.007]
    lid_center = np.r_[dish_xy, front_top + 0.017]
    for name, radius, half_height, object_center, color in (
        ("petri_dish", 0.045, 0.007, dish_center, "#70b7e6"),
        ("petri_lid", 0.047, 0.003, lid_center, "#4f9dd1"),
    ):
        v, f = _primitive(
            "cylinder", np.array([radius, half_height]), object_center
        )
        traces.append(_mesh_trace(v, f, name, color, 0.58))
    report["objects"]["platforms"] = platform_records
    report["objects"]["petri_dish"] = {
        "completion": "transparent_90mm_cylinder_on_front_platform",
        "center_level_m": dish_center.tolist(),
    }
    report["objects"]["petri_lid"] = {
        "completion": "transparent_94mm_lid_on_dish",
        "center_level_m": lid_center.tolist(),
    }

    # Keep measured surfaces as a translucent accuracy reference.
    for name, object_mask in masks.items():
        vertex_mask = object_mask.reshape(-1) & valid
        selected = faces[np.all(vertex_mask[faces], axis=1)]
        if len(selected):
            traces.append(_mesh_trace(vertices, selected, f"observed_{name}", "#243447", 0.16))
    arm_positions = np.asarray(
        [
            report["objects"]["left_robot"]["base_xyz_level_m"],
            report["objects"]["right_robot"]["base_xyz_level_m"],
        ]
    )
    arm_midpoint = np.mean(arm_positions[:, :2], axis=0)
    arm_baseline = arm_positions[0, :2] - arm_positions[1, :2]
    arm_baseline /= np.linalg.norm(arm_baseline)
    perpendiculars = (
        np.array([-arm_baseline[1], arm_baseline[0]]),
        np.array([arm_baseline[1], -arm_baseline[0]]),
    )
    incubator_hint = incubator_center[:2] - arm_midpoint
    toward_wall = max(
        perpendiculars,
        key=lambda direction: float(np.dot(direction, incubator_hint)),
    )
    # Operator visual verification is authoritative for display handedness.
    # Reflect only the lateral display axis: forward/depth and Z stay intact.
    lateral = np.array([-toward_wall[1], toward_wall[0]])
    display_reflection_xy = np.eye(2) - 2 * np.outer(lateral, lateral)
    for trace in traces:
        xy = display_reflection_xy @ np.vstack(
            [np.asarray(trace.x, float), np.asarray(trace.y, float)]
        )
        trace.x = xy[0]
        trace.y = xy[1]
        if isinstance(trace, go.Mesh3d):
            trace.j, trace.k = trace.k, trace.j
    eye_xy = -toward_wall * 1.65
    report["initial_view"] = {
        "description": "from_between_arms_straight_toward_opposite_wall",
        "arm_midpoint_xy_level_m": arm_midpoint.tolist(),
        "forward_xy": toward_wall.tolist(),
        "left_right_display_reflection_xy": display_reflection_xy.tolist(),
    }
    figure = go.Figure(traces)
    figure.update_layout(
        title="SAM-guided completed 3D objects (dark = observed RGB-D)",
        paper_bgcolor="#f8fafc", margin={"l": 0, "r": 0, "t": 52, "b": 0},
        legend={"orientation": "h", "y": 1.01},
        scene={
            "aspectmode": "data", "bgcolor": "#eef2f7",
            "xaxis": {"title": "X (m)"}, "yaxis": {"title": "Y (m)"}, "zaxis": {"title": "Z (m)"},
            "camera": {
                "eye": {
                    "x": float(eye_xy[0]),
                    "y": float(eye_xy[1]),
                    "z": 0.82,
                }
            },
        },
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output / "index.html", include_plotlyjs=True, full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--robot-mask", required=True)
    parser.add_argument("--microscope-mask", required=True)
    parser.add_argument("--incubator-mask", required=True)
    parser.add_argument("--bottle-mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--platform-obj",
        action="append",
        required=True,
        help="exactly two measured support OBJ files",
    )
    parser.add_argument(
        "--robot-model",
        default="robot/piper-mujoco/xml/lab-scene.xml",
    )
    args = parser.parse_args()
    if len(args.platform_obj) != 2:
        parser.error("--platform-obj must be supplied exactly twice")
    print(json.dumps(complete(args), indent=2))


if __name__ == "__main__":
    main()
