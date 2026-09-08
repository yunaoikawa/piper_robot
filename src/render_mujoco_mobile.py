#!/usr/bin/env python3
"""Export a MuJoCo model as a phone-friendly interactive Plotly view."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def _box(size):
    x, y, z = np.asarray(size, dtype=float)
    vertices = np.array(
        [[sx * x, sy * y, sz * z] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    faces = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ]
    )
    return vertices, faces


def _cylinder(radius, half_height, segments=24):
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    vertices = np.vstack(
        [
            np.c_[ring, np.full(segments, -half_height)],
            np.c_[ring, np.full(segments, half_height)],
            [0, 0, -half_height],
            [0, 0, half_height],
        ]
    )
    bottom, top = 2 * segments, 2 * segments + 1
    faces = []
    for index in range(segments):
        following = (index + 1) % segments
        faces.extend(
            [
                [index, following, segments + following],
                [index, segments + following, segments + index],
                [bottom, following, index],
                [top, segments + index, segments + following],
            ]
        )
    return vertices, np.asarray(faces)


def _sphere(radius, latitude=10, longitude=20):
    vertices = [[0, 0, radius], [0, 0, -radius]]
    for row in range(1, latitude):
        phi = np.pi * row / latitude
        for column in range(longitude):
            theta = 2 * np.pi * column / longitude
            vertices.append(
                [
                    radius * np.sin(phi) * np.cos(theta),
                    radius * np.sin(phi) * np.sin(theta),
                    radius * np.cos(phi),
                ]
            )
    faces = []
    for column in range(longitude):
        following = (column + 1) % longitude
        faces.append([0, 2 + column, 2 + following])
        faces.append(
            [
                1,
                2 + (latitude - 2) * longitude + following,
                2 + (latitude - 2) * longitude + column,
            ]
        )
    for row in range(latitude - 2):
        base = 2 + row * longitude
        following_base = base + longitude
        for column in range(longitude):
            following = (column + 1) % longitude
            faces.extend(
                [
                    [base + column, following_base + column, following_base + following],
                    [base + column, following_base + following, base + following],
                ]
            )
    return np.asarray(vertices), np.asarray(faces)


def _rotation(mujoco, quaternion):
    matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def render(
    model_path,
    output,
    *,
    right_q=None,
    left_q=None,
    keyframe=None,
    camera_eye=None,
):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nkey:
        key_id = 0 if keyframe is None else model.key(keyframe).id
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    def has_joint(name):
        try:
            model.joint(name)
            return True
        except KeyError:
            return False

    joint_prefixes = (
        (
            "left_arm" if has_joint("left_arm_joint1") else "",
            right_q,
        ),
        (
            "right_arm" if has_joint("right_arm_joint1") else "left_",
            left_q,
        ),
    )
    for prefix, values in joint_prefixes:
        if values is None:
            continue
        for index, value in enumerate(values, 1):
            candidates = (
                f"{prefix}_joint{index}",
                f"{prefix}joint{index}",
            )
            joint = next(
                (model.joint(name) for name in candidates if has_joint(name)),
                None,
            )
            if joint is None:
                break
            address = joint.qposadr[0]
            data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)

    traces = []
    for geom_id in range(model.ngeom):
        if int(model.geom_group[geom_id]) == 3:
            continue
        geom_type = int(model.geom_type[geom_id])
        size = np.asarray(model.geom_size[geom_id], dtype=float)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                continue
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            face_start = int(model.mesh_faceadr[mesh_id])
            face_count = int(model.mesh_facenum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[vertex_start : vertex_start + vertex_count],
                dtype=float,
            )
            # MuJoCo has already normalized mesh_vert into the runtime mesh
            # frame.  mesh_pos/mesh_quat are bookkeeping for that conversion,
            # not another render transform.
            faces = np.asarray(
                model.mesh_face[face_start : face_start + face_count], dtype=int
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            vertices, faces = _box(size[:3])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            vertices, faces = _cylinder(size[0], size[1])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            vertices, faces = _cylinder(size[0], size[1])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            vertices, faces = _sphere(size[0])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
            vertices, faces = _box((min(size[0], 2.0), min(size[1], 2.0), 0.002))
        else:
            continue
        geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        vertices = (
            geom_rotation @ vertices.T
        ).T + np.asarray(data.geom_xpos[geom_id], dtype=float)
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=float)
        body_name = model.body(model.geom_bodyid[geom_id]).name
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})",
                opacity=max(0.05, float(rgba[3])),
                flatshading=False,
                name=body_name,
                hovertemplate=body_name + "<extra></extra>",
                showlegend=False,
            )
        )
    figure = go.Figure(traces)
    figure.update_layout(
        title="Current Piper MuJoCo model",
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        margin=dict(l=0, r=0, t=48, b=0),
        scene=dict(
            aspectmode="data",
            bgcolor="#eef2f7",
            xaxis=dict(title="X", backgroundcolor="#ffffff", gridcolor="#cbd5e1"),
            yaxis=dict(title="Y", backgroundcolor="#ffffff", gridcolor="#cbd5e1"),
            zaxis=dict(title="Z", backgroundcolor="#ffffff", gridcolor="#cbd5e1"),
            camera=dict(
                eye=dict(
                    x=(camera_eye or (1.4, -1.6, 1.0))[0],
                    y=(camera_eye or (1.4, -1.6, 1.0))[1],
                    z=(camera_eye or (1.4, -1.6, 1.0))[2],
                )
            ),
        ),
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(output),
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="robot/piper-mujoco/xml/lab-scene.xml",
        help="defaults to the MacBook-authored nominal lab scene",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--right-q", nargs=6, type=float)
    parser.add_argument("--left-q", nargs=6, type=float)
    parser.add_argument("--keyframe")
    parser.add_argument(
        "--camera-eye",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="initial Plotly camera direction, e.g. -1.4 1.6 1.0",
    )
    args = parser.parse_args(argv)
    render(
        args.model,
        args.output,
        right_q=args.right_q,
        left_q=args.left_q,
        keyframe=args.keyframe,
        camera_eye=args.camera_eye,
    )
    print(args.output)


if __name__ == "__main__":
    main()
