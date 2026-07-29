#!/usr/bin/env python3
"""Export a MuJoCo model as a phone-friendly interactive Plotly view."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def _rotation(mujoco, quaternion):
    matrix = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def render(model_path, output, *, right_q=None, left_q=None):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    for prefix, values in (("left_arm", right_q), ("right_arm", left_q)):
        if values is None:
            continue
        for index, value in enumerate(values, 1):
            address = model.joint(f"{prefix}_joint{index}").qposadr[0]
            data.qpos[address] = float(value)
    mujoco.mj_forward(model, data)

    traces = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
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
        mesh_rotation = _rotation(mujoco, model.mesh_quat[mesh_id])
        vertices = (
            mesh_rotation @ vertices.T
        ).T + np.asarray(model.mesh_pos[mesh_id], dtype=float)
        geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        vertices = (
            geom_rotation @ vertices.T
        ).T + np.asarray(data.geom_xpos[geom_id], dtype=float)
        faces = np.asarray(
            model.mesh_face[face_start : face_start + face_count], dtype=int
        )
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
                opacity=max(0.25, float(rgba[3])),
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
            camera=dict(eye=dict(x=1.4, y=-1.6, z=1.0)),
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--right-q", nargs=6, type=float)
    parser.add_argument("--left-q", nargs=6, type=float)
    args = parser.parse_args(argv)
    render(
        args.model,
        args.output,
        right_q=args.right_q,
        left_q=args.left_q,
    )
    print(args.output)


if __name__ == "__main__":
    main()
