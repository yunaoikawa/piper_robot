"""MuJoCo tool-body geometry against an observed support plane."""

from __future__ import annotations

import numpy as np


def transform_plane(point_xyz, normal_xyz, transform):
    point = np.asarray(point_xyz, dtype=float)
    normal = np.asarray(normal_xyz, dtype=float)
    matrix = np.asarray(transform, dtype=float)
    if point.shape != (3,) or normal.shape != (3,) or matrix.shape != (4, 4):
        raise ValueError("invalid point, normal, or transform")
    length = float(np.linalg.norm(normal))
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("plane normal is invalid")
    transformed_point = (matrix @ np.asarray((*point, 1.0)))[:3]
    transformed_normal = matrix[:3, :3] @ (normal / length)
    transformed_normal /= np.linalg.norm(transformed_normal)
    return transformed_point, transformed_normal


class MuJoCoToolPlane:
    """Evaluate a configured tool body's true mesh vertices in robot space."""

    def __init__(
        self,
        model_path,
        *,
        physical_right_joint_prefix="left_arm_joint",
        physical_left_joint_prefix="right_arm_joint",
        tool_body="left_arm_gripper_base",
    ):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.right_joint_names = [
            f"{physical_right_joint_prefix}{index}" for index in range(1, 7)
        ]
        self.left_joint_names = [
            f"{physical_left_joint_prefix}{index}" for index in range(1, 7)
        ]
        self.tool_body_id = int(self.model.body(tool_body).id)

    def _data(self, right_q, left_q):
        right = np.asarray(right_q, dtype=float)
        left = np.asarray(left_q, dtype=float)
        if right.shape != (6,) or left.shape != (6,):
            raise ValueError("both arms require six joint values")
        data = self.mujoco.MjData(self.model)
        for name, value in zip(self.right_joint_names, right):
            data.qpos[self.model.joint(name).qposadr[0]] = value
        for name, value in zip(self.left_joint_names, left):
            data.qpos[self.model.joint(name).qposadr[0]] = value
        self.mujoco.mj_forward(self.model, data)
        return data

    def tool_vertices_robot(self, right_q, left_q) -> np.ndarray:
        data = self._data(right_q, left_q)
        body = self.model.body(self.tool_body_id)
        vertices = []
        for geom_id in range(int(body.geomadr), int(body.geomadr + body.geomnum)):
            geom = self.model.geom(geom_id)
            mesh_id = int(geom.dataid)
            if mesh_id < 0:
                # Primitive fallback is conservative in every direction.
                radius = float(geom.rbound)
                center = np.asarray(data.geom_xpos[geom_id], dtype=float)
                for axis in np.eye(3):
                    vertices.extend((center + radius * axis, center - radius * axis))
                continue
            mesh = self.model.mesh(mesh_id)
            start = int(mesh.vertadr)
            count = int(mesh.vertnum)
            local = np.asarray(
                self.model.mesh_vert[start : start + count], dtype=float
            )
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            translation = np.asarray(data.geom_xpos[geom_id], dtype=float)
            vertices.extend(local @ rotation.T + translation)
        if not vertices:
            raise ValueError("configured tool body has no geometry")
        return np.asarray(vertices, dtype=float)

    def clearance(self, right_q, left_q, plane_point, plane_normal) -> float:
        vertices = self.tool_vertices_robot(right_q, left_q)
        point = np.asarray(plane_point, dtype=float)
        normal = np.asarray(plane_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        signed = (vertices - point) @ normal
        # Plane-normal sign is arbitrary.  The tool centre is expected on the
        # free side; choose that sign before taking the nearest vertex.
        centre_signed = float((vertices.mean(axis=0) - point) @ normal)
        if centre_signed < 0:
            signed = -signed
        return float(np.min(signed))

    def clearance_and_free_normal(
        self, right_q, left_q, plane_point, plane_normal
    ) -> tuple[float, np.ndarray]:
        """Return clearance and the plane normal pointing toward the tool."""

        vertices = self.tool_vertices_robot(right_q, left_q)
        point = np.asarray(plane_point, dtype=float)
        normal = np.asarray(plane_normal, dtype=float)
        normal /= np.linalg.norm(normal)
        if float((vertices.mean(axis=0) - point) @ normal) < 0:
            normal = -normal
        return float(np.min((vertices - point) @ normal)), normal
