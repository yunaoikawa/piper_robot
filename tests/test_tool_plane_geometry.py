import numpy as np

from rollout.tool_plane_geometry import MuJoCoToolPlane, transform_plane


def test_transform_plane_rotates_point_and_normal():
    transform = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    point, normal = transform_plane([2, 0, 1], [1, 0, 0], transform)
    assert np.allclose(point, [1, 4, 4])
    assert np.allclose(normal, [0, 1, 0])


def test_mujoco_tool_mesh_returns_finite_vertices():
    geometry = MuJoCoToolPlane(
        "robot/cone-e-description/robot-welded-base-and-lift.mjcf"
    )
    vertices = geometry.tool_vertices_robot(np.zeros(6), np.zeros(6))
    assert vertices.ndim == 2
    assert vertices.shape[0] > 100
    assert vertices.shape[1] == 3
    assert np.all(np.isfinite(vertices))


def test_tool_plane_clearance_uses_nearest_mesh_vertex():
    geometry = MuJoCoToolPlane(
        "robot/cone-e-description/robot-welded-base-and-lift.mjcf"
    )
    vertices = geometry.tool_vertices_robot(np.zeros(6), np.zeros(6))
    support_z = float(np.min(vertices[:, 2]) - 0.01)
    clearance = geometry.clearance(
        np.zeros(6), np.zeros(6), [0, 0, support_z], [0, 0, 1]
    )
    assert np.isclose(clearance, 0.01, atol=1e-6)


def test_tool_plane_normal_points_from_support_toward_tool():
    geometry = MuJoCoToolPlane(
        "robot/cone-e-description/robot-welded-base-and-lift.mjcf"
    )
    vertices = geometry.tool_vertices_robot(np.zeros(6), np.zeros(6))
    support_z = float(np.min(vertices[:, 2]) - 0.01)
    clearance, normal = geometry.clearance_and_free_normal(
        np.zeros(6), np.zeros(6), [0, 0, support_z], [0, 0, -1]
    )
    assert np.isclose(clearance, 0.01, atol=1e-6)
    assert np.allclose(normal, [0, 0, 1])
