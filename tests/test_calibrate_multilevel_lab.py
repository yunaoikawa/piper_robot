import xml.etree.ElementTree as ET

import numpy as np

from src.calibrate_multilevel_lab import (
    _discover_horizontal_surfaces,
    _triangulate_depth_grid,
    _write_measured_support_mjcf,
    _write_support_prism,
)


def _two_level_grid():
    height, width = 64, 80
    yy, xx = np.mgrid[:height, :width]
    points = np.stack(
        [
            (xx - width / 2) * 0.005,
            (yy - height / 2) * 0.005,
            np.where(xx < 40, 0.15, 0.0),
        ],
        axis=-1,
    )
    return points, np.ones((height, width), dtype=bool)


def test_discovers_distinct_horizontal_support_heights():
    points, valid = _two_level_grid()
    _, surfaces = _discover_horizontal_surfaces(
        points, points, valid, np.array([0.0, 0.0, 1.0])
    )
    major_heights = sorted(
        surface["height_m"]
        for surface in surfaces
        if surface["area_pixels_lowres"] > 500
    )
    assert np.allclose(major_heights, [0.0, 0.15], atol=0.005)


def test_support_prism_is_watertight_and_mjcf_compiles(tmp_path):
    points, valid = _two_level_grid()
    _, surfaces = _discover_horizontal_surfaces(
        points, points, valid, np.array([0.0, 0.0, 1.0])
    )
    obj_path = tmp_path / "one.obj"
    metadata = _write_support_prism(obj_path, points[points[..., 2] > 0.1])
    assert metadata["vertices"] >= 8
    assert metadata["faces"] >= 12

    xml_path, exported = _write_measured_support_mjcf(
        tmp_path, points, surfaces
    )
    assert len(exported) == 2
    ET.parse(xml_path)

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.ngeom == 2


def test_depth_mesh_does_not_bridge_height_discontinuity():
    points, valid = _two_level_grid()
    faces = _triangulate_depth_grid(points, valid, maximum_edge_m=0.04)
    flat_height = points[..., 2].reshape(-1)
    assert np.all(np.ptp(flat_height[faces], axis=1) < 0.04)
