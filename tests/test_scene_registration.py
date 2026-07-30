from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rollout.scene_registration import (
    apply_independent_base_translations_to_mjcf,
    apply_shared_planar_transform_to_mjcf,
    assign_independent_base_translations,
    bridge_camera_from_fixed_tag,
    depth_layer_foreground_mask,
    fit_shared_planar_robot_transform,
    intersect_pixel_with_horizontal_plane,
    persistent_depth_component_centers,
    rigid_transform_consensus,
)


def _transform(xyz, yaw=0.0):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    value[:3, 3] = xyz
    return value


def test_fixed_tag_bridge_recovers_scene_camera():
    scene_from_camera = _transform([0.4, -0.2, 0.8], 0.3)
    camera_from_tag = _transform([0.1, 0.0, 0.6], -0.1)
    scene_from_tag = scene_from_camera @ camera_from_tag
    np.testing.assert_allclose(
        bridge_camera_from_fixed_tag(scene_from_tag, camera_from_tag),
        scene_from_camera,
        atol=1e-9,
    )


def test_consensus_reports_outlier_spread():
    values = [
        _transform([0.0, 0.0, 0.0], 0.0),
        _transform([0.002, 0.0, 0.0], 0.01),
        _transform([-0.002, 0.0, 0.0], -0.01),
    ]
    result = rigid_transform_consensus(values)
    assert result.sample_count == 3
    assert result.translation_spread_m == 0.002
    assert 0.5 < result.rotation_spread_deg < 0.6


def test_shared_fit_preserves_base_baseline():
    initial = {
        "left/base_link": np.array([-0.3, 0.1, -0.7]),
        "right/base_link": np.array([0.3, 0.1, -0.7]),
    }
    yaw = 0.2
    rotation = Rotation.from_euler("z", yaw).as_matrix()[:2, :2]
    translation = np.array([0.08, -0.04])
    observed = [
        rotation @ initial[name][:2] + translation
        for name in initial
    ]
    result = fit_shared_planar_robot_transform(initial, observed)
    np.testing.assert_allclose(result["translation_xy_m"], translation)
    assert abs(result["yaw_delta_rad"] - yaw) < 1e-9
    assert result["rms_residual_m"] < 1e-9


def test_mjcf_transform_moves_both_roots_rigidly(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text(
        "<mujoco><worldbody>"
        '<body name="left" pos="-1 0 0" euler="0 0 0.2"/>'
        '<body name="right" pos="1 0 0" euler="0 0 0.2"/>'
        "</worldbody></mujoco>"
    )
    output = tmp_path / "output.xml"
    apply_shared_planar_transform_to_mjcf(
        source,
        output,
        root_bodies=("left", "right"),
        translation_xy_m=(0.5, 0.2),
        yaw_delta_rad=np.pi / 2,
    )
    text = output.read_text()
    assert 'pos="0.5000000000 -0.8000000000 0.0000000000"' in text
    assert 'pos="0.5000000000 1.2000000000 0.0000000000"' in text


def test_ray_plane_intersection():
    point = intersect_pixel_with_horizontal_plane(
        (0.0, 0.0),
        np.eye(3),
        _transform([0.0, 0.0, 1.0]),
        plane_z_m=2.0,
    )
    np.testing.assert_allclose(point, [0.0, 0.0, 2.0])


def test_depth_layer_separates_touching_foreground_from_static_background():
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:10, 2:14] = 255
    depth = np.full(mask.shape, 2.0, dtype=np.float32)
    depth[2:10, 2:7] = 1.0
    background = np.full(mask.shape, 2.0, dtype=np.float32)
    confidence = np.ones(mask.shape, dtype=np.uint8)
    selected, records = depth_layer_foreground_mask(
        mask,
        depth,
        confidence,
        background,
        maximum_neighbor_depth_jump_m=0.05,
        minimum_component_pixels=10,
        minimum_dynamic_pixels=5,
    )
    assert np.all(selected[2:10, 2:7] == 255)
    assert np.all(selected[2:10, 7:14] == 0)
    assert sum(item["accepted"] for item in records) == 1


def test_persistent_depth_components_reject_moving_points():
    stationary = np.array(
        [[0.00, 0.00, 0.00], [0.02, 0.00, 0.00], [0.00, 0.02, 0.00]]
    )
    clouds = [
        np.vstack((stationary, [[0.2 + index * 0.1, 0.0, 0.0]]))
        for index in range(4)
    ]
    centers, records = persistent_depth_component_centers(
        clouds,
        voxel_size_m=0.02,
        minimum_views=4,
        minimum_voxels=3,
    )
    assert len(centers) == 1
    assert records[0]["voxel_count"] == 3


def test_independent_base_assignment_and_mjcf_translation(tmp_path):
    initial = {
        "left": np.array([-0.4, 0.0, -0.7]),
        "right": np.array([0.4, 0.0, -0.7]),
    }
    observed = [np.array([0.42, 0.01, -0.5]), np.array([-0.3, 0.2, -0.5])]
    fit = assign_independent_base_translations(initial, observed)
    np.testing.assert_allclose(
        fit["translations_xy_m"]["left"],
        [0.1, 0.2],
    )
    source = tmp_path / "source.xml"
    source.write_text(
        "<mujoco><worldbody>"
        '<body name="left" pos="-0.4 0 -0.7" euler="0 0 0.3"/>'
        '<body name="right" pos="0.4 0 -0.7" euler="0 0 0.3"/>'
        "</worldbody></mujoco>"
    )
    output = tmp_path / "output.xml"
    apply_independent_base_translations_to_mjcf(
        source,
        output,
        translations_xy_m=fit["translations_xy_m"],
    )
    text = output.read_text()
    assert 'pos="-0.3000000000 0.2000000000 -0.7000000000"' in text
    assert 'euler="0 0 0.3"' in text
