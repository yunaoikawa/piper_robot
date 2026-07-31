from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from rollout.scene_registration import (
    apply_independent_base_translations_to_mjcf,
    apply_shared_planar_transform_to_mjcf,
    assign_components_by_joint_excitation,
    assign_independent_base_translations,
    assign_named_base_translations,
    assign_visible_base_translations,
    bridge_camera_from_fixed_tag,
    depth_layer_foreground_mask,
    fit_shared_planar_robot_transform,
    intersect_pixel_with_horizontal_plane,
    persistent_depth_component_centers,
    reject_base_candidates_inside_semantic_objects,
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


def test_named_base_assignment_does_not_swap_by_initial_proximity():
    initial = {
        "left/base_link": np.array([-0.4, 0.0, -0.7]),
        "right/base_link": np.array([0.4, 0.0, -0.7]),
    }
    observed = {
        "left/base_link": np.array([0.5, 0.1, -0.5]),
        "right/base_link": np.array([-0.5, 0.2, -0.5]),
    }
    fit = assign_named_base_translations(initial, observed)
    np.testing.assert_allclose(
        fit["translations_xy_m"]["left/base_link"],
        [0.9, 0.1],
    )
    np.testing.assert_allclose(
        fit["translations_xy_m"]["right/base_link"],
        [-0.9, 0.2],
    )


def test_semantic_object_volume_rejects_microscope_base_false_positive():
    candidates = [
        np.array([-0.37, 1.25, -0.57]),
        np.array([-0.11, 0.57, -0.71]),
    ]
    objects = [
        {
            "instance_id": "robot",
            "semantic_name": "robot",
            "geometry": {
                "kind": "box",
                "center_xyz_m": [-0.2, 0.8, -0.5],
                "size_xyz_m": [1.0, 1.0, 1.0],
                "yaw_rad": 0.0,
            },
        },
        {
            "instance_id": "microscope-1",
            "semantic_name": "microscope",
            "geometry": {
                "kind": "box",
                "center_xyz_m": [-0.341, 1.249, -0.518],
                "size_xyz_m": [0.55, 0.453, 0.403],
                "yaw_rad": 0.892,
            },
        },
        {
            "instance_id": "measured-static-scene",
            "semantic_name": "measured_static_scene",
            "source": "multiview_rgbd_background_faces",
            "geometry": {
                "kind": "box",
                "center_xyz_m": [0.0, 0.8, -0.55],
                "size_xyz_m": [2.0, 2.0, 2.0],
                "yaw_rad": 0.0,
            },
        },
    ]
    accepted, records = reject_base_candidates_inside_semantic_objects(
        candidates,
        objects,
    )
    assert len(accepted) == 1
    np.testing.assert_allclose(accepted[0], candidates[1])
    assert records[0]["accepted"] is False
    assert records[0]["overlapping_semantic_objects"] == [
        {
            "instance_id": "microscope-1",
            "semantic_name": "microscope",
        }
    ]
    assert records[1]["accepted"] is True


def test_partial_visible_base_fit_retains_unobserved_left_base():
    initial = {
        "left/base_link": np.array([-0.65, 0.71, -0.76]),
        "right/base_link": np.array([-0.10, 0.56, -0.76]),
    }
    fit = assign_visible_base_translations(
        initial,
        [np.array([-0.11, 0.57, -0.71])],
    )
    assert fit["observed_bases"] == ["right/base_link"]
    assert fit["retained_unobserved_bases"] == ["left/base_link"]
    assert fit["all_bases_observed"] is False
    np.testing.assert_allclose(
        fit["translations_xy_m"]["left/base_link"],
        [0.0, 0.0],
    )
    np.testing.assert_allclose(
        fit["translations_xy_m"]["right/base_link"],
        [-0.01, 0.01],
    )


def test_partial_visible_base_fit_rejects_large_static_object_jump():
    initial = {
        "left/base_link": np.array([-0.65, 0.71, -0.76]),
        "right/base_link": np.array([-0.10, 0.56, -0.76]),
    }
    with np.testing.assert_raises_regex(ValueError, "above"):
        assign_visible_base_translations(
            initial,
            [np.array([-0.37, 1.25, -0.57])],
            maximum_translation_m=0.15,
        )


def test_joint_excitation_assigns_components_without_view_name_heuristics():
    shape = (100, 80)
    baseline = np.zeros(shape, dtype=bool)
    masks = {
        "arbitrary_baseline": baseline,
        "misnamed_left": baseline.copy(),
        "misnamed_right": baseline.copy(),
    }
    # Physical right moved around component 1 even though the view name says
    # left; physical left moved around component 0 in the oppositely named
    # view.
    masks["misnamed_left"][15:35, 55:75] = True
    masks["misnamed_right"][65:90, 5:30] = True
    qpos = {
        "arbitrary_baseline": {
            "left": np.zeros(6),
            "right": np.zeros(6),
        },
        "misnamed_left": {
            "left": np.full(6, 0.002),
            "right": np.full(6, 0.2),
        },
        "misnamed_right": {
            "left": np.full(6, 0.3),
            "right": np.full(6, 0.02),
        },
    }
    result = assign_components_by_joint_excitation(
        qpos_by_view=qpos,
        robot_masks_by_view=masks,
        component_centers_px=[
            np.array([17.0, 77.0]),
            np.array([65.0, 25.0]),
        ],
        baseline_view="arbitrary_baseline",
        motion_radius_fraction=0.3,
    )
    assert result["physical_arm_to_component_index"] == {
        "left": 0,
        "right": 1,
    }
    assert result["evidence"]["right"]["view"] == "misnamed_left"
    assert result["evidence"]["left"]["view"] == "misnamed_right"
