from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

from rollout.daily_scene import DailySceneStore
from rollout.semantic_scene_pipeline import MaskObservation
from src.build_semantic_scene import (
    _canonicalize_mesh,
    _install_configured_end_effectors,
    _position_articulated_model,
    _robot_environment_penetrations,
    _semantic_visual_parts,
    _write_mjcf,
    build,
)


def test_sparse_observed_object_can_keep_a_visual_only_semantic_completion(
    tmp_path,
):
    observed = tmp_path / "observed.obj"
    observed.write_text(
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n"
        "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n"
    )
    record = {
        "instance_id": "instrument-1",
        "semantic_name": "instrument",
        "geometry": {
            "kind": "box",
            "center_xyz_m": [1.0, 2.0, 0.3],
            "size_xyz_m": [0.4, 0.3, 0.5],
            "yaw_rad": np.pi / 2,
        },
        "completion": "observed_mesh",
        "observed_mesh": str(observed),
        "color": "#e333d1",
        "collision_boxes": [],
    }
    profile = {
        # completion=observed_mesh alone must prevent a filled AABB even when
        # the profile does not repeat the semantic name in a side list.
        "observed_surface_objects": [],
        "semantic_visual_templates": {
            "instrument": {
                "collision_authority": False,
                "parts": [
                    {
                        "name": "stage",
                        "kind": "box",
                        "size_xyz_m": [0.2, 0.1, 0.02],
                        "center_xy_offset_m": [0.1, 0.0],
                        "center_z_fraction": 0.5,
                    }
                ],
            }
        },
    }
    parts = _semantic_visual_parts(record, profile)
    assert len(parts) == 1
    assert parts[0]["inferred_visual_only"]
    assert np.allclose(parts[0]["geometry"]["center_xyz_m"], [1, 2.1, 0.3])

    output = tmp_path / "scene.xml"
    _write_mjcf(output, [record], profile)
    xml = output.read_text()
    assert 'name="instrument-1-observed"' in xml
    assert 'name="instrument-1-inferred-stage"' in xml
    inferred = ET.fromstring(xml).find(
        ".//body[@name='instrument-1-inferred-stage']/geom"
    )
    assert inferred is not None
    assert inferred.get("contype") == "0"
    assert inferred.get("conaffinity") == "0"
    mujoco.MjModel.from_xml_path(str(output))


def _fixture(tmp_path: Path):
    height, width = 32, 48
    x = np.linspace(-0.4, 0.4, width)
    y = np.linspace(-0.25, 0.25, height)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    mask = np.zeros((height, width), dtype=bool)
    mask[10:19, 20:29] = True
    zz[mask] = 0.04
    vertices = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    valid = np.ones(height * width, dtype=bool)
    grid = np.arange(height * width).reshape(height, width)
    faces = np.concatenate(
        [
            np.stack(
                [grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1]], axis=-1
            ).reshape(-1, 3),
            np.stack(
                [grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]], axis=-1
            ).reshape(-1, 3),
        ]
    )
    mesh = tmp_path / "mesh.npz"
    np.savez_compressed(
        mesh,
        vertices_xyz_m=vertices,
        faces=faces,
        valid_vertex_mask=valid,
    )
    rgb = tmp_path / "rgb.png"
    assert cv2.imwrite(str(rgb), np.full((height, width, 3), 100, np.uint8))
    mask_path = tmp_path / "sample.png"
    assert cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema": "piper_robot.scene_object_catalog/v1",
                "objects": [
                    {
                        "name": "sample",
                        "prompts": ["sample"],
                        "completion": "primitive",
                        "primitive": "box",
                        "nominal_size_m": [0.14, 0.10, 0.04],
                        "size_range_m": [
                            [0.08, 0.06, 0.02],
                            [0.20, 0.16, 0.08],
                        ],
                        "transparent": False,
                        "support": "horizontal_surface",
                        "minimum_confidence": 0.65,
                        "color": "#00cc88",
                    }
                ],
            }
        )
    )
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema": "piper_robot.semantic_scene_profile/v1",
                "catalog": "catalog.json",
                "objects": ["sample"],
                "organized_shape_hw": [height, width],
                "accepted_mask_score": 1.0,
                "support_height_tolerance_m": 0.006,
                "support_minimum_area_fraction": 0.01,
                "unknown_minimum_area_fraction": 0.2,
                "accepted_camera_to_robot": False,
            }
        )
    )
    return rgb, mesh, mask_path, profile


def test_end_to_end_accepted_mask_builds_compileable_scene(tmp_path):
    rgb, mesh, mask, profile = _fixture(tmp_path)
    output = tmp_path / "output"
    scene = build(
        argparse.Namespace(
            capture=None,
            rgb=str(rgb),
            mesh=str(mesh),
            profile=str(profile),
            output_dir=str(output),
            sam_endpoint="unused",
            mask=[f"sample={mask}"],
            daily_scene=None,
            resume_confirmed=False,
        )
    )
    assert scene["schema"] == "piper_robot.semantic_scene/v1"
    assert scene["objects"][0]["semantic_name"] == "sample"
    assert scene["mujoco_compile"]["ok"]
    assert scene["readiness"]["display_ready"]
    assert not scene["readiness"]["motion_ready"]
    assert (output / "sam_overlay.png").is_file()
    assert (output / "index.html").is_file()
    assert (output / "scene_esdf.npz").is_file()
    assert (output / "esdf.html").is_file()
    assert scene["esdf"]["unknown_collision_fraction"] > 0
    mujoco.MjModel.from_xml_path(str(output / "scene.xml"))


def test_low_touch_confirmation_resume_uses_same_revision(tmp_path):
    rgb, mesh, mask, profile = _fixture(tmp_path)
    output = tmp_path / "output"
    daily_path = tmp_path / "daily.json"
    arguments = argparse.Namespace(
        capture=None,
        rgb=str(rgb),
        mesh=str(mesh),
        profile=str(profile),
        output_dir=str(output),
        sam_endpoint="unused",
        mask=[f"sample={mask}"],
        daily_scene=str(daily_path),
        resume_confirmed=False,
    )
    initial = build(arguments)
    revision = initial["daily_scene"]["revision"]
    store = DailySceneStore(daily_path)
    proposed = store.load()
    assert proposed is not None
    assert all(item.status == "confirmed" for item in proposed.objects)
    assert proposed.status == "confirmed"
    assert proposed.confirmed_by == "semantic-scene-auto-gate"
    arguments.resume_confirmed = True
    arguments.rgb = None
    arguments.mesh = None
    resumed = build(arguments)
    assert resumed["resumed_from_confirmed_revision"] == revision
    assert resumed["readiness"]["collision_ready"]


def test_articulated_root_placement_uses_named_anchors_and_sam_height(tmp_path):
    model = tmp_path / "robot.xml"
    model.write_text(
        "<mujoco><compiler meshdir=\"assets\"/><worldbody>"
        "<body name=\"left/base_link\" pos=\"0 0.25 0\"/>"
        "<body name=\"right/base_link\" pos=\"0 -0.25 0\"/>"
        "</worldbody></mujoco>"
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "anchor_xyz_level_m": {
                    "left": [-0.2, 0.4, 0.0],
                    "right": [0.1, -0.1, 0.0],
                },
                "level_heights_m": {"right_platform": 0.01},
            }
        )
    )
    height, width = 12, 16
    vertices = np.zeros((height, width, 3), dtype=float)
    vertices[..., 0] = np.linspace(-0.4, 0.4, width)
    vertices[..., 1] = np.linspace(-0.3, 0.5, height)[:, None]
    left_mask = np.zeros((height, width), dtype=bool)
    right_mask = np.zeros((height, width), dtype=bool)
    left_mask[7:11, 2:6] = True
    right_mask[2:6, 10:14] = True
    vertices[..., 2][left_mask] = -0.22
    vertices[..., 2][right_mask] = -0.27
    valid = np.ones(height * width, dtype=bool)
    observation = MaskObservation(
        "robot", "robot", "robot", "unused", 1.0, "accepted", 0.0
    )
    runtime, report = _position_articulated_model(
        profile={
            "robot_model": str(model),
            "robot_placement": {
                "semantic_name": "robot",
                "anchor_map": "anchor_xyz_level_m",
                "yaw_offset_rad": np.pi / 2,
                "instances": [
                    {"body": "left/base_link", "anchor": "left"},
                    {"body": "right/base_link", "anchor": "right"},
                ],
            },
        },
        calibration_report=str(calibration),
        owned=[(observation, left_mask | right_mask)],
        mesh={
            "vertices": vertices.reshape(-1, 3),
            "valid": valid,
            "faces": np.empty((0, 3), dtype=np.int32),
        },
        shape_hw=(height, width),
        output_dir=tmp_path,
    )
    assert report["accepted"]
    assert Path(runtime["robot_model"]).name == "positioned_robot.xml"
    positioned = Path(runtime["robot_model"]).read_text()
    assert 'name="left/base_link"' in positioned
    assert "-0.2200000000" in positioned
    assert "-0.2700000000" in positioned


def test_canonical_axis_and_shared_base_height_are_applied_together(tmp_path):
    model = tmp_path / "robot.xml"
    model.write_text(
        "<mujoco><worldbody>"
        '<body name="left/base_link" pos="0 0.25 0"/>'
        '<body name="right/base_link" pos="0 -0.25 0"/>'
        "</worldbody></mujoco>"
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "anchor_xyz_level_m": {
                    "left": [-0.2, 0.4, 0.0],
                    "right": [0.1, -0.1, 0.0],
                },
                "level_heights_m": {"right_platform": 0.01},
            }
        )
    )
    height, width = 12, 16
    vertices = np.zeros((height, width, 3), dtype=float)
    vertices[..., 0] = np.linspace(-0.4, 0.4, width)
    vertices[..., 1] = np.linspace(-0.3, 0.5, height)[:, None]
    left_mask = np.zeros((height, width), dtype=bool)
    right_mask = np.zeros((height, width), dtype=bool)
    left_mask[7:11, 2:6] = True
    right_mask[2:6, 10:14] = True
    vertices[..., 2][left_mask] = -0.22
    vertices[..., 2][right_mask] = -0.27
    profile = {
        "scene_axis_sign": [-1, 1, 1],
        "robot_model": str(model),
        "robot_placement": {
            "semantic_name": "robot",
            "anchor_map": "anchor_xyz_level_m",
            "yaw_offset_rad": -np.pi / 2,
            "shared_base_height": {
                "support_level": "right_platform",
                "mount_below_support_m": 0.197,
            },
            "instances": [
                {"body": "left/base_link", "anchor": "left"},
                {"body": "right/base_link", "anchor": "right"},
            ],
        },
    }
    mesh = _canonicalize_mesh(
        {
            "vertices": vertices.reshape(-1, 3),
            "valid": np.ones(height * width, dtype=bool),
            "faces": np.array([[0, 1, width]], dtype=np.int32),
        },
        profile,
    )
    observation = MaskObservation(
        "robot", "robot", "robot", "unused", 1.0, "accepted", 0.0
    )
    _, report = _position_articulated_model(
        profile=profile,
        calibration_report=str(calibration),
        owned=[(observation, left_mask | right_mask)],
        mesh=mesh,
        shape_hw=(height, width),
        output_dir=tmp_path,
    )
    left = report["base_xyz_level_m"]["left/base_link"]
    right = report["base_xyz_level_m"]["right/base_link"]
    assert left[:2] == [0.2, 0.4]
    assert right[:2] == [-0.1, -0.1]
    assert left[2] == right[2] == 0.01 - 0.197
    assert report["raw_sam_base_height_m"]["left/base_link"] != report[
        "raw_sam_base_height_m"
    ]["right/base_link"]


def test_nyu_gripper_is_pinned_and_stock_fingers_are_rejected():
    root_dir = Path(__file__).resolve().parents[1]
    tree = ET.parse(
        root_dir / "robot/arm/mujoco/bimanual_piper_table.xml"
    )
    configuration = {
        "variant": "nyu_gripper_body",
        "mesh": str(
            root_dir / "robot/arm/mujoco/assets/gripper_body.stl"
        ),
        "bodies": ["left/gripper_base", "right/gripper_base"],
        "required_visual_geoms": [
            "left/nyu_gripper_visual",
            "right/nyu_gripper_visual",
        ],
        "forbidden_bodies": [
            "left/link7",
            "left/link8",
            "right/link7",
            "right/link8",
        ],
    }
    report = _install_configured_end_effectors(
        tree.getroot(), configuration
    )
    assert report["accepted"]
    for name in configuration["required_visual_geoms"]:
        assert tree.getroot().find(f".//geom[@name='{name}']") is not None
    for name in configuration["forbidden_bodies"]:
        assert tree.getroot().find(f".//body[@name='{name}']") is None


def test_home_robot_environment_penetration_is_reported(tmp_path):
    model_path = tmp_path / "penetration.xml"
    model_path.write_text(
        "<mujoco><worldbody>"
        '<body name="robot"><freejoint/>'
        '<geom name="robot_geom" type="box" size=".1 .1 .1"/>'
        "</body>"
        '<body name="wall" pos=".15 0 0">'
        '<geom name="wall_geom" type="box" size=".1 .1 .1"/>'
        "</body>"
        "</worldbody><keyframe>"
        '<key name="home" qpos="0 0 0 1 0 0 0"/>'
        "</keyframe></mujoco>"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    records = _robot_environment_penetrations(
        model,
        mujoco.MjData(model),
        profile={
            "robot_placement": {
                "instances": [{"body": "robot"}],
            }
        },
        keyframe="home",
    )
    assert records
    assert all(item["robot_body"] == "robot" for item in records)
    assert all(item["environment_body"] == "wall" for item in records)
    assert all(item["penetration_depth_m"] > 0.001 for item in records)


def test_operator_can_mark_false_unknown_candidate_absent_and_resume(tmp_path):
    rgb, mesh, mask, profile = _fixture(tmp_path)
    catalog_path = tmp_path / "catalog.json"
    catalog = json.loads(catalog_path.read_text())
    catalog["objects"][0]["minimum_confidence"] = 1.0
    catalog_path.write_text(json.dumps(catalog))
    profile_payload = json.loads(profile.read_text())
    profile_payload["accepted_mask_score"] = 0.5
    profile.write_text(json.dumps(profile_payload))
    output = tmp_path / "output"
    daily_path = tmp_path / "daily.json"
    arguments = argparse.Namespace(
        capture=None,
        rgb=str(rgb),
        mesh=str(mesh),
        profile=str(profile),
        output_dir=str(output),
        sam_endpoint="unused",
        mask=[f"sample={mask}"],
        daily_scene=str(daily_path),
        resume_confirmed=False,
        calibration_report=None,
    )
    initial = build(arguments)
    store = DailySceneStore(daily_path)
    draft = store.load()
    assert draft is not None and draft.status == "pending_confirmation"
    absent = [
        type(item)(
            **{
                **item.__dict__,
                "status": "absent",
            }
        )
        for item in draft.objects
    ]
    draft = store.replace_objects(absent, revision=draft.revision)
    store.confirm(revision=draft.revision, operator="test")
    arguments.resume_confirmed = True
    arguments.rgb = None
    arguments.mesh = None
    resumed = build(arguments)
    assert resumed["objects"] == []
    assert resumed["resumed_from_confirmed_revision"] == initial["daily_scene"][
        "revision"
    ]
