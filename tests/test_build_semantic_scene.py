from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from rollout.daily_scene import DailySceneStore
from rollout.semantic_scene_pipeline import MaskObservation
from src.build_semantic_scene import _position_articulated_model, build


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
                }
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
