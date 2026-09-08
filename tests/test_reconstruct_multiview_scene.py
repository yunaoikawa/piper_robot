#!/usr/bin/env python3

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.multiview_scene import PoseRefinement
from src.reconstruct_multiview_scene import _tracked_pose_choice, build


def _write_synthetic_capture(root: Path) -> tuple[Path, list[str]]:
    capture = root / "capture"
    capture.mkdir()
    height, width = 72, 96
    matrix = [
        [92.0, 0.0, width / 2],
        [0.0, 92.0, height / 2],
        [0.0, 0.0, 1.0],
    ]
    views = []
    mask_specs = []
    for view_index, name in enumerate(("center", "right")):
        records = []
        for sequence in range(3):
            directory = capture / "raw" / "head" / name / f"{sequence:06d}"
            directory.mkdir(parents=True)
            rgb = np.full((height, width, 3), 110, dtype=np.uint8)
            rgb[:, : width // 5] = (40, 80, 160)
            depth = np.full((height, width), 0.80, dtype=np.float32)
            confidence = np.full((height, width), 2, dtype=np.uint8)
            rgb_path = directory / "rgb.png"
            depth_path = directory / "depth.npy"
            confidence_path = directory / "confidence.npy"
            assert cv2.imwrite(str(rgb_path), rgb)
            np.save(depth_path, depth)
            np.save(confidence_path, confidence)
            records.append(
                {
                    "sequence": sequence,
                    "camera_pose": {
                        "translation_xyz_m": [0.01 * view_index, 0.0, 0.0],
                        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "intrinsics": {"K_raw_rgb": matrix},
                    "files": {
                        "rgb_png": {
                            "path": str(rgb_path.relative_to(capture))
                        },
                        "depth_npy": {
                            "path": str(depth_path.relative_to(capture))
                        },
                        "confidence_npy": {
                            "path": str(confidence_path.relative_to(capture))
                        },
                    },
                }
            )
        robot_mask = np.zeros((height, width), dtype=np.uint8)
        robot_mask[:, : width // 5] = 255
        incubator_mask = np.zeros((height, width), dtype=np.uint8)
        incubator_mask[12:58, 38:82] = 255
        for label, mask in (
            ("robot", robot_mask),
            ("incubator", incubator_mask),
        ):
            path = capture / f"{name}_{label}.png"
            assert cv2.imwrite(str(path), mask)
            mask_specs.append(f"{name}:{label}={path}")
        views.append(
            {
                "name": name,
                "pose_stability": {
                    "maximum_translation_m": 0.0,
                    "maximum_rotation_deg": 0.0,
                    "accepted": True,
                },
                "frames": records,
            }
        )
    manifest = {
        "schema": "piper_robot.rgbd_multiview_capture/v1",
        "status": "complete",
        "view_order": ["center", "right"],
        "device": {"record3d_device_type": 1},
        "views": views,
    }
    (capture / "manifest.json").write_text(json.dumps(manifest))
    return capture, mask_specs


def test_saved_multiview_pipeline_produces_gated_phone_artifacts():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        capture, masks = _write_synthetic_capture(root)
        output = root / "output"
        args = Namespace(
            capture=str(capture),
            profile=str(
                Path(__file__).resolve().parents[1]
                / "src/configs/pasteur_semantic_scene.json"
            ),
            output_dir=str(output),
            sam_endpoint="tcp://127.0.0.1:1",
            mask=masks,
            dynamic_object=[
                "robot",
                "culture_media_bottle",
                "petri_dish",
                "petri_lid",
            ],
            attempt=1,
            voxel_size=0.02,
            truncation=0.04,
            minimum_confidence=1,
            min_depth=0.20,
            max_depth=2.0,
            registration_stride=3,
            mesh_stride=2,
            maximum_correspondence_m=0.06,
            acceptance_median_m=0.010,
            acceptance_p90_m=0.025,
            acceptance_overlap=0.30,
            acceptance_support_spread_m=0.008,
            minimum_baseline_m=0.04,
            minimum_baseline_deg=5.0,
            maximum_voxels=2_000_000,
            maximum_esdf_points=1000,
            maximum_frontier_points=1000,
        )
        report = build(args)
        assert report["registration"]["accepted_view_names"] == [
            "center",
            "right",
        ]
        assert not report["readiness"]["motion_ready"]
        assert (
            report["readiness"]["next_stage"]
            == "repeat_multiview_capture_with_revised_angles"
        )
        for name in (
            "index.html",
            "scene.html",
            "multiview_report.json",
            "scene_esdf_multiview.npz",
            "scene_mesh_multiview.ply",
        ):
            assert (output / name).is_file()


def test_continuous_pose_rejects_icp_that_loses_overlap():
    seed_transform = np.eye(4)
    refined_transform = np.eye(4)
    refined_transform[0, 3] = 0.01
    seed = PoseRefinement(
        seed_transform, 0.040, 0.070, 0.30, 0, False, ()
    )
    refined = PoseRefinement(
        refined_transform, 0.015, 0.055, 0.12, 8, False, ()
    )
    chosen, audit = _tracked_pose_choice(
        seed,
        refined,
        minimum_overlap=0.15,
        maximum_translation_m=0.03,
        maximum_rotation_deg=3.0,
    )
    assert chosen.accepted
    assert np.allclose(chosen.reference_from_camera, seed_transform)
    assert audit["pose_authority"] == "record3d_continuous_tracking"


if __name__ == "__main__":
    test_saved_multiview_pipeline_produces_gated_phone_artifacts()
    test_continuous_pose_rejects_icp_that_loses_overlap()
    print("multiview reconstruction checks passed")
