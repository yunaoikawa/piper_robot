import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.scene_semantics import LABEL_LID, LABEL_ROBOT
from wetrobo.perception.catalog import LabwareCatalog, LabwareEntry
from wetrobo.perception.sam import (
    CalibrationRejected,
    SamCalibrationArtifact,
    SamLabelBinding,
)


def _expect_rejected(callback):
    try:
        callback()
    except CalibrationRejected:
        return
    raise AssertionError("expected CalibrationRejected")


def _write_artifact(
    directory: Path,
    *,
    sync_delta=0.001,
    nominal_pose=None,
    T_level_camera=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    gx, gy = np.meshgrid(
        np.linspace(-0.04, 0.04, 8),
        np.linspace(-0.025, 0.025, 6),
    )
    lid = np.column_stack(
        (0.25 + gx.ravel(), -0.10 + gy.ravel(), np.full(gx.size, 0.002))
    )
    robot = np.column_stack(
        (
            -0.15 + gx.ravel(),
            0.12 + gy.ravel(),
            np.linspace(0.02, 0.15, gx.size),
        )
    )
    vertices = np.vstack((lid, robot))
    labels = np.r_[
        np.full(len(lid), LABEL_LID, dtype=np.uint8),
        np.full(len(robot), LABEL_ROBOT, dtype=np.uint8),
    ]
    faces = np.array(
        [[i, i + 1, i + 2] for i in range(0, len(vertices) - 2, 3)],
        dtype=np.int32,
    )
    np.savez_compressed(
        directory / "scene_mesh_levelled.npz",
        vertices_xyz_m=vertices,
        faces=faces,
        colors_rgb=np.zeros((len(vertices), 3), dtype=np.uint8),
        semantic_labels=labels,
    )
    if T_level_camera is None:
        transform = np.eye(4)
        transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
        transform[:3, 3] = [0.0, 0.0, 0.5]
    else:
        transform = T_level_camera
    np.savez_compressed(
        directory / "scene_esdf.npz",
        camera_to_level_rotation=transform[:3, :3],
        camera_to_level_translation=transform[:3, 3],
    )
    label_image = np.zeros((100, 120), dtype=np.uint8)
    label_image[42:58, 50:70] = LABEL_LID
    label_image[20:40, 10:35] = LABEL_ROBOT
    np.save(directory / "semantic_labels_rgb.npy", label_image)
    camera_matrix = np.array(
        [[100.0, 0.0, 60.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    report = {
        "artifact_version": 2,
        "sam_semantics": True,
        "sam_scores_available": False,
        "capture_sync_verified": True,
        "capture_sync_evidence": "synthetic_test_timestamp",
        "rgb_depth_file_mtime_delta_s": sync_delta,
        "sam_registration": {
            "inliers": 120,
            "median_inlier_residual_px": 0.12,
        },
        "sam_mask_overlap_pixels": 0,
        "support_plane_fit": {
            "rms_m": 0.002,
            "inlier_fraction": 0.65,
        },
        "coordinate_frame": "support-plane-levelled",
        "rgb_camera_matrix": camera_matrix.tolist(),
        "semantic_surface_pixels": {
            "robot_surface": 1800,
            "lid_surface": 600,
        },
        "rgb_source": "rgb.png",
        "depth_source": "depth.npy",
        # Deliberately irrelevant. The SAM pose must not depend on nominal CAD.
        "nominal_pose": nominal_pose,
    }
    (directory / "esdf_report.json").write_text(json.dumps(report))


def test_extrinsic_is_required_for_robot_frame_publication(tmp_path):
    _write_artifact(tmp_path)
    artifact = SamCalibrationArtifact.load(tmp_path)
    assert artifact.quality.accepted
    footprint = artifact.estimate_horizontal_footprint(LABEL_LID)
    assert footprint.frame == "support-plane-levelled"
    _expect_rejected(lambda: artifact.compute_T_robot_level(None))

    catalog = LabwareCatalog(
        [
            LabwareEntry(
                "dish",
                "petri dish",
                "Labware",
                "cyl",
                (0.045, 0.014),
                transparent=True,
                graspable=True,
                provisional_dims=True,
            )
        ]
    )
    binding = SamLabelBinding(LABEL_LID, "dish_0", "dish", sam_score=0.93)
    _expect_rejected(
        lambda: artifact.to_bench_state([binding], catalog, None)
    )


def test_known_camera_transform_is_composed_and_applied(tmp_path):
    T_level_camera = np.eye(4)
    T_level_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_level_camera[:3, 3] = [0.10, -0.20, 0.30]
    _write_artifact(tmp_path, T_level_camera=T_level_camera)
    artifact = SamCalibrationArtifact.load(tmp_path)
    T_robot_camera = np.eye(4)
    T_robot_camera[:3, 3] = [1.0, 2.0, 3.0]
    expected = T_robot_camera @ np.linalg.inv(T_level_camera)
    actual = artifact.compute_T_robot_level(T_robot_camera)
    assert np.allclose(actual, expected)

    level = artifact.estimate_horizontal_footprint(LABEL_LID)
    robot = artifact.estimate_horizontal_footprint(
        LABEL_LID, T_robot_level=actual
    )
    expected_center = (
        actual[:3, :3] @ level.center_m + actual[:3, 3]
    )
    assert robot.frame == "robot_base"
    assert np.allclose(robot.center_m, expected_center)


def test_bad_rgb_depth_sync_rejects_pose(tmp_path):
    _write_artifact(tmp_path, sync_delta=0.5)
    artifact = SamCalibrationArtifact.load(tmp_path)
    assert not artifact.quality.accepted
    assert any("RGB-depth delta" in issue for issue in artifact.quality.issues)
    _expect_rejected(
        lambda: artifact.estimate_horizontal_footprint(LABEL_LID)
    )


def test_nominal_cad_pose_cannot_change_sam_pose(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_artifact(first, nominal_pose=[0, 0, 0])
    _write_artifact(second, nominal_pose=[99, -42, 8])
    a = SamCalibrationArtifact.load(first)
    b = SamCalibrationArtifact.load(second)
    pa = a.estimate_horizontal_footprint(LABEL_LID)
    pb = b.estimate_horizontal_footprint(LABEL_LID)
    assert np.allclose(pa.center_m, pb.center_m)
    assert np.allclose(pa.rotation, pb.rotation)


def test_actionable_bench_state_requires_recorded_sam_score(tmp_path):
    _write_artifact(tmp_path)
    artifact = SamCalibrationArtifact.load(tmp_path)
    catalog = LabwareCatalog(
        [
            LabwareEntry(
                "dish",
                "petri dish",
                "Labware",
                "cyl",
                (0.045, 0.014),
                transparent=True,
                graspable=True,
                provisional_dims=True,
            )
        ]
    )
    no_score = SamLabelBinding(LABEL_LID, "dish_0", "dish")
    _expect_rejected(
        lambda: artifact.to_bench_state(
            [no_score], catalog, np.eye(4)
        )
    )
    accepted = SamLabelBinding(
        LABEL_LID, "dish_0", "dish", sam_score=0.93
    )
    calibrated = artifact.calibrate(
        [accepted], catalog, np.eye(4)
    )
    assert calibrated.bench_state.frame == "robot_base"
    assert calibrated.bench_state.items[0].confidence == 0.93
    assert calibrated.provenance["bindings"][0]["pose_source"] == "sam_rgbd"

    wrong_shape_catalog = LabwareCatalog(
        [
            LabwareEntry(
                "tiny_dish",
                "wrong tiny dish",
                "Labware",
                "cyl",
                (0.010, 0.010),
            )
        ]
    )
    wrong_shape = SamLabelBinding(
        LABEL_LID, "wrong_0", "tiny_dish", sam_score=0.93
    )
    _expect_rejected(
        lambda: artifact.calibrate(
            [wrong_shape], wrong_shape_catalog, np.eye(4)
        )
    )


def _run_without_pytest():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for index, test in enumerate(
            (
                test_extrinsic_is_required_for_robot_frame_publication,
                test_known_camera_transform_is_composed_and_applied,
                test_bad_rgb_depth_sync_rejects_pose,
                test_nominal_cad_pose_cannot_change_sam_pose,
                test_actionable_bench_state_requires_recorded_sam_score,
            )
        ):
            case = root / str(index)
            case.mkdir()
            test(case)


if __name__ == "__main__":
    _run_without_pytest()
    print("WetRobo SAM calibration checks passed")
