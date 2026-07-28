#!/usr/bin/env python3

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.local_feature_calibration import (
    fit_horizontal_feature_model,
    fit_probe_records,
    load_probe_record,
    recommend_next_probe,
    register_fixed_camera_view,
    verify_probe_records,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


rng = np.random.default_rng(7)
reference = np.zeros((360, 480), np.uint8)
for _ in range(180):
    center = tuple(rng.integers([5, 5], [475, 355]).tolist())
    radius = int(rng.integers(2, 8))
    color = int(rng.integers(80, 256))
    cv2.circle(reference, center, radius, color, -1)
same_view = cv2.convertScaleAbs(reference, alpha=0.95, beta=5)
registration = register_fixed_camera_view(reference, same_view)
assert registration.accepted
assert registration.maximum_corner_motion_px < 2.0

shift = np.float32([[1, 0, 6], [0, 1, 0]])
moved_view = cv2.warpAffine(reference, shift, (480, 360))
moved_registration = register_fixed_camera_view(reference, moved_view)
assert not moved_registration.accepted
assert "camera pose changed" in moved_registration.reason


true_matrix = np.array([[500.0, -100.0], [80.0, 400.0]])
true_uvd_matrix = np.vstack((true_matrix, [300.0, -250.0]))
robot = np.array(
    [
        [0.0058, 0.0001, 0.0002],
        [-0.0001, 0.0057, -0.0001],
        [0.0030, 0.0040, 0.0001],
        [0.0030, -0.0040, -0.0002],
    ]
)
feature = robot[:, :2] @ true_matrix.T
model = fit_horizontal_feature_model(robot, feature)
assert not model.verified
assert np.allclose(model.matrix, true_matrix)
expected = np.array([0.014, -0.021])
try:
    model.solve(
        true_matrix @ expected,
        allow_unchecked_context=True,
    )
    raise AssertionError("provisional calibration was used for motion")
except RuntimeError:
    pass
step = model.solve(
    true_matrix @ expected,
    allow_provisional=True,
    allow_unchecked_context=True,
)
assert step[2] == 0.0
assert np.linalg.norm(step[:2]) <= 0.008001
assert np.dot(step[:2], expected) > 0.0

for invalid_limits in (
    {"max_norm_m": -0.001},
    {"max_axis_m": float("nan")},
):
    try:
        model.solve(
            true_matrix @ expected,
            allow_provisional=True,
            allow_unchecked_context=True,
            **invalid_limits,
        )
        raise AssertionError("invalid step limit was accepted")
    except ValueError:
        pass

for invalid_fit_gate in (
    {"maximum_motion_condition": float("nan")},
    {"maximum_feature_condition": float("nan")},
    {"maximum_residual_rms": float("nan")},
    {"minimum_probe_count": 3.0},
):
    try:
        fit_horizontal_feature_model(
            robot, feature, **invalid_fit_gate
        )
        raise AssertionError("invalid fit gate was accepted")
    except ValueError:
        pass

try:
    fit_horizontal_feature_model(robot[:2], feature[:2])
    raise AssertionError("two probes produced a calibration")
except ValueError:
    pass

uvd_feature = robot[:, :2] @ true_uvd_matrix.T
uvd_model = fit_horizontal_feature_model(
    robot,
    uvd_feature,
    feature_scale=[2.0, 2.0, 20.0],
    feature_components=(0, 1, 2),
)
assert np.allclose(uvd_model.matrix, true_uvd_matrix)

outlier_feature = feature.copy()
outlier_feature[3] += [25.0, -30.0]
robust = fit_horizontal_feature_model(robot, outlier_feature)
assert not robust.verified
assert np.count_nonzero(robust.inlier_mask) == 3
assert np.allclose(robust.matrix, true_matrix, atol=1e-8)

try:
    fit_horizontal_feature_model(
        np.array(
            [[0.006, 0, 0], [0.004, 0, 0], [-0.005, 0, 0]]
        ),
        np.array([[3, 0], [2, 0], [-2.5, 0]]),
    )
    raise AssertionError("rank-one probes produced a model")
except ValueError:
    pass

ill_conditioned_robot = np.array(
    [[0.006, 0.0, 0.0], [0.006, 0.001, 0.0], [-0.006, 0.0, 0.0]]
)
try:
    fit_horizontal_feature_model(
        ill_conditioned_robot,
        ill_conditioned_robot[:, :2] @ true_matrix.T,
    )
    raise AssertionError("ill-conditioned motion design produced a model")
except ValueError as exc:
    assert "condition is too high" in str(exc)

inconsistent_feature = feature[:3].copy()
inconsistent_feature[2] += [20.0, -20.0]
try:
    fit_horizontal_feature_model(robot[:3], inconsistent_feature)
    raise AssertionError("inconsistent visual probes produced a model")
except ValueError as exc:
    assert "no visual consensus" in str(exc)


def stable_context():
    return {
        "schema": "sam_probe_context/v2",
        "scene_config_sha256": "1" * 64,
        "torque_config_sha256": "2" * 64,
        "head_camera_udid": "test-head-udid",
        "head_camera_matrix_rotated": [
            [100, 0, 240],
            [0, 100, 180],
            [0, 0, 1],
        ],
        "head_camera_reference_shape_hw": [360, 480],
        "head_image_shape_hw": [360, 480],
        "head_rotation": "clockwise_90",
        "feature_definition": (
            "lid-left-ellipse/gripper-sam-hsv-terminal-roi-v4"
        ),
        "placement_reference_capture_id": "test-placement",
        "placement_reference_rgb_sha256": "3" * 64,
        "registration_gate": "orb_identity_v1",
        "sam_protocol_version": 1,
        "sam_models": {"lid": "fake-sam", "gripper": "fake-sam"},
        "sam_prompts": {
            "lid": "transparent round petri dish lid with blue cross",
            "gripper": "blue clamp",
        },
        "sam_policy": {
            "lid_prompt_sequence": [
                "transparent round petri dish lid with blue cross",
                "petri dish lid",
                "round transparent plastic dish",
            ],
            "lid_confidence_threshold": 0.05,
            "gripper_prompt": "blue clamp",
            "gripper_confidence_threshold": 0.10,
            "jpeg_quality": 90,
            "preprocess": "identity",
        },
        "depth_frames_requested": 3,
        "feature_layout": [
            "image_u_px",
            "image_v_px",
            "camera_depth_mm",
        ],
        "temporal_depth_method": (
            "fresh-frame-median/rotate-clockwise/nearest-resize-v1"
        ),
        "control_frame": "piper_right_base_xyz_m",
        "observation_pipeline_sha256": "4" * 64,
        "feature_extractor_sha256": "5" * 64,
        "segmentation_selector_sha256": "6" * 64,
    }


def make_record(
    root,
    index,
    actual,
    *,
    image=reference,
    relative_delta=None,
    lid_delta=(0.0, 0.0, 0.0),
    usable=True,
    context_mutation=None,
    requested_xyz_m=None,
):
    run = root / f"run-{index}"
    run.mkdir()
    before_image = run / "before.png"
    after_image = run / "after.png"
    cv2.imwrite(str(before_image), image)
    cv2.imwrite(
        str(after_image),
        cv2.convertScaleAbs(image, alpha=0.98, beta=2),
    )

    def observation_artifacts(prefix, image_path, feature_metadata):
        depth_path = run / f"{prefix}_depth.npz"
        np.savez_compressed(
            depth_path,
            depth_m=np.ones((12, 16), dtype=np.float32),
            camera_matrix=np.array(
                [[100.0, 0.0, 8.0], [0.0, 100.0, 6.0], [0.0, 0.0, 1.0]]
            ),
            source_timestamps=np.array([100.0, 100.1]),
        )
        paths = {
            "raw_image": str(image_path),
            "sam_input_png": str(image_path),
            "sam_request_jpeg_q90": str(image_path),
            "sam_roi_input_png": str(image_path),
            "sam_roi_request_jpeg_q90": str(image_path),
            "overlay_image": str(image_path),
            "lid_mask": str(image_path),
            "gripper_mask": str(image_path),
            "depth_npz": str(depth_path),
        }
        files = {
            field: {
                "path": Path(path).name,
                "sha256": sha256(path),
                "bytes": Path(path).stat().st_size,
                "media_type": (
                    "application/x-npz"
                    if field == "depth_npz"
                    else "image/png"
                ),
            }
            for field, path in paths.items()
        }
        manifest_document = {
            "schema": "sam_head_observation/v2",
            "sequence": index,
            "run_id": f"test-run-{index}",
            "attempt_id": f"test-attempt-{prefix}",
            "files": files,
            "image_shape_hw": list(image.shape[:2]),
            "preprocess": "identity",
            "lid": {
                "prompt": (
                    "transparent round petri dish lid with blue cross"
                ),
                "model": "fake-sam",
            },
            "gripper": {
                "prompt": "blue clamp",
                "model": "fake-sam",
            },
            "feature": {
                key: np.asarray(value, dtype=float).tolist()
                for key, value in feature_metadata.items()
            },
        }
        manifest_path = run / f"{prefix}_observation.json"
        manifest_path.write_text(
            json.dumps(
                manifest_document,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        paths["manifest"] = str(manifest_path)
        return {
            **manifest_document,
            **paths,
            "manifest_sha256": sha256(manifest_path),
            "sha256": {
                field: sha256(path) for field, path in paths.items()
            },
        }

    actual = np.asarray(actual, dtype=float)
    axis = "X" if abs(actual[0]) >= abs(actual[1]) else "Y"
    if requested_xyz_m is None:
        requested = np.zeros(3)
        selected_axis = 0 if axis == "X" else 1
        requested[selected_axis] = np.copysign(
            0.006, actual[selected_axis]
        )
    else:
        requested = np.asarray(requested_xyz_m, dtype=float)
    before_pose = np.array([1, 0, 0, 0, 0.30, 0.00, 0.80], dtype=float)
    final_pose = before_pose.copy()
    final_pose[4:7] += actual
    if relative_delta is None:
        relative_delta = true_uvd_matrix @ actual[:2]
    relative_delta = np.asarray(relative_delta, dtype=float)
    if relative_delta.shape == (2,):
        relative_delta = np.r_[relative_delta, 0.0]
    assert relative_delta.shape == (3,)
    lid_delta = np.asarray(lid_delta, dtype=float)
    gripper_delta = relative_delta + lid_delta
    before_gripper = np.array([100.0, 80.0, 900.0])
    before_lid = np.array([220.0, 260.0, 840.0])
    after_gripper = before_gripper + gripper_delta
    after_lid = before_lid + lid_delta
    before_artifacts = observation_artifacts(
        "before",
        before_image,
        {
            "lid_grasp_feature": before_lid,
            "gripper_feature": before_gripper,
            "error": before_lid - before_gripper,
        },
    )
    after_artifacts = observation_artifacts(
        "after",
        after_image,
        {
            "lid_grasp_feature": after_lid,
            "gripper_feature": after_gripper,
            "error": after_lid - after_gripper,
        },
    )
    stable = stable_context()
    if context_mutation is not None:
        context_mutation(stable)
    canonical = json.dumps(
        stable, sort_keys=True, separators=(",", ":")
    ).encode()
    context = {
        "stable": stable,
        "context_id_sha256": hashlib.sha256(canonical).hexdigest(),
        "anchor_head_raw_image": str(before_image),
        "anchor_head_raw_sha256": sha256(before_image),
        "host_boot_id": "test-boot",
        "placement_registration": {
            "accepted": True,
            "matches": 1000,
            "inliers": 900,
            "inlier_fraction": 0.9,
            "median_inlier_error_px": 0.1,
            "maximum_corner_motion_px": 0.2,
        },
    }
    hold = {
        "verified": True,
        "torque_within_limit": True,
        "torque_limit_nm": [1.0] * 6,
        "xyz_span_m": [0.0, 0.0, 0.0],
        "joint_span_rad": [0.0] * 6,
        "max_abs_torque_nm": [0.1] * 6,
        "final_state": {
            "pose_wxyz_xyz": final_pose.tolist(),
            "joint_position_rad": [0.0] * 6,
            "joint_torque_nm": [0.1] * 6,
            "monotonic_s": 2.0,
        },
    }
    final_hold = copy.deepcopy(hold)
    final_hold["final_state"]["monotonic_s"] = 3.0
    pixel_signal = float(np.linalg.norm(relative_delta[:2]))
    pixel_noise = max(0.5, float(np.linalg.norm(lid_delta[:2])))
    signal_to_noise = pixel_signal / pixel_noise
    record = {
        "schema": "sam_horizontal_probe/v2",
        "record_id": f"{index:032x}",
        "created_at_utc": "2026-07-28T00:00:00+00:00",
        "status": f"SINGLE_{axis}_PROBE_COMMITTED",
        "authorization": {"motion_token_sha256": "4" * 64},
        "execution": {
            "stage": "committed",
            "motion_attempted": True,
            "motion_command_completed": True,
            "immediate_motion_validated": True,
            "settled_motion_validated": True,
            "failure_timing": None,
        },
        "context": context,
        "motion": {
            "requested_xyz_m": requested.tolist(),
            "actual_immediate_xyz_m": actual.tolist(),
            "actual_settled_xyz_m": actual.tolist(),
            "before_state": {
                "pose_wxyz_xyz": before_pose.tolist(),
                "joint_position_rad": [0.0] * 6,
                "joint_torque_nm": [0.1] * 6,
                "monotonic_s": 1.0,
            },
            "post_motion_state": {
                "pose_wxyz_xyz": final_pose.tolist(),
                "joint_position_rad": [0.0] * 6,
                "joint_torque_nm": [0.1] * 6,
                "monotonic_s": 1.5,
            },
            "initial_hold": copy.deepcopy(hold),
            "hold": final_hold,
            "observation_xyz_shift_m": [0.0, 0.0, 0.0],
            "observation_joint_shift_rad": [0.0] * 6,
        },
        "observation": {
            "before": {
                "feature_error": (before_lid - before_gripper).tolist(),
                "gripper_feature": before_gripper.tolist(),
                "lid_grasp_feature": before_lid.tolist(),
                "head_image": str(before_image),
                "head_image_sha256": sha256(before_image),
                "head_raw_image": str(before_image),
                "head_raw_image_sha256": sha256(before_image),
                "right_image": str(before_image),
                "right_image_sha256": sha256(before_image),
                "head_timestamp": 100.0,
                "head_artifacts": before_artifacts,
            },
            "after": {
                "feature_error": (after_lid - after_gripper).tolist(),
                "gripper_feature": after_gripper.tolist(),
                "lid_grasp_feature": after_lid.tolist(),
                "head_image": str(after_image),
                "head_image_sha256": sha256(after_image),
                "head_raw_image": str(after_image),
                "head_raw_image_sha256": sha256(after_image),
                "right_image": str(after_image),
                "right_image_sha256": sha256(after_image),
                "head_timestamp": 101.0,
                "head_artifacts": after_artifacts,
            },
            "gripper_feature_delta": gripper_delta.tolist(),
            "lid_feature_delta": lid_delta.tolist(),
            "relative_feature_delta": relative_delta.tolist(),
        },
        "quality": {
            "pixel_signal_norm": pixel_signal,
            "pixel_noise_norm": pixel_noise,
            "signal_to_noise": signal_to_noise,
            "usable_for_fit": bool(usable),
            "reasons": (
                []
                if usable
                else [
                    reason
                    for reason, rejected in (
                        (
                            "image motion was below 2 px",
                            pixel_signal < 2.0,
                        ),
                        (
                            "relative image motion SNR was below 3",
                            signal_to_noise < 3.0,
                        ),
                    )
                    if rejected
                ]
            ),
        },
    }
    path = run / "probe_record.json"
    path.write_text(json.dumps(record))
    return path, record


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    x_path, x_record = make_record(
        root, 1, [0.0058, 0.0001, 0.0002]
    )
    y_path, _ = make_record(
        root, 2, [-0.0001, 0.0057, -0.0001]
    )
    diagonal_path, _ = make_record(
        root, 3, [0.0030, 0.0040, 0.0001]
    )
    negative_x_path, _ = make_record(
        root, 4, [-0.0058, -0.0001, 0.0]
    )
    negative_y_path, _ = make_record(
        root, 14, [-0.0001, -0.0058, 0.0]
    )

    sample = load_probe_record(x_path)
    assert np.allclose(sample.actual_xyz_m, [0.0058, 0.0001, 0.0002])
    assert np.allclose(
        sample.feature_delta[:2],
        true_matrix @ np.array([0.0058, 0.0001]),
    )
    negative_sample = load_probe_record(negative_x_path)
    assert negative_sample.requested_xyz_m[0] < 0.0
    assert negative_sample.actual_xyz_m[0] < 0.0

    try:
        fit_probe_records([x_path, y_path])
        raise AssertionError("two probe records produced a calibration")
    except ValueError as exc:
        assert "at least 3 probe records" in str(exc)

    provisional, samples = fit_probe_records(
        [x_path, y_path, diagonal_path]
    )
    assert len(samples) == 3
    assert not provisional.verified
    assert np.allclose(provisional.matrix, true_matrix)
    try:
        provisional.solve(
            np.array([10.0, -5.0]),
            allow_unchecked_context=True,
        )
        raise AssertionError("unverified record model was used for motion")
    except RuntimeError:
        pass

    partial, partial_report, partial_samples = verify_probe_records(
        provisional, [negative_x_path]
    )
    assert not partial.verified
    assert not partial_report.accepted
    assert partial_report.horizontal_rank == 1
    assert len(partial_samples) == 1
    assert "at least two held-out probes are required" in (
        partial_report.reasons
    )
    try:
        partial.solve(
            np.array([10.0, -5.0]),
            allow_unchecked_context=True,
        )
        raise AssertionError("one held-out probe enabled solve")
    except RuntimeError:
        pass

    verified, report, held_out = verify_probe_records(
        provisional, [negative_x_path, negative_y_path]
    )
    assert verified.verified
    assert report.accepted
    assert report.horizontal_rank == 2
    assert all(
        positive and negative
        for positive, negative in report.combined_signed_axis_coverage
    )
    assert len(held_out) == 2
    assert report.normalized_residual_rms < 1e-8
    positive_x_path, _ = make_record(
        root, 15, [0.0054, 0.0002, 0.0]
    )
    positive_y_path, _ = make_record(
        root, 16, [0.0002, 0.0054, 0.0]
    )
    same_sign, same_sign_report, _ = verify_probe_records(
        provisional, [positive_x_path, positive_y_path]
    )
    assert same_sign_report.horizontal_rank == 2
    assert not same_sign.verified
    assert (
        "combined probes lack positive and negative X/Y excitation"
        in same_sign_report.reasons
    )
    try:
        verified.solve(np.array([10.0, -5.0]))
        raise AssertionError("verified model bypassed context checks")
    except RuntimeError as exc:
        assert "requires solve_for_observation" in str(exc)
    try:
        verified.solve(
            np.array([10.0, -5.0]),
            allow_unchecked_context=True,
        )
        raise AssertionError("record-backed model allowed unchecked solve")
    except RuntimeError as exc:
        assert "requires solve_for_observation" in str(exc)
    current_relative = verified.reference_midpoint_feature.copy()
    current_error = -current_relative[:2]
    assert verified.applicability_xy_radius_m <= 0.020
    assert verified.applicability_z_radius_m <= 0.003
    assert verified.applicability_uv_radius_px <= 40.0
    assert verified.applicability_depth_radius_mm <= 25.0
    applicability = verified.validate_applicability(
        context_fingerprint=verified.context_fingerprint,
        current_raw_head_image=verified.reference_head_image,
        current_ee_xyz_m=verified.reference_midpoint_xyz_m,
        current_orientation_wxyz=verified.reference_orientation_wxyz,
        current_relative_feature=current_relative,
    )
    assert applicability.accepted
    checked_step = verified.solve_for_observation(
        current_error,
        context_fingerprint=verified.context_fingerprint,
        current_raw_head_image=verified.reference_head_image,
        current_ee_xyz_m=verified.reference_midpoint_xyz_m,
        current_orientation_wxyz=verified.reference_orientation_wxyz,
        current_relative_feature=current_relative,
    )
    assert np.isfinite(checked_step).all()
    try:
        verified.solve_for_observation(
            current_error,
            context_fingerprint=verified.context_fingerprint,
            current_raw_head_image=verified.reference_head_image,
            current_ee_xyz_m=verified.reference_midpoint_xyz_m,
            current_orientation_wxyz=verified.reference_orientation_wxyz,
            current_relative_feature=current_relative,
            max_axis_m=-0.001,
        )
        raise AssertionError("checked solve accepted a negative step limit")
    except ValueError:
        pass
    wrong_context = verified.validate_applicability(
        context_fingerprint="f" * 64,
        current_raw_head_image=verified.reference_head_image,
        current_ee_xyz_m=verified.reference_midpoint_xyz_m,
        current_orientation_wxyz=verified.reference_orientation_wxyz,
        current_relative_feature=current_relative,
    )
    assert not wrong_context.accepted
    outside_locality = verified.validate_applicability(
        context_fingerprint=verified.context_fingerprint,
        current_raw_head_image=verified.reference_head_image,
        current_ee_xyz_m=(
            verified.reference_midpoint_xyz_m + [0.05, 0.0, 0.0]
        ),
        current_orientation_wxyz=verified.reference_orientation_wxyz,
        current_relative_feature=current_relative,
    )
    assert not outside_locality.accepted
    moved_applicability = verified.validate_applicability(
        context_fingerprint=verified.context_fingerprint,
        current_raw_head_image=moved_view,
        current_ee_xyz_m=verified.reference_midpoint_xyz_m,
        current_orientation_wxyz=verified.reference_orientation_wxyz,
        current_relative_feature=current_relative,
    )
    assert not moved_applicability.accepted
    assert verified.artifact_sha256 != provisional.artifact_sha256

    uvd_provisional, _ = fit_probe_records(
        [x_path, y_path, diagonal_path],
        feature_components=(0, 1, 2),
    )
    uvd_verified, uvd_report, _ = verify_probe_records(
        uvd_provisional, [negative_x_path, negative_y_path]
    )
    assert uvd_report.accepted
    assert uvd_verified.verified
    assert np.allclose(uvd_verified.matrix, true_uvd_matrix)

    try:
        verify_probe_records(provisional, [x_path])
        raise AssertionError("training record was reused as held-out data")
    except ValueError as exc:
        assert "used for model fitting" in str(exc)
    tampered_model = copy.deepcopy(provisional)
    tampered_model.matrix[0, 0] += 1.0
    try:
        verify_probe_records(tampered_model, [negative_x_path])
        raise AssertionError("tampered calibration model was verified")
    except ValueError as exc:
        assert "artifact hash mismatch" in str(exc)
    tampered_verified = copy.deepcopy(verified)
    tampered_verified.matrix[0, 0] += 1.0
    try:
        tampered_verified.solve(
            current_error, allow_unchecked_context=True
        )
        raise AssertionError("tampered verified model was used for solve")
    except RuntimeError as exc:
        assert "artifact hash mismatch" in str(exc)

    bad_verification_path, _ = make_record(
        root,
        5,
        [-0.0001, -0.0058, 0.0],
        relative_delta=-(
            true_uvd_matrix @ np.array([-0.0001, -0.0058])
        ),
    )
    rejected, rejected_report, _ = verify_probe_records(
        provisional, [negative_x_path, bad_verification_path]
    )
    assert not rejected.verified
    assert not rejected_report.accepted
    assert "held-out feature direction is inconsistent" in (
        rejected_report.reasons
    )

    try:
        fit_probe_records([x_path, x_path, y_path])
        raise AssertionError("duplicate probe record was counted twice")
    except ValueError as exc:
        assert "duplicate probe record" in str(exc)

    mismatch_path, _ = make_record(
        root,
        6,
        [0.0001, 0.0058, 0.0],
        context_mutation=lambda stable: stable.update(
            {"head_camera_udid": "different-camera"}
        ),
    )
    try:
        fit_probe_records([x_path, y_path, mismatch_path])
        raise AssertionError("context mismatch was accepted")
    except ValueError as exc:
        assert "contexts do not match" in str(exc)

    moved_path, _ = make_record(
        root, 7, [0.0001, 0.0058, 0.0], image=moved_view
    )
    try:
        fit_probe_records([x_path, y_path, moved_path])
        raise AssertionError("moved camera was accepted")
    except ValueError as exc:
        assert "head camera moved" in str(exc)

    low_path, low_record = make_record(
        root,
        8,
        [0.006, 0.0, 0.0],
        relative_delta=[1.9, 0.0],
        usable=False,
    )
    try:
        load_probe_record(low_path)
        raise AssertionError("low-pixel probe was accepted")
    except ValueError as exc:
        assert "not marked usable" in str(exc)
    recommendation = recommend_next_probe([low_path])
    assert recommendation["requested_xyz_m"][1] > 0.0
    assert abs(recommendation["requested_xyz_m"][0]) < 1e-9
    balanced = recommend_next_probe([x_path, y_path])
    balanced_xy = np.asarray(balanced["requested_xyz_m"][:2])
    attempted_sum = (
        np.asarray(x_record["motion"]["requested_xyz_m"][:2])
        + np.array([0.0, 0.006])
    )
    assert np.dot(balanced_xy, attempted_sum) < 0.0
    assert np.isclose(np.linalg.norm(balanced_xy), 0.006)
    assert np.count_nonzero(np.abs(balanced_xy) > 1e-12) == 1

    wrong_direction_path, _ = make_record(
        root,
        9,
        [0.0058, 0.0, 0.0],
        requested_xyz_m=[-0.006, 0.0, 0.0],
    )
    try:
        load_probe_record(wrong_direction_path)
        raise AssertionError("opposite requested and actual motion was accepted")
    except ValueError as exc:
        assert "did not follow requested direction" in str(exc)

    derived_path, derived_record = make_record(
        root, 10, [0.0058, 0.0, 0.0]
    )
    derived_record["observation"]["relative_feature_delta"][0] += 1.0
    derived_path.write_text(json.dumps(derived_record))
    try:
        load_probe_record(derived_path)
        raise AssertionError("tampered derived feature delta was accepted")
    except ValueError as exc:
        assert "relative feature delta is inconsistent" in str(exc)

    manifest_cross_path, manifest_cross_record = make_record(
        root, 17, [0.0058, 0.0, 0.0]
    )
    for phase in ("before", "after"):
        manifest_cross_record["observation"][phase][
            "gripper_feature"
        ][0] += 1.0
        manifest_cross_record["observation"][phase][
            "feature_error"
        ][0] -= 1.0
    manifest_cross_path.write_text(json.dumps(manifest_cross_record))
    try:
        load_probe_record(manifest_cross_path)
        raise AssertionError("manifest/record feature mismatch was accepted")
    except ValueError as exc:
        assert "manifest and record" in str(exc)

    shift_path, shift_record = make_record(
        root, 18, [0.0058, 0.0, 0.0]
    )
    shift_record["motion"]["observation_xyz_shift_m"][0] = 0.0001
    shift_path.write_text(json.dumps(shift_record))
    try:
        load_probe_record(shift_path)
        raise AssertionError("tampered observation shift was accepted")
    except ValueError as exc:
        assert "Cartesian shift is inconsistent" in str(exc)

    artifact_path, artifact_record = make_record(
        root, 11, [0.0058, 0.0, 0.0]
    )
    artifact_image = Path(
        artifact_record["observation"]["before"]["head_raw_image"]
    )
    artifact_image.write_bytes(artifact_image.read_bytes() + b"tamper")
    try:
        load_probe_record(artifact_path)
        raise AssertionError("tampered image artifact was accepted")
    except ValueError as exc:
        assert "artifact hash mismatch" in str(exc)

    committed_path, committed_record = make_record(
        root, 19, [0.0058, 0.0, 0.0]
    )
    expected_record_hash = sha256(committed_path)
    assert (
        load_probe_record(
            committed_path,
            expected_record_sha256=expected_record_hash,
        ).content_sha256
        == expected_record_hash
    )
    try:
        load_probe_record(
            committed_path,
            expected_record_sha256="0" * 64,
        )
        raise AssertionError("wrong external record digest was accepted")
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    committed_record["execution"]["settled_motion_validated"] = False
    committed_path.write_text(json.dumps(committed_record))
    try:
        load_probe_record(committed_path)
        raise AssertionError("incomplete execution journal was accepted")
    except ValueError as exc:
        assert "not fully committed" in str(exc)

    high_torque = copy.deepcopy(x_record)
    high_torque_hold = high_torque["motion"]["initial_hold"]
    high_torque_hold["max_abs_torque_nm"][0] = 1.1
    high_torque_hold["final_state"]["joint_torque_nm"][0] = 1.1
    high_torque_hold["torque_within_limit"] = False
    high_torque_hold["verified"] = False
    x_path.write_text(json.dumps(high_torque))
    try:
        load_probe_record(x_path)
        raise AssertionError("high-torque probe was accepted")
    except ValueError as exc:
        assert "torque was outside" in str(exc)

print("local feature calibration checks passed")
