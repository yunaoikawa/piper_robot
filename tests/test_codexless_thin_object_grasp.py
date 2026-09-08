from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import copy
import json

import cv2
import numpy as np
import pytest

from rollout.grasp_window import GraspWindowTemplate
from rollout.thin_object_grasp import (
    ClosureCalibration,
    ConsecutiveSuccessLedger,
    observe_local_blue_evidence_target,
    observe_marked_target,
    select_local_blue_evidence_marker,
    select_relocated_target_marker,
    track_target_center_lk,
    target_follow_evidence,
)
from rollout.tapped_lid_target import register_fixed_head
from src.run_codexless_thin_object_grasp import (
    _load_pending_preclose_alignment,
    _load_runtime_alignment,
    _preclose_correction_replay_alignment,
    _runtime_fast_replay_alignment,
    _runtime_head_continuity_marker,
    _save_pending_preclose_alignment,
    audit_profile,
    load_profile,
)
import src.run_codexless_thin_object_grasp as runner


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src/configs/pasteur_codexless_thin_object_grasp.json"
GOAL = ROOT / "data/runs/pasteur/live_lid_grasp_20260805/right_marker_goal_selection.json"
PRECLOSE = Path(
    "/tmp/record3d_checks/2026-08-05/2026-08-05/"
    "20260805T090908.648901Z_right_right_level_fixed_preclose_333ebaa7/"
    "derived/head_rgb_landscape.png"
)


def test_closure_uses_calibrated_populations_not_absolute_threshold():
    calibration = ClosureCalibration((0.0048,), (0.54, 0.58))
    assert calibration.classify(0.542)["nonempty"] is True
    assert calibration.classify(0.006)["nonempty"] is False


def test_follow_requires_small_normalized_motion_and_persistent_obstruction():
    nonempty = {"nonempty": True}
    accepted = target_follow_evidence(
        (100.0, 100.0),
        (103.0, 101.0),
        (480, 640),
        maximum_displacement_diagonal_fraction=0.02,
        closure_before=nonempty,
        closure_after=nonempty,
    )
    assert accepted["accepted"] is True
    slipped = target_follow_evidence(
        (100.0, 100.0),
        (150.0, 100.0),
        (480, 640),
        maximum_displacement_diagonal_fraction=0.02,
        closure_before=nonempty,
        closure_after=nonempty,
    )
    assert slipped["accepted"] is False


def test_consecutive_successes_reset_after_failure():
    ledger = ConsecutiveSuccessLedger(3)
    ledger.record(True)
    ledger.record(True)
    assert ledger.record(False)["consecutive_successes"] == 0
    ledger.record(True)
    ledger.record(True)
    assert ledger.record(True)["complete"] is True


def test_runtime_alignment_replays_camera_branch_not_descent_ik_branch(tmp_path):
    profile = copy.deepcopy(load_profile(PROFILE))
    profile["target_identity"]["runtime_alignment_calibration"] = str(
        tmp_path / "runtime_alignment.json"
    )
    semantic_q = np.asarray([0.15, 0.05, -0.10, 0.14, -0.01, 2.20])
    executed_q = np.asarray([-0.35, 0.07, -0.18, 0.87, 0.39, 1.51])
    report = runner._save_runtime_alignment(
        profile,
        {
            "target_center_scene_xyz_m": [0.0, 0.8, -0.5],
            "marker": {"center_px": [100.0, 200.0]},
        },
        {
            "hover": {
                "error_uv": [0.02, -0.03],
                "normalized_center_error": 0.04,
                "hover_q_physical_rad": semantic_q.tolist(),
                "measured_hover_q_physical_rad": (semantic_q + 0.001).tolist(),
                "observation": {
                    "source": "blue_marker",
                    "tool_frame": {
                        "origin_px": [344.0, 333.0],
                        "forward_xy": [-0.8, 0.6],
                        "lateral_xy": [0.6, 0.8],
                        "scale_px": 153.0,
                        "cyan_pixels": 1000,
                        "light_pad_pixels": 200,
                    },
                },
            }
        },
        executed_q,
        source_run="test",
        servo_state={"last_semantic_low_q_physical_rad": semantic_q.tolist()},
    )
    value = report["value"]
    np.testing.assert_allclose(value["low_q_physical_rad"], semantic_q)
    np.testing.assert_allclose(
        value["semantic_reacquisition_low_q_physical_rad"], semantic_q
    )
    np.testing.assert_allclose(
        value["executed_descent_low_q_physical_rad"], executed_q
    )
    np.testing.assert_allclose(
        value["servo_seed"]["best_low_q_physical_rad"], semantic_q
    )
    np.testing.assert_allclose(
        value["hover_q_physical_rad"], semantic_q + 0.001
    )
    assert value["tool_frame"]["scale_px"] == 153.0
    replay_q, loaded = runner._load_runtime_alignment(
        profile, {"target_center_scene_xyz_m": [0.0, 0.8, -0.5]}
    )
    np.testing.assert_allclose(replay_q, semantic_q)
    np.testing.assert_allclose(
        loaded["hover_q_physical_rad"], semantic_q + 0.001
    )
    assert loaded["tool_frame"]["scale_px"] == 153.0


def test_persisted_camera_visible_hover_is_replayed_exactly():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low_q = np.asarray(
        [
            0.18609088411462485,
            0.04015884155129084,
            -0.10513443051175639,
            0.14309088394413758,
            -0.024984570559591,
            2.19653200086162,
        ]
    )
    hover_q = np.asarray(
        [
            0.005846852902323008,
            0.10302678495645523,
            -0.23841197788715363,
            -0.20729275047779083,
            0.2779087722301483,
            2.5340261459350586,
        ]
    )
    report = runner._audit_camera_visible_hover_seed(
        profile, fk, low_q=low_q, hover_q=hover_q
    )
    assert report["accepted"] is True
    assert report["role"] == "persisted_camera_visible_hover"
    np.testing.assert_allclose(report["q_physical_rad"], hover_q)
    assert report["fresh_wrist_alignment_required"] is True
    assert report["closure_authorized"] is False


def test_free_space_camera_progress_is_replayable_without_descent_authority():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low_q = np.asarray(
        [
            -0.03140956102174967,
            0.9305339968207808,
            -0.36159481481484934,
            0.1693694238258889,
            -0.3886659890297031,
            2.200711976464044,
        ]
    )
    camera_q = np.asarray(
        [
            -0.16317082941532135,
            0.9020035862922668,
            -0.5816484093666077,
            -0.865351676940918,
            0.16554448008537292,
            2.923269510269165,
        ]
    )
    report = runner._audit_camera_visible_hover_seed(
        profile, fk, low_q=low_q, hover_q=camera_q
    )
    assert report["accepted"] is True
    assert report["height_above_low_m"] > profile["trajectory"][
        "verification_lift_m"
    ]
    assert report["hover_level"]["accepted"] is False
    assert report["hover_level"]["combined_tilt_deg"] > 12.0
    assert report["hover_level"]["tip_height_difference_m"] > 0.035
    assert report["descent_level_authorized"] is False
    assert report["closure_authorized"] is False


def test_relocated_marker_excludes_registered_stationary_anchor():
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    def cross(center):
        x, y = center
        cv2.rectangle(image, (x - 18, y - 5), (x + 18, y + 5), (255, 0, 0), -1)
        cv2.rectangle(image, (x - 5, y - 18), (x + 5, y + 18), (255, 0, 0), -1)

    cross((70, 90))
    cross((230, 150))
    selected = select_relocated_target_marker(
        image,
        homography_reference_to_current=np.eye(3),
        reference_target_center_px=(170, 90),
        reference_target_component_pixels=693,
        stationary_anchor_centers_px=[(70, 90)],
        maximum_target_displacement_diagonal_fraction=0.4,
        minimum_component_area_scale=0.3,
        maximum_component_area_scale=3.0,
    )
    assert np.linalg.norm(np.asarray(selected.center_px) - (230, 150)) < 2.0
    assert selected.excluded_anchor_count == 1


def test_relocated_marker_recovers_when_target_overlaps_registered_anchor():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (142, 115), (178, 125), (255, 0, 0), -1)
    cv2.rectangle(image, (155, 102), (165, 138), (255, 0, 0), -1)
    selected = select_relocated_target_marker(
        image,
        homography_reference_to_current=np.eye(3),
        reference_target_center_px=(175, 120),
        reference_target_component_pixels=693,
        stationary_anchor_centers_px=[(160, 120)],
        maximum_anchor_displacement_diagonal_fraction=0.05,
        maximum_target_displacement_diagonal_fraction=0.2,
        minimum_component_area_scale=0.3,
        maximum_component_area_scale=3.0,
    )
    assert selected.anchor_overlap_fallback_used is True
    np.testing.assert_allclose(selected.center_px, (160, 120), atol=2.0)


def test_local_blue_evidence_prefers_direct_mark_over_faint_old_refraction():
    image = np.full((240, 320, 3), 55, dtype=np.uint8)
    # A weak refracted satellite stays at the old pixel while the physical
    # object and its stronger mark move within the fixed-camera envelope.
    cv2.rectangle(image, (94, 94), (106, 106), (90, 78, 76), -1)
    cv2.rectangle(image, (142, 122), (162, 142), (165, 75, 50), -1)
    selected = select_local_blue_evidence_marker(
        image,
        reference_target_center_px=(100, 100),
        reference_target_component_pixels=169,
        maximum_target_displacement_diagonal_fraction=0.20,
        minimum_component_area_scale=0.1,
        maximum_component_area_scale=4.0,
    )
    np.testing.assert_allclose(selected.center_px, (152, 132), atol=2.0)
    assert selected.component_pixels > 300


def test_brighter_head_frame_reacquires_moved_lid_mark_not_old_refraction():
    image_path = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v4"
        / "preflight_head.png"
    )
    if not image_path.exists():
        return
    profile = load_profile(PROFILE)
    runtime = json.loads(
        Path(profile["target_identity"]["runtime_alignment_calibration"])
        .read_text()
    )
    previous = runtime["head_marker"]
    selected = select_local_blue_evidence_marker(
        cv2.imread(str(image_path)),
        reference_target_center_px=previous["center_px"],
        reference_target_component_pixels=previous["component_pixels"],
        maximum_target_displacement_diagonal_fraction=profile[
            "head_localization"
        ]["maximum_runtime_head_color_continuity_diagonal_fraction"],
        minimum_component_area_scale=profile["head_localization"][
            "minimum_runtime_head_continuity_area_scale"
        ],
        maximum_component_area_scale=profile["head_localization"][
            "maximum_runtime_head_continuity_area_scale"
        ],
    )
    np.testing.assert_allclose(selected.center_px, (888.7, 762.2), atol=3.0)


def test_reviewed_head_identity_tracks_lid_not_moving_blue_arm():
    profile = load_profile(PROFILE)
    reference = cv2.imread(profile["target_identity"]["head_reference_image"])
    current_path = (
        ROOT
        / "data/runs/pasteur/codexless_preflight_20260805_current/"
        "head_registration/preflight_head.png"
    )
    if reference is None or not current_path.exists():
        return
    current = cv2.imread(str(current_path))
    registration = register_fixed_head(reference, current)
    assert registration.accepted is True
    settings = profile["head_localization"]
    selected = select_relocated_target_marker(
        current,
        homography_reference_to_current=registration.homography,
        reference_target_center_px=settings["reference_target_marker_center_px"],
        reference_target_component_pixels=settings[
            "reference_target_component_pixels"
        ],
        stationary_anchor_centers_px=settings[
            "stationary_anchor_reference_centers_px"
        ],
        maximum_target_displacement_diagonal_fraction=settings[
            "maximum_target_displacement_diagonal_fraction"
        ],
        minimum_component_area_scale=settings["minimum_component_area_scale"],
        maximum_component_area_scale=settings["maximum_component_area_scale"],
    )
    np.testing.assert_allclose(selected.center_px, (844.73, 709.49), atol=3.0)
    assert np.linalg.norm(np.asarray(selected.center_px) - (946.11, 727.88)) > 90


def test_runtime_continuity_prefers_full_marker_over_nearby_reflection(tmp_path):
    image_path = (
        ROOT
        / "data/runs/pasteur/codexless_semantic_margin_hover_20260805/preflight_head.png"
    )
    if not image_path.exists():
        return
    profile = load_profile(PROFILE)
    runtime_path = Path(
        profile["target_identity"]["runtime_alignment_calibration"]
    )
    runtime = json.loads(runtime_path.read_text())
    runtime["head_marker"] = {
        "center_px": [844.7175925925926, 709.5138888888889],
        "component_pixels": 216,
    }
    fixed_runtime = tmp_path / "runtime_alignment.json"
    fixed_runtime.write_text(json.dumps(runtime))
    profile["target_identity"]["runtime_alignment_calibration"] = str(
        fixed_runtime
    )
    marker, report = _runtime_head_continuity_marker(
        profile, cv2.imread(str(image_path))
    )
    assert report["accepted"] is True
    assert report["proxy_used"] is False
    assert report["reacquisition_method"] == (
        "full_marker_supersedes_nearby_reflection"
    )
    np.testing.assert_allclose(marker.center_px, (890.56, 761.80), atol=1.0)
    assert report["rejected_proxy_component_area_scale"] < profile[
        "head_localization"
    ]["minimum_direct_runtime_head_component_area_scale"]
    assert np.linalg.norm(
        np.asarray(report["rejected_proxy_component_center_px"])
        - np.asarray(marker.center_px)
    ) > 40


def test_current_bright_head_frame_rejects_small_dish_reflection():
    image_path = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v15"
        / "preflight_head.png"
    )
    if not image_path.exists():
        return
    profile = load_profile(PROFILE)
    marker, report = _runtime_head_continuity_marker(
        profile, cv2.imread(str(image_path))
    )
    assert report["proxy_used"] is False
    np.testing.assert_allclose(marker.center_px, (916.93, 755.36), atol=1.0)


def test_saved_success_target_is_selected_by_tool_relative_goal_not_area():
    if not PRECLOSE.exists():
        return
    image = cv2.imread(str(PRECLOSE))
    selection = json.loads(GOAL.read_text())
    observation = observe_marked_target(
        image,
        GraspWindowTemplate.from_dict(selection["template"]),
        reference_component_area_per_tool_scale_sq=0.032292655989660676,
    )
    assert observation.grasp_window.normalized_center_error < 0.10
    assert observation.component_pixels == 994
    assert observation.component_area_per_tool_scale_sq > 0.03


def test_preclose_proximity_rejects_gripper_reflection():
    image_path = (
        ROOT
        / "data/runs/pasteur/codexless_proxy_fast_cartesian_grasp_20260805/"
        "attempt_01/preclose.png"
    )
    if not image_path.exists():
        return
    image = cv2.imread(str(image_path))
    template = GraspWindowTemplate.from_dict(json.loads(GOAL.read_text())["template"])
    observation = observe_marked_target(
        image,
        template,
        reference_component_area_per_tool_scale_sq=0.032292655989660676,
        minimum_reference_area_fraction=0.1,
        maximum_reference_area_fraction=3.0,
        proximity_score_weight=2.0,
    )
    np.testing.assert_allclose(observation.center_px, (247.64, 344.18), atol=3.0)


def test_changed_lighting_prefers_complete_marker_over_cross_fragment():
    image_path = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v50/"
        "attempt_01/preclose.png"
    )
    if not image_path.exists():
        return
    profile = load_profile(PROFILE)
    image = cv2.imread(str(image_path))
    template = GraspWindowTemplate.from_dict(
        json.loads(
            Path(profile["target_identity"]["grasp_window_selection"]).read_text()
        )["template"]
    )
    expected = runner.ToolImageFrame(
        origin_px=(344.0, 333.0),
        forward_xy=(-0.7978906986914203, 0.6028021507441036),
        lateral_xy=(0.6028021507441036, 0.7978906986914203),
        scale_px=153.09680971010064,
        cyan_pixels=10949,
        light_pad_pixels=1679,
    )
    observation = observe_marked_target(
        image,
        template,
        reference_component_area_per_tool_scale_sq=profile["target_identity"][
            "right_reference_component_area_per_tool_scale_sq"
        ],
        minimum_reference_area_fraction=profile["target_identity"][
            "minimum_right_reference_area_fraction"
        ],
        maximum_reference_area_fraction=profile["target_identity"][
            "maximum_right_reference_area_fraction"
        ],
        proximity_score_weight=profile["perception"][
            "preclose_proximity_score_weight"
        ],
        expected_tool_frame=expected,
        prefer_expected_tool_frame=True,
        prefer_cross_shape=False,
    )
    assert observation.component_pixels == 1142
    np.testing.assert_allclose(observation.center_px, (222.1, 300.1), atol=1.0)


def test_profile_compiles_and_audits_level_and_non_support_contacts():
    profile = load_profile(PROFILE)
    report = audit_profile(profile)
    assert report["accepted"] is True
    assert report["safe_level"]["accepted"] is True
    assert report["preclose_level"]["accepted"] is True
    assert report["non_support_contacts"] == []
    target = profile["target_identity"]
    template = GraspWindowTemplate.from_dict(
        json.loads(Path(target["grasp_window_selection"]).read_text())["template"]
    )
    assert target["canonical_hover_goal_source"] == (
        "successful_demo_approach_frame_60_blue_marker_center"
    )
    assert target["canonical_preclose_goal_source"] == (
        "successful_demo_grip_start_frame_82_blue_marker_center"
    )
    assert target["grasp_window_selection"].endswith(
        "pasteur_grasp_window_selection.json"
    )
    fixed_tool_frame = runner.ToolImageFrame(
        **target["fixed_right_tool_frame"]
    )
    assert fixed_tool_frame.scale_px > 100
    np.testing.assert_allclose(
        target["canonical_hover_goal_uv"],
        [0.8326834802433452, -0.7531536116793749],
    )
    np.testing.assert_allclose(
        target["canonical_preclose_goal_uv"],
        [0.6232310734519456, -0.6762500174103647],
    )
    # The mask template is the transparent ellipse centre, while runtime
    # perception tracks the blue marker.  Conflating them created a centimetre
    # scale physical offset, so they must intentionally remain distinct.
    assert np.linalg.norm(
        np.asarray(target["canonical_preclose_goal_uv"])
        - np.asarray(template.reference_center_uv)
    ) > 0.1
    # A larger gate accepted a lighting-induced semantic centroid jump of
    # roughly 24 px as real motion and poisoned the online Jacobian.  Keep the
    # semantic/LK arbitration below that displacement at the calibrated tool
    # scale while still allowing small contour changes.
    assert 0.10 < profile["perception"][
        "maximum_semantic_flow_disagreement_tool_units"
    ] < 0.20
    assert profile["perception"]["hover_alignment_mode"] == (
        "identity_only_before_fresh_preclose"
    )


def test_head_seeded_hover_uses_direct_identity_not_approach_pixel_goal():
    profile = load_profile(PROFILE)
    direct = SimpleNamespace(
        marker_cross_shaped=True,
        source="blue_marker",
    )
    accepted, mode = runner._hover_alignment_policy(
        profile,
        direct,
        canonical_goal_aligned=False,
        direct_cross_verified=True,
    )
    assert accepted is True
    assert mode == "identity_only_before_fresh_preclose"

    # A locally continued fragment or another blue region cannot inherit the
    # direct-cross proof merely because low-pose correction happens later.
    fragment = SimpleNamespace(
        marker_cross_shaped=False,
        source="local_blue_evidence_continuity",
    )
    accepted, _ = runner._hover_alignment_policy(
        profile,
        fragment,
        canonical_goal_aligned=False,
        direct_cross_verified=True,
        replaying_low_pose_correction=True,
    )
    assert accepted is False
    assert profile["execution"][
        "persisted_hover_replay_final_tolerance_rad"
    ] < profile["execution"]["hover_visual_final_tolerance_rad"]
    assert profile["execution"]["level_hover_transition_final_tolerance_rad"] <= 0.008
    assert profile["execution"][
        "level_hover_transition_maximum_endpoint_correction_rad"
    ] >= 0.20
    assert profile["execution"]["camera_replay_maximum_tip_height_difference_m"] > (
        profile["perception"]["maximum_preclose_tip_height_difference_m"]
    )
    assert profile["execution"]["descent_direct_joint_final_tolerance_rad"] < (
        profile["execution"]["direct_joint_final_tolerance_rad"]
    )
    # Piper's low-pose controller can settle about 0.007 rad from the requested
    # joint endpoint.  This tolerance only decides whether to run the fresh
    # geometric preclose gates; it never authorizes closure itself.
    assert profile["execution"]["descent_direct_joint_final_tolerance_rad"] <= 0.008
    assert profile["execution"]["descent_require_final_joint_convergence"] is False
    assert profile["execution"]["hover_probe_require_final_joint_convergence"] is False
    assert profile["execution"][
        "hover_probe_converge_to_audited_joint_endpoint"
    ] is False
    assert profile["execution"]["preclose_same_position_level_correction"] is True
    assert profile["execution"]["preclose_level_correction_final_tolerance_rad"] <= 0.002
    assert profile["execution"]["preclose_level_correction_integral_endpoint_correction"] is True
    assert 0 < profile["execution"]["preclose_maximum_commanded_tip_height_bias_m"] <= 0.0017
    assert profile["execution"]["preclose_maximum_same_position_level_corrections"] == 4
    assert profile["execution"]["descent_direct_joint_endpoint_correction_gain"] > 0
    assert profile["execution"]["descent_direct_joint_integral_endpoint_correction"] is True
    assert profile["execution"][
        "descent_direct_joint_maximum_endpoint_correction_rad"
    ] <= 0.05
    assert profile["trajectory"]["verification_lift_yaw_deg"] == 0.0
    assert profile["trajectory"]["initial_hold_check_lift_m"] <= 0.002
    assert 0 < profile["trajectory"]["initial_hold_check_lift_duration_s"] <= 0.5
    assert profile["closure"]["holding_preload_ratio_delta"] > profile["closure"]["minimum_preloaded_obstruction_gap_ratio"] > 0
    assert profile["perception"]["maximum_preclose_tip_height_difference_m"] <= 0.0007
    assert profile["perception"]["maximum_preclose_height_above_verified_m"] <= 0.001
    assert profile["perception"]["maximum_preclose_height_below_verified_m"] <= 0.001
    assert profile["perception"]["maximum_preclose_center_error_scale"] < (
        profile["perception"]["maximum_hover_center_error_scale"]
    )
    assert profile["perception"]["maximum_hover_center_error_scale"] <= 0.15
    assert profile["perception"]["maximum_preclose_center_error_scale"] <= 0.10
    assert profile["perception"]["minimum_preclose_target_inside_fraction"] >= 0.98
    assert 0.28 < profile["perception"][
        "maximum_preclose_parallax_reacquisition_diagonal_fraction"
    ] <= 0.35
    assert profile["perception"]["require_runtime_hover_axis_calibration"] is True


def test_preclose_visual_gate_uses_normalized_containment_not_mask_shape():
    profile = load_profile(PROFILE)
    observation = SimpleNamespace(
        tool_frame_source="rigid_expected_tool_frame",
        center_uv=profile["target_identity"]["canonical_preclose_goal_uv"],
        grasp_window=SimpleNamespace(
            target_inside_fraction=1.0,
            normalized_center_error=0.087,
            normalized_quantile_error=0.61,
        ),
    )
    assert runner._preclose_visual_alignment_allowed(profile, observation) is True
    observation.center_uv = np.asarray(observation.center_uv) + [0.2, 0.0]
    assert runner._preclose_visual_alignment_allowed(profile, observation) is False
    observation.center_uv = profile["target_identity"]["canonical_preclose_goal_uv"]
    observation.grasp_window.target_inside_fraction = 0.90
    assert runner._preclose_visual_alignment_allowed(profile, observation) is False


def test_preclose_height_gate_rejects_excessive_support_penetration():
    profile = load_profile(PROFILE)
    verified = np.asarray(
        profile["trajectory"]["verified_support_contact_pose_wxyz_xyz"],
        dtype=float,
    )
    maximum_below = float(
        profile["perception"]["maximum_preclose_height_below_verified_m"]
    )
    accepted_pose = verified.copy()
    accepted_pose[6] -= 0.5 * maximum_below
    rejected_pose = verified.copy()
    rejected_pose[6] -= 1.1 * maximum_below
    accepted = runner._preclose_height_report(
        profile,
        low_pose=accepted_pose,
        verified_preclose_pose=verified,
        support_up=[0.0, 0.0, 1.0],
    )
    rejected = runner._preclose_height_report(
        profile,
        low_pose=rejected_pose,
        verified_preclose_pose=verified,
        support_up=[0.0, 0.0, 1.0],
    )
    assert accepted["accepted"] is True
    assert rejected["accepted"] is False


def test_contact_settle_is_deferred_only_for_misaligned_exploration():
    profile = load_profile(PROFILE)
    assert runner._defer_contact_settle_until_visual_alignment(
        profile, visually_aligned=False
    ) is True
    assert runner._defer_contact_settle_until_visual_alignment(
        profile, visually_aligned=True
    ) is False


def test_marker_goal_error_is_distinct_from_transparent_mask_center_error():
    profile = load_profile(PROFILE)
    observation = SimpleNamespace(
        center_uv=np.asarray([0.7821257616739911, -0.6406254572524702])
    )
    marker_error = runner._preclose_marker_center_error_scale(
        profile, observation
    )
    assert 0.15 < marker_error < 0.17


def test_preclose_tip_height_bias_uses_measured_secant_zero_crossing():
    first, first_method = runner._next_preclose_tip_height_bias_m(
        current_command_m=0.0,
        measured_error_m=-0.00134,
        maximum_bias_m=0.0017,
        fallback_gain=1.0,
    )
    assert first_method == "bounded_residual"
    assert abs(first - 0.00134) < 1e-12
    second, second_method = runner._next_preclose_tip_height_bias_m(
        current_command_m=first,
        measured_error_m=0.002397,
        previous_command_m=0.0,
        previous_error_m=-0.00134,
        maximum_bias_m=0.0017,
        fallback_gain=1.0,
    )
    assert second_method == "bounded_secant_zero_crossing"
    assert 0.00045 < second < 0.00052


def test_same_position_level_execution_keeps_measured_xyz(monkeypatch):
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    seed_q = np.asarray(
        profile["trajectory"]["verified_preclose_q_physical_rad"]
    )
    _, level_audit = runner._plan_same_position_level_joint_samples(
        profile,
        fk,
        start_q=seed_q,
        target_signed_tip_height_difference_m=0.0,
    )
    planned_q = np.asarray(level_audit["planned_level_q_physical_rad"])
    measured_pose = np.asarray(fk.pose(planned_q).parameters(), dtype=float)
    measured_pose[4:7] += np.asarray([0.011, -0.007, 0.003])

    class Pose:
        def parameters(self):
            return measured_pose.copy()

    rpc = SimpleNamespace(
        get_right_ee_pose=lambda: Pose(),
        get_right_joint_positions=lambda: planned_q.copy(),
    )
    captured = {}

    def cartesian(_profile, _rpc, _fk, **kwargs):
        captured.update(kwargs)
        return {"commands_sent": True}

    monkeypatch.setattr(runner, "_cartesian_move", cartesian)
    report = runner._execute_same_position_level_cartesian(
        profile,
        rpc,
        fk,
        planned_level_q=planned_q,
        aperture=1.0,
        stage="test_level",
    )
    np.testing.assert_allclose(captured["target_pose"][4:7], measured_pose[4:7])
    np.testing.assert_allclose(
        captured["target_pose"][:4], fk.pose(planned_q).parameters()[:4]
    )
    assert report["method"] == (
        "teleop_cartesian_measured_xyz_plus_wrist_roll_level"
    )
    assert report["measured_position_drift_m"] == 0.0


def test_same_position_level_endpoint_corrects_only_wrist_roll(monkeypatch):
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    seed_q = np.asarray(
        profile["trajectory"]["verified_preclose_q_physical_rad"]
    )
    _, level_audit = runner._plan_same_position_level_joint_samples(
        profile,
        fk,
        start_q=seed_q,
        target_signed_tip_height_difference_m=0.0,
    )
    planned_q = np.asarray(level_audit["planned_level_q_physical_rad"])
    state = {"q": planned_q.copy()}
    state["q"][5] += 0.005

    class Pose:
        def parameters(self):
            return np.asarray(fk.pose(state["q"]).parameters(), dtype=float)

    rpc = SimpleNamespace(
        get_right_ee_pose=lambda: Pose(),
        get_right_joint_positions=lambda: state["q"].copy(),
    )
    monkeypatch.setattr(
        runner,
        "_cartesian_move",
        lambda *_args, **_kwargs: {"commands_sent": True},
    )
    monkeypatch.setattr(
        runner,
        "_right_joint_path_contact_audit",
        lambda *_args, **_kwargs: {"accepted": True},
    )
    captured = {}

    def execute(_profile, _rpc, _fk, samples, **_kwargs):
        final_q = np.asarray(samples[-1].right_q_physical_rad, dtype=float)
        captured["final_q"] = final_q.copy()
        captured["execute_kwargs"] = dict(_kwargs)
        state["q"] = final_q
        return {"commands_sent": True}

    monkeypatch.setattr(runner, "_execute_direct_joint_samples", execute)
    report = runner._execute_same_position_level_cartesian(
        profile,
        rpc,
        fk,
        planned_level_q=planned_q,
        aperture=1.0,
        stage="test_wrist_roll_level",
    )
    np.testing.assert_allclose(captured["final_q"][:5], planned_q[:5])
    assert captured["final_q"][5] == planned_q[5]
    assert abs(
        report["wrist_roll_endpoint"]["commanded_delta_rad"] + 0.005
    ) < 1e-12
    assert report["wrist_roll_endpoint"]["planned_position_shift_m"] < 1e-12
    assert captured["execute_kwargs"]["endpoint_correction_gain"] == 1.0
    assert captured["execute_kwargs"][
        "maximum_endpoint_correction_rad"
    ] == 0.02
    assert captured["execute_kwargs"]["accumulate_endpoint_correction"] is True


def test_wrist_roll_endpoint_search_stays_in_measured_branch():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    planned_q = np.asarray(
        profile["trajectory"]["verified_preclose_q_physical_rad"],
        dtype=float,
    )
    measured_q = planned_q.copy()
    measured_q[:5] += np.asarray([0.004, -0.003, 0.002, 0.003, -0.002])
    measured_q[5] += 0.012
    endpoint_q, endpoint_level, report = (
        runner._select_wrist_roll_level_endpoint(
            profile,
            fk,
            measured_q=measured_q,
            planned_level_q=planned_q,
        )
    )
    np.testing.assert_allclose(endpoint_q[:5], measured_q[:5])
    assert abs(endpoint_q[5] - measured_q[5]) <= 0.03 + 1e-12
    assert report["method"] == "bounded_measured_branch_j6_search"
    assert report["sample_count"] >= 121
    measured_reference = replace(
        runner._load_level_reference(profile["level_config"]),
        maximum_tip_height_difference_m=float(
            profile["perception"]["maximum_preclose_tip_height_difference_m"]
        ),
    )
    before = runner.assess_jaw_level(
        np.asarray(fk.pose(measured_q).parameters(), dtype=float),
        measured_reference,
        planned=True,
    )
    assert abs(endpoint_level.tip_height_difference_m) <= abs(
        before.tip_height_difference_m
    )


def test_canonical_hover_preserves_camera_orientation_on_local_translation():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    low_position = np.asarray(fk.pose(low).parameters()[4:])
    shifted_low, _ = runner._plan_fixed_orientation_pose(
        profile,
        fk,
        target_position=low_position + np.asarray([-0.01, 0.0, 0.0]),
        orientation_q=low,
        seed_q=low,
        role="test_shifted_low",
    )
    first, _ = runner._plan_level_vertical_offset(profile, fk, low, 0.02)
    shifted, report = runner._plan_level_vertical_offset(
        profile, fk, shifted_low, 0.02, seed_hover_q=first
    )
    rotation_error = np.linalg.norm(
        runner._rotation_error_vector(
            fk.pose(first).as_matrix()[:3, :3],
            fk.pose(shifted).as_matrix()[:3, :3],
        )
    )
    assert report["accepted"] is True
    assert rotation_error < 0.05
    assert np.max(np.abs(shifted - first)) < 0.65


def test_dense_descent_levels_then_moves_strictly_vertical_on_one_joint_branch():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    # The hover must inherit the low pose's complete orientation.  An
    # independently reachable hover yaw can become unreachable during the
    # final millimetres and make one fingertip lead.
    reference_low_q = np.asarray(
        [
            -0.2518974437,
            0.0001000000,
            -0.2100953359,
            1.7399000000,
            0.6221884196,
            0.7518198557,
        ]
    )
    low_position = np.asarray(fk.pose(reference_low_q).parameters()[4:])
    start_q, hover_report = runner._plan_fixed_orientation_pose(
        profile,
        fk,
        target_position=(
            low_position
            + np.asarray([0.0, 0.0, profile["trajectory"]["verification_lift_m"]])
        ),
        orientation_q=reference_low_q,
        seed_q=reference_low_q,
        role="test_low_orientation_vertical_hover",
    )
    assert hover_report["role"] == "test_low_orientation_vertical_hover"
    samples, audit = runner._plan_level_vertical_joint_descent(
        profile,
        fk,
        start_q=start_q,
        reference_low_q=reference_low_q,
    )
    assert audit["accepted"] is True
    assert audit["method"] == "dense_joint_branch_replay_vertical_descent"
    assert audit["orientation_pairing"].endswith(
        "previous_low_orientation_endpoint"
    )
    assert audit["dynamic_low_plan"]["method"] == (
        "position_plus_fixed_orientation"
    )
    assert (
        audit["maximum_planar_motion_m"]
        < profile["trajectory"]["maximum_descent_planar_motion_m"]
    )
    assert audit["maximum_combined_tilt_deg"] < 1.5
    assert audit["maximum_tip_height_difference_m"] < 0.0017
    assert audit["contact"]["accepted"] is True
    assert samples[0].stage == "level_vertical_descent_branch_replay"
    assert samples[-1].stage == "level_vertical_descent_branch_replay"
    start_xyz = np.asarray(fk.pose(start_q).parameters()[4:])
    end_xyz = np.asarray(
        fk.pose(samples[-1].right_q_physical_rad).parameters()[4:]
    )
    reference_low_xyz = np.asarray(fk.pose(reference_low_q).parameters()[4:])
    np.testing.assert_allclose(end_xyz[:2], start_xyz[:2], atol=0.0006)
    np.testing.assert_allclose(
        end_xyz[2],
        reference_low_xyz[2]
        - profile["execution"]["descent_low_height_bias_m"],
        atol=0.0005,
    )


def test_pasteur_verified_hover_preserves_the_motor_proven_low_branch():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(
        profile["trajectory"]["verified_preclose_q_physical_rad"]
    )
    hover, _ = runner._plan_level_vertical_offset(
        profile,
        fk,
        low,
        profile["trajectory"]["verification_lift_m"],
    )
    samples, audit = runner._plan_level_vertical_joint_descent(
        profile, fk, start_q=hover, reference_low_q=low
    )
    assert audit["accepted"] is True
    assert audit["fresh_cartesian_ik_used"] is False
    assert audit["dynamic_low_plan"]["method"] == (
        "position_plus_fixed_orientation"
    )
    assert audit["maximum_planar_motion_m"] < 0.0006
    assert audit["maximum_tip_height_difference_m"] < 0.0017
    start_xyz = np.asarray(fk.pose(hover).parameters()[4:])
    end_xyz = np.asarray(
        fk.pose(samples[-1].right_q_physical_rad).parameters()[4:]
    )
    low_xyz = np.asarray(fk.pose(low).parameters()[4:])
    np.testing.assert_allclose(end_xyz[:2], start_xyz[:2], atol=0.0006)
    np.testing.assert_allclose(end_xyz[2], low_xyz[2], atol=0.0005)


def test_runtime_xy_probe_can_descend_for_a_fresh_low_visual_check():
    """A safe probe must reach the camera checkpoint before it can be scored."""

    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    measured_hover = np.asarray(
        [
            -0.16051793098449707,
            0.37948694825172424,
            -0.22588051855564117,
            -0.5603204965591431,
            0.28529152274131775,
            2.900300979614258,
        ]
    )
    probed_low = np.asarray(
        [
            -0.2968869249225818,
            0.3319394498101684,
            -0.11004772235793839,
            1.3158898103347,
            0.13277949255600102,
            1.0065037120770253,
        ]
    )

    samples, audit = runner._plan_level_vertical_joint_descent(
        profile,
        fk,
        start_q=measured_hover,
        reference_low_q=probed_low,
    )

    assert audit["accepted"] is True
    assert audit["samplewise_cartesian_audit"]["accepted"] is True
    assert audit["samplewise_cartesian_audit"]["orientation_mode"] == (
        "level_fixed_yaw"
    )
    assert audit["maximum_tip_height_difference_m"] < 0.0007
    assert audit["contact"]["accepted"] is True
    assert samples[-1].right_gripper_open_ratio == 1.0


def test_samplewise_descent_uses_fk_audit_when_solver_hits_nfev(monkeypatch):
    """A feasible Cartesian path must not depend on scipy's status bit."""

    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    hover = np.asarray(
        [
            -0.16051793098449707,
            0.37948694825172424,
            -0.22588051855564117,
            -0.5603204965591431,
            0.28529152274131775,
            2.900300979614258,
        ]
    )
    low = np.asarray(
        [
            -0.2968869249225818,
            0.3319394498101684,
            -0.11004772235793839,
            1.3158898103347,
            0.13277949255600102,
            1.0065037120770253,
        ]
    )
    original_least_squares = runner.least_squares
    forced_nonconvergence = {"done": False}

    def report_nfev_after_feasible_solution(fun, *args, **kwargs):
        result = original_least_squares(fun, *args, **kwargs)
        if (
            not forced_nonconvergence["done"]
            and "_plan_level_vertical_joint_descent.<locals>.residual"
            in getattr(fun, "__qualname__", "")
        ):
            result.success = False
            result.message = "The maximum number of function evaluations is exceeded."
            forced_nonconvergence["done"] = True
        return result

    monkeypatch.setattr(runner, "least_squares", report_nfev_after_feasible_solution)
    samples, audit = runner._plan_level_vertical_joint_descent(
        profile,
        fk,
        start_q=hover,
        reference_low_q=low,
    )

    assert forced_nonconvergence["done"] is True
    assert samples
    assert audit["accepted"] is True
    samplewise = audit["samplewise_cartesian_audit"]
    assert samplewise["accepted"] is True
    assert samplewise["solver_nonconverged_sample_count"] == 1
    assert samplewise["solver_nonconverged_samples"][0]["sample_index"] >= 1


def test_runtime_y_probe_levels_during_descent_without_losing_xy():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    measured_hover = np.asarray(
        [
            -0.13987068831920624,
            0.43823471665382385,
            -0.24005258083343506,
            -0.552291989326477,
            0.17985618114471436,
            2.896810293197632,
        ]
    )
    probed_low = np.asarray(
        [
            0.023684594976052572,
            0.41942315150566467,
            -0.11787882243144097,
            0.7309017705579418,
            -0.1660150514437894,
            1.6381112958736859,
        ]
    )

    samples, audit = runner._plan_level_vertical_joint_descent(
        profile,
        fk,
        start_q=measured_hover,
        reference_low_q=probed_low,
    )

    assert audit["accepted"] is True
    assert audit["prelevel"]["skipped"] is True
    assert audit["prelevel"]["reason"] == (
        "level_constrained_descent_levels_before_lowering"
    )
    assert audit["maximum_planar_motion_m"] < 0.0006
    assert audit["contact"]["accepted"] is True
    assert samples[-1].right_gripper_open_ratio == 1.0


def test_measured_tip_mismatch_levels_before_motor_proven_descent():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    measured_hover = np.asarray(
        [
            -0.31379273533821106,
            0.1687558889389038,
            -0.26569145917892456,
            -0.030281461775302887,
            0.4720941185951233,
            2.3244469165802,
        ]
    )
    prior_low = np.asarray(
        [
            -0.1020523433395839,
            0.06308074229378897,
            -0.13872033473434384,
            0.06029171584627764,
            0.25259840295939373,
            2.2924595125158582,
        ]
    )
    samples, audit = runner._plan_level_vertical_joint_descent(
        profile,
        fk,
        start_q=measured_hover,
        reference_low_q=prior_low,
    )
    assert audit["accepted"] is True
    assert audit["level_sample_count"] > 0
    assert samples[0].stage == "level_jaws_above_target"
    assert audit["prelevel"].get("skipped") is not True
    assert audit["maximum_joint_step_rad"] < 0.01
    assert audit["maximum_tip_height_difference_m"] < 0.001


def test_same_position_leveling_removes_visual_correction_tilt_in_free_space():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    tilted_hover_q = np.asarray(
        [
            0.0460243337,
            0.1126959100,
            -0.2331585288,
            -0.3642153144,
            0.2850995362,
            2.6874580383,
        ]
    )
    samples, audit = runner._plan_same_position_level_joint_samples(
        profile, fk, start_q=tilted_hover_q
    )
    assert audit["accepted"] is True
    assert audit["start_level"]["accepted"] is False
    assert audit["final_level"]["accepted"] is True
    assert audit["maximum_planar_motion_m"] < 1e-6
    assert audit["maximum_height_motion_m"] < 1e-6
    assert audit["contact"]["accepted"] is True
    assert samples[-1].stage == "level_jaws_above_target"


def test_verification_lift_is_fixed_orientation_and_strictly_vertical():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    level_low_q = np.asarray(
        [
            -0.3471458447,
            0.0677485216,
            -0.1836704456,
            0.8688075525,
            0.3906514672,
            1.5082207104,
        ]
    )
    samples, audit = runner._plan_straight_level_joint_lift(
        profile,
        fk,
        start_q=level_low_q,
        aperture=0.0,
    )
    assert audit["accepted"] is True
    assert audit["planned_rotation_change_rad"] == 0.0
    assert audit["maximum_planar_motion_m"] < 1e-6
    assert audit["maximum_combined_tilt_deg"] < 0.01
    assert audit["contact"]["accepted"] is True
    assert samples[-1].stage == "verification_lift_straight_level"
    start_pose = np.asarray(fk.pose(level_low_q).parameters())
    end_pose = np.asarray(fk.pose(samples[-1].right_q_physical_rad).parameters())
    np.testing.assert_allclose(end_pose[:4], start_pose[:4], atol=1e-6)
    np.testing.assert_allclose(end_pose[4:6], start_pose[4:6], atol=1e-6)
    np.testing.assert_allclose(
        end_pose[6],
        start_pose[6] + profile["trajectory"]["verification_lift_m"],
        atol=1e-6,
    )

    descent, descent_audit = runner._plan_straight_level_joint_lift(
        profile,
        fk,
        start_q=level_low_q,
        aperture=1.0,
        distance_m=-0.001,
        duration_s=0.8,
        stage="preclose_height_settle_test",
    )
    assert descent_audit["accepted"] is True
    assert descent_audit["achieved_progress_m"] > 0.00099
    descent_pose = np.asarray(
        fk.pose(descent[-1].right_q_physical_rad).parameters()
    )
    np.testing.assert_allclose(descent_pose[4:6], start_pose[4:6], atol=1e-6)
    np.testing.assert_allclose(descent_pose[6], start_pose[6] - 0.001, atol=1e-6)


def test_preclose_height_settle_executes_audited_direct_joint_path(monkeypatch):
    profile = load_profile(PROFILE)
    assert profile["execution"]["preclose_height_settle_control"] == "direct_joint"
    fk = runner.ProductionRightFK(profile["production_model"])
    start_q = np.asarray(
        profile["trajectory"]["verified_preclose_q_physical_rad"],
        dtype=float,
    )
    rpc = SimpleNamespace(get_right_joint_positions=lambda: start_q.copy())
    captured = {}

    def execute(_profile, _rpc, _fk, samples, **kwargs):
        captured["samples"] = samples
        captured["kwargs"] = kwargs
        return {"commands_sent": True, "command_path": "set_right_joint_target"}

    monkeypatch.setattr(runner, "_execute_direct_joint_samples", execute)
    monkeypatch.setattr(
        runner,
        "_cartesian_move",
        lambda *_args, **_kwargs: pytest.fail("Cartesian fallback was used"),
    )
    report = runner._execute_preclose_vertical_height_settle(
        profile,
        rpc,
        fk,
        requested_down_m=0.001,
        support_up=[0.0, 0.0, 1.0],
        settle_index=1,
    )
    assert report["height_settle_control"] == "direct_joint"
    assert report["offline_vertical_audit"]["accepted"] is True
    assert captured["samples"][-1].stage == "preclose_vertical_height_settle_01"
    assert captured["kwargs"]["accumulate_endpoint_correction"] is True
    assert captured["kwargs"]["require_final_convergence"] is True
    assert captured["kwargs"]["final_tolerance_rad"] == 0.012


def test_metric_replan_is_physical_xy_and_trust_region_bounded():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    jacobian = np.asarray(
        profile["perception"]["hover_error_jacobian_uv_per_physical_m"]
    )
    intended_delta = np.asarray([-0.005, 0.0])
    error = -jacobian @ intended_delta
    goal = np.asarray([0.2, -0.3])
    observation = SimpleNamespace(center_uv=goal + error)
    corrected, report = runner._metric_replan(
        profile,
        observation,
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state={},
    )
    before = np.asarray(fk.pose(low).parameters()[4:6])
    after = np.asarray(fk.pose(corrected).parameters()[4:6])
    assert report["method"] == "fixed_camera_physical_xy_trust_region_broyden"
    assert np.linalg.norm(after - before) <= (
        profile["perception"]["maximum_planar_correction_m"] + 0.001
    )
    np.testing.assert_allclose(after - before, intended_delta, atol=1e-3)


def test_hover_tracking_bias_is_fed_forward_into_next_low_target(monkeypatch):
    profile = load_profile(PROFILE)
    captured = {}

    class FK:
        def pose(self, _q):
            return SimpleNamespace(
                parameters=lambda: [1.0, 0.0, 0.0, 0.0, 0.08, 0.03, 0.84]
            )

    def plan(_profile, _fk, *, target_position, seed_q, role):
        captured["target_position"] = np.asarray(target_position)
        captured["role"] = role
        return np.full(6, 0.1), {"accepted": True, "role": role}

    monkeypatch.setattr(runner, "_plan_level_fixed_yaw_pose", plan)
    q, report = runner._compensate_hover_tracking_bias(
        profile,
        FK(),
        np.zeros(6),
        {"target_low_xy_m": [0.08, 0.03]},
        hover={"measured_hover_xy_m": [0.102, 0.203]},
        hover_plan={"target_position_m": [0.1, 0.2, 0.86]},
    )
    np.testing.assert_allclose(captured["target_position"][:2], [0.078, 0.027])
    assert captured["role"] == (
        "tracking_bias_compensated_fixed_yaw_low_visual_servo"
    )
    np.testing.assert_allclose(q, np.full(6, 0.1))
    assert report["hover_tracking_bias_compensation"]["accepted"] is True
    np.testing.assert_allclose(
        report["hover_tracking_bias_compensation"]["measured_bias_xy_m"],
        [0.002, 0.003],
    )


def test_metric_replan_uses_measured_xy_for_camera_feedback():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    planned_xy = np.asarray(fk.pose(low).parameters()[4:6])
    measured_xy = planned_xy + np.asarray([0.004, -0.003])
    goal = np.asarray([0.2, -0.3])
    observation = SimpleNamespace(center_uv=goal + np.asarray([0.2, -0.1]))
    _, report = runner._metric_replan(
        profile,
        observation,
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state={},
        measured_current_xy_m=measured_xy,
    )
    np.testing.assert_allclose(report["current_low_xy_m"], measured_xy)


def test_preclose_replan_preserves_validated_wrist_orientation_across_ik_branches():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    aligned_low = np.asarray(
        [
            0.21473241436040588,
            0.2381226602522587,
            -0.1000575436141818,
            0.12167405603303395,
            -0.2330830758666415,
            2.2160149695691023,
        ]
    )
    # This is the alternate joint branch reached by the fixed-quaternion
    # Cartesian descent.  Its EE pose is valid, but it must not become the
    # camera-orientation reference for the next visual attempt.
    descended_branch = np.asarray(
        [
            -0.6870313882827759,
            0.18458601832389832,
            -0.18095573782920837,
            1.445516586303711,
            0.6598042845726013,
            0.9333148002624512,
        ]
    )
    measured_xy = np.asarray(fk.pose(descended_branch).parameters()[4:6])
    goal = np.asarray([0.56843862, -0.38745984])
    observation = SimpleNamespace(center_uv=goal + np.asarray([-0.05, 0.28]))
    corrected, report = runner._metric_replan(
        profile,
        observation,
        aligned_low,
        fk=fk,
        reference_center_uv=goal,
        servo_state={},
        measured_current_xy_m=measured_xy,
        fixed_orientation_q=aligned_low,
    )
    assert report["ik"]["role"] == "fixed_orientation_low_visual_servo"
    np.testing.assert_allclose(
        report["ik"]["orientation_source_q_physical_rad"], aligned_low
    )
    rotation_error = np.linalg.norm(
        runner._rotation_error_vector(
            fk.pose(aligned_low).as_matrix()[:3, :3],
            fk.pose(corrected).as_matrix()[:3, :3],
        )
    )
    assert rotation_error < 0.03


def test_metric_replan_does_not_deadlock_after_returning_to_best_pose():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    jacobian = np.asarray(
        profile["perception"]["hover_error_jacobian_uv_per_physical_m"]
    )
    goal = np.asarray([0.2, -0.3])
    error = -jacobian @ np.asarray([-0.005, 0.0])
    state = {
        "best_error_norm": 1e-4,
        "best_low_q_physical_rad": low.tolist(),
    }
    corrected, report = runner._metric_replan(
        profile,
        SimpleNamespace(center_uv=goal + error),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=np.asarray(fk.pose(low).parameters()[4:6])
        + np.asarray([0.004, -0.003]),
    )
    assert report["regressed_from_best"] is False
    assert np.linalg.norm(report["selected_delta_xy_m"]) > 0.0
    assert not np.allclose(corrected, low)
    assert state["backtrack_anchor_refreshes"] == 1
    assert state["best_error_norm"] == 1e-4


def test_metric_replan_backtrack_obeys_cartesian_trust_radius():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    best_q = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    current_q = best_q + np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_xy = np.asarray(fk.pose(best_q).parameters()[4:6])
    current_xy = best_xy + np.asarray([0.020, -0.015])
    goal = np.asarray([0.2, -0.3])
    state = {
        "best_error_norm": 0.01,
        "best_low_q_physical_rad": best_q.tolist(),
        "best_xy_m": best_xy.tolist(),
    }
    _, report = runner._metric_replan(
        profile,
        SimpleNamespace(
            center_uv=goal + np.asarray([0.4, -0.3]),
            component_touches_border=False,
            component_area_per_tool_scale_sq=0.03,
        ),
        current_q,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=current_xy,
        level_yaw_free=True,
    )
    assert report["regressed_from_best"] is True
    assert report["method"] == "fixed_camera_physical_xy_broyden_backtrack"
    assert np.linalg.norm(report["selected_delta_xy_m"]) <= (
        profile["perception"]["maximum_planar_correction_m"] + 1e-12
    )
    np.testing.assert_allclose(
        report["target_low_xy_m"],
        current_xy + np.asarray(report["selected_delta_xy_m"]),
    )


def test_metric_replan_backtracks_when_bad_step_pushes_target_to_border():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    best_q = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    current_q = best_q + np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_xy = np.asarray(fk.pose(best_q).parameters()[4:6])
    current_xy = best_xy + np.asarray([-0.012, -0.010])
    goal = np.asarray([0.2, -0.3])
    state = {
        "best_error_norm": 0.2,
        "best_low_q_physical_rad": best_q.tolist(),
        "best_xy_m": best_xy.tolist(),
    }
    _, report = runner._metric_replan(
        profile,
        SimpleNamespace(
            center_uv=goal + np.asarray([0.5, -0.4]),
            center_px=[12.0, 220.0],
            component_touches_border=False,
            component_area_per_tool_scale_sq=0.03,
        ),
        current_q,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=current_xy,
        level_yaw_free=True,
    )
    assert report["border_search"] is True
    assert report["regressed_from_best"] is True
    assert report["method"] == "fixed_camera_physical_xy_broyden_backtrack"
    selected = np.asarray(report["selected_delta_xy_m"])
    assert selected @ (best_xy - current_xy) > 0.0


def test_metric_replan_rejects_broyden_update_when_component_area_changes():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    configured = np.asarray(
        profile["perception"]["hover_error_jacobian_uv_per_physical_m"]
    )
    goal = np.asarray([0.2, -0.3])
    state = {
        "jacobian": configured.tolist(),
        "last_xy_m": (
            np.asarray(fk.pose(low).parameters()[4:6]) - np.asarray([0.004, 0.0])
        ).tolist(),
        "last_error_uv": [0.3, -0.2],
        "last_component_area_per_tool_scale_sq": 0.04,
    }
    _, report = runner._metric_replan(
        profile,
        SimpleNamespace(
            center_uv=goal + np.asarray([0.15, -0.04]),
            component_touches_border=False,
            component_area_per_tool_scale_sq=0.01,
        ),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
    )
    assert report["broyden_update"]["accepted"] is False
    assert report["broyden_update"]["reason"] == (
        "semantic_component_area_changed"
    )
    np.testing.assert_allclose(report["jacobian_uv_per_physical_m"], configured)


def test_metric_replan_calibrates_cartesian_axes_before_goal_correction():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    base_xy = np.asarray(fk.pose(low).parameters()[4:6])
    goal = np.asarray([0.2, -0.3])
    base_error = np.asarray([0.12, -0.08])
    true_jacobian = np.asarray([[12.0, -5.0], [7.0, 22.0]])
    state = {"enable_runtime_axis_calibration": True}
    probe_m = profile["perception"]["runtime_hover_axis_near_goal_probe_m"]

    def observation(error):
        return SimpleNamespace(
            center_uv=goal + error,
            component_touches_border=False,
            component_area_per_tool_scale_sq=0.03,
        )

    x_q, x_report = runner._metric_replan(
        profile,
        observation(base_error),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=base_xy,
    )
    np.testing.assert_allclose(x_report["selected_delta_xy_m"], [probe_m, 0.0])
    assert x_report["runtime_axis_probe_stage"] == "runtime_axis_x_probe"

    y_q, y_report = runner._metric_replan(
        profile,
        observation(base_error + true_jacobian @ np.asarray([probe_m, 0.0])),
        x_q,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=base_xy + np.asarray([probe_m, 0.0]),
    )
    np.testing.assert_allclose(
        y_report["selected_delta_xy_m"], [0.0, probe_m]
    )
    assert y_report["runtime_axis_probe_stage"] == "runtime_axis_y_probe"

    _, correction = runner._metric_replan(
        profile,
        observation(base_error + true_jacobian @ np.asarray([probe_m, probe_m])),
        y_q,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=base_xy + np.asarray([probe_m, probe_m]),
    )
    assert correction["runtime_axis_probe_stage"] is None
    assert state["runtime_axis_calibration"]["completed"] is True
    np.testing.assert_allclose(state["jacobian"], true_jacobian, atol=1e-9)


def test_metric_replan_freezes_high_signal_axis_probe_jacobian():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    current_xy = np.asarray(fk.pose(low).parameters()[4:6])
    goal = np.asarray([0.2, -0.3])
    calibrated = np.asarray([[12.0, -5.0], [7.0, 22.0]])
    noisy_online = np.asarray([[45.0, 30.0], [-20.0, 8.0]])
    state = {
        "enable_runtime_axis_calibration": True,
        "jacobian": noisy_online.tolist(),
        "last_xy_m": (current_xy - np.asarray([0.004, 0.0])).tolist(),
        "last_error_uv": [0.01, -0.01],
        "runtime_axis_calibration": {
            "completed": True,
            "baseline_xy_m": (current_xy - np.asarray([0.010, 0.0])).tolist(),
            "jacobian_uv_per_physical_m": calibrated.tolist(),
        },
    }

    _, report = runner._metric_replan(
        profile,
        SimpleNamespace(
            center_uv=goal + np.asarray([0.12, -0.08]),
            component_touches_border=False,
            component_area_per_tool_scale_sq=0.03,
        ),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=current_xy,
    )

    np.testing.assert_allclose(report["jacobian_uv_per_physical_m"], calibrated)
    np.testing.assert_allclose(state["jacobian"], calibrated)
    assert report["broyden_update"] == {
        "accepted": False,
        "reason": "frozen_high_signal_runtime_axis_calibration",
    }


def test_metric_replan_recalibrates_axes_after_leaving_local_camera_region():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    current_xy = np.asarray(fk.pose(low).parameters()[4:6])
    goal = np.asarray([0.2, -0.3])
    state = {
        "enable_runtime_axis_calibration": True,
        "jacobian": [[40.0, 20.0], [5.0, 15.0]],
        "last_xy_m": (current_xy - [0.002, 0.001]).tolist(),
        "last_error_uv": [0.1, -0.2],
        "runtime_axis_calibration": {
            "completed": True,
            "baseline_xy_m": (current_xy - [0.060, 0.0]).tolist(),
        },
    }
    _, report = runner._metric_replan(
        profile,
        SimpleNamespace(
            center_uv=goal + np.asarray([0.12, -0.08]),
            component_touches_border=False,
            component_area_per_tool_scale_sq=0.03,
        ),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=current_xy,
    )
    assert report["runtime_axis_probe_stage"] == "runtime_axis_x_probe"
    np.testing.assert_allclose(
        report["selected_delta_xy_m"],
        [profile["perception"]["runtime_hover_axis_near_goal_probe_m"], 0.0],
    )
    assert report["runtime_axis_calibration_invalidation"]["distance_m"] > (
        profile["perception"]["maximum_runtime_axis_calibration_radius_m"]
    )
    assert state["runtime_axis_calibration"].get("completed") is not True


def test_metric_replan_uses_local_prior_after_replayed_probes_become_collinear():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    base_xy = np.asarray(fk.pose(low).parameters()[4:6])
    goal = np.asarray([0.2, -0.3])
    base_error = np.asarray([0.02, -0.01])
    prior_jacobian = np.asarray([[12.0, -5.0], [7.0, 22.0]])
    x_delta = np.asarray([0.006, 0.0])
    # A replayed Cartesian Y probe was contaminated by planar residual from
    # height/roll settling and is almost parallel to the X probe.
    y_delta = np.asarray([0.004, 0.0001])
    current_xy = base_xy + x_delta + y_delta
    current_error = base_error + prior_jacobian @ (x_delta + y_delta)
    state = {
        "enable_runtime_axis_calibration": True,
        "jacobian": prior_jacobian.tolist(),
        "last_xy_m": (current_xy - np.asarray([0.002, 0.001])).tolist(),
        "runtime_axis_calibration": {
            "baseline_xy_m": base_xy.tolist(),
            "baseline_error_uv": base_error.tolist(),
            "x_delta_xy_m": x_delta.tolist(),
            "x_delta_error_uv": (prior_jacobian @ x_delta).tolist(),
            "x_pose_xy_m": (base_xy + x_delta).tolist(),
            "x_pose_error_uv": (
                base_error + prior_jacobian @ x_delta
            ).tolist(),
            "stage": "y_probe_commanded",
        },
    }
    _, report = runner._metric_replan(
        profile,
        SimpleNamespace(
            center_uv=goal + current_error,
            center_px=[220.0, 300.0],
            component_touches_border=False,
            component_area_per_tool_scale_sq=0.03,
        ),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=current_xy,
    )
    assert state["runtime_axis_calibration"]["completed"] is True
    assert state["runtime_axis_calibration"]["stage"] == (
        "completed_with_prior_local_jacobian"
    )
    assert report["broyden_update"]["reason"] == (
        "prior_local_jacobian_after_degenerate_runtime_probes"
    )
    np.testing.assert_allclose(report["jacobian_uv_per_physical_m"], prior_jacobian)


def test_fixed_yaw_visual_replan_line_searches_before_changing_camera_branch(
    monkeypatch,
):
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    goal = np.asarray([0.2, -0.3])
    calls = []

    monkeypatch.setattr(
        runner,
        "_plan_fixed_orientation_pose",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fixed")),
    )

    def fixed_yaw(_profile, _fk, *, target_position, seed_q, role):
        calls.append(np.asarray(target_position, dtype=float))
        if len(calls) == 1:
            raise RuntimeError("full step unreachable")
        return np.asarray(seed_q, dtype=float), {"accepted": True, "role": role}

    monkeypatch.setattr(runner, "_plan_level_fixed_yaw_pose", fixed_yaw)
    _, report = runner._metric_replan(
        profile,
        SimpleNamespace(center_uv=goal + np.asarray([0.2, -0.1])),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state={},
    )
    assert report["ik"]["trust_region_line_search"]["scale"] == 0.5
    np.testing.assert_allclose(
        report["selected_delta_xy_m"],
        np.asarray(report["raw_delta_xy_m"])
        * report["trust_scale"]
        * 0.5,
    )


def test_border_axis_probe_really_reverses_after_wrong_first_direction():
    profile = load_profile(PROFILE)
    profile["perception"]["runtime_border_nonprogress_patience"] = 0
    fk = runner.ProductionRightFK(profile["production_model"])
    low = np.asarray(profile["trajectory"]["verified_preclose_q_physical_rad"])
    base_xy = np.asarray(fk.pose(low).parameters()[4:6])
    goal = np.asarray([0.2, -0.3])
    state = {"enable_runtime_axis_calibration": True}

    def clipped(center_x):
        return SimpleNamespace(
            center_uv=goal + np.asarray([0.3, -0.4]),
            center_px=[center_x, 210.0],
            component_touches_border=True,
            component_area_per_tool_scale_sq=0.03,
        )

    _, first = runner._metric_replan(
        profile,
        clipped(14.0),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=base_xy,
    )
    assert first["selected_delta_xy_m"][0] > 0.0
    _, reverse = runner._metric_replan(
        profile,
        clipped(13.0),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=base_xy + np.asarray([0.008, 0.0]),
    )
    assert reverse["runtime_axis_probe_stage"] == "runtime_axis_x_probe_reverse"
    assert reverse["selected_delta_xy_m"][0] < 0.0
    assert abs(reverse["selected_delta_xy_m"][1]) < 1e-12

    _, y_probe = runner._metric_replan(
        profile,
        clipped(12.5),
        low,
        fk=fk,
        reference_center_uv=goal,
        servo_state=state,
        measured_current_xy_m=base_xy - np.asarray([0.004, 0.0]),
    )
    assert y_probe["runtime_axis_probe_stage"] == "runtime_axis_y_border_probe"
    assert abs(y_probe["selected_delta_xy_m"][0]) < 1e-12
    assert y_probe["selected_delta_xy_m"][1] > 0.0


def test_optional_hover_joint_endpoint_convergence_is_collision_audited(
    monkeypatch,
):
    profile = load_profile(PROFILE)
    profile["perception"]["minimum_hover_directional_progress_fraction"] = -1.0
    profile["execution"][
        "hover_probe_converge_to_audited_joint_endpoint"
    ] = True
    measured_q = np.asarray([0.1, 0.2, -0.1, 0.3, -0.2, 0.4])
    target_q = measured_q + np.asarray([0.01, -0.02, 0.01, 0.02, -0.01, 0.03])
    calls = {}

    class Pose:
        def parameters(self):
            return [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.9]

    class RPC:
        def get_right_ee_pose(self):
            return Pose()

        def get_right_joint_positions(self):
            return measured_q.copy()

    def cartesian(*_args, **kwargs):
        calls["cartesian"] = kwargs
        return {"commands_sent": True}

    def audit(_profile, q_path):
        calls["audited_path"] = np.asarray(q_path)
        return {"accepted": True, "non_support_contacts": []}

    def execute(_profile, _rpc, _fk, samples, **kwargs):
        calls["samples"] = list(samples)
        calls["execution"] = kwargs
        return {"final_settle_joint_error_rad": 0.004}

    monkeypatch.setattr(runner, "_cartesian_move", cartesian)
    monkeypatch.setattr(runner, "_right_joint_path_contact_audit", audit)
    monkeypatch.setattr(runner, "_execute_direct_joint_samples", execute)
    report = runner._move_between_hovers(
        profile,
        RPC(),
        object(),
        target_q,
        selected_delta_xy_m=[0.006, 0.0],
        fixed_orientation_wxyz=[0.0, 1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        calls["cartesian"]["target_pose"][:4], [0.0, 1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(calls["audited_path"][0], measured_q)
    np.testing.assert_allclose(calls["audited_path"][-1], target_q)
    np.testing.assert_allclose(
        calls["samples"][-1].right_q_physical_rad, target_q
    )
    assert calls["execution"]["final_tolerance_rad"] == 0.008
    assert calls["execution"]["accumulate_endpoint_correction"] is True
    assert report["orientation_endpoint_convergence"]["collision_audit"][
        "accepted"
    ] is True


def test_hover_cartesian_stall_uses_collision_audited_branch_escape(monkeypatch):
    profile = load_profile(PROFILE)
    measured_q = np.asarray([0.1, 0.2, -0.1, 0.3, -0.2, 0.4])
    target_q = measured_q + np.asarray([0.02, 0.01, 0.0, -0.01, 0.02, 0.01])
    calls = {}

    class Pose:
        def parameters(self):
            return [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.9]

    class RPC:
        def get_right_ee_pose(self):
            return Pose()

        def get_right_joint_positions(self):
            return measured_q.copy()

    monkeypatch.setattr(
        runner,
        "_cartesian_move",
        lambda *_args, **_kwargs: {"commands_sent": True},
    )

    def audit(_profile, path):
        calls["path"] = np.asarray(path)
        return {"accepted": True, "non_support_contacts": []}

    def execute(_profile, _rpc, _fk, samples, **kwargs):
        calls["samples"] = list(samples)
        calls["kwargs"] = kwargs
        return {"final_settle_joint_error_rad": 0.02}

    monkeypatch.setattr(runner, "_right_joint_path_contact_audit", audit)
    monkeypatch.setattr(runner, "_execute_direct_joint_samples", execute)
    report = runner._move_between_hovers(
        profile,
        RPC(),
        object(),
        target_q,
        selected_delta_xy_m=[0.004, 0.0],
        fixed_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
    )
    assert report["method"] == (
        "audited_joint_branch_escape_after_cartesian_ik_limit"
    )
    assert "stalled at workspace boundary" in report["cartesian_error"]
    np.testing.assert_allclose(calls["path"][0], measured_q)
    np.testing.assert_allclose(calls["path"][-1], target_q)
    assert report["fresh_hover_observation_required"] is True
    assert report["closure_authorized"] is False


def test_identity_locked_hover_stall_holds_camera_branch(monkeypatch):
    profile = load_profile(PROFILE)
    measured_q = np.asarray([0.1, 0.2, -0.1, 0.3, -0.2, 0.4])

    class Pose:
        def parameters(self):
            return [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.9]

    class RPC:
        def get_right_ee_pose(self):
            return Pose()

        def get_right_joint_positions(self):
            return measured_q.copy()

    monkeypatch.setattr(
        runner,
        "_cartesian_move",
        lambda *_args, **_kwargs: {"commands_sent": True},
    )
    monkeypatch.setattr(
        runner,
        "_execute_direct_joint_samples",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity-locked probing must not change joint branch")
        ),
    )
    report = runner._move_between_hovers(
        profile,
        RPC(),
        object(),
        measured_q + 0.01,
        selected_delta_xy_m=[0.004, 0.0],
        fixed_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        allow_branch_escape=False,
    )
    assert report["method"] == "fixed_camera_workspace_boundary_hold"
    assert report["camera_branch_preserved"] is True
    assert report["fresh_hover_observation_required"] is True
    assert report["closure_authorized"] is False


def test_optical_flow_bridges_one_semantic_dropout_without_authorizing_close():
    profile = load_profile(PROFILE)
    template = GraspWindowTemplate.from_dict(
        json.loads(Path(profile["target_identity"]["grasp_window_selection"]).read_text())[
            "template"
        ]
    )
    run_dir = ROOT / "data/runs/pasteur/codexless_rgbd_reprojected_hover_20260805/attempt_01"
    previous_image = cv2.imread(str(run_dir / "hover_04.png"))
    current_image = cv2.imread(str(run_dir / "hover_05.png"))
    if previous_image is None or current_image is None:
        return
    previous = observe_marked_target(
        previous_image,
        template,
        reference_component_area_per_tool_scale_sq=profile["target_identity"][
            "right_reference_component_area_per_tool_scale_sq"
        ],
        minimum_reference_area_fraction=profile["target_identity"][
            "minimum_right_reference_area_fraction"
        ],
        maximum_reference_area_fraction=profile["target_identity"][
            "maximum_right_reference_area_fraction"
        ],
    )
    tracked = track_target_center_lk(
        previous_image, current_image, previous, template
    )
    assert np.linalg.norm(np.asarray(tracked.center_px) - previous.center_px) < 100
    assert tracked.tracking_inlier_fraction >= 0.45
    assert tracked.source == "pyramidal_lk_semantic_dropout_bridge"
    assert tracked.grasp_window.allowed_to_close is False


def test_runtime_alignment_rejected_when_tapped_target_moves(tmp_path):
    profile = load_profile(PROFILE)
    profile["target_identity"] = dict(profile["target_identity"])
    calibration = tmp_path / "runtime.json"
    profile["target_identity"]["runtime_alignment_calibration"] = str(calibration)
    calibration.write_text(
        json.dumps(
            {
                "schema": "piper_robot.thin_object_runtime_alignment/v1",
                    "canonical_hover_goal_uv": profile["target_identity"][
                        "canonical_hover_goal_uv"
                    ],
                    "canonical_preclose_goal_uv": profile["target_identity"][
                        "canonical_preclose_goal_uv"
                    ],
                "target_center_scene_xyz_m": [0.0, 0.0, 0.0],
                "low_q_physical_rad": profile["trajectory"][
                    "verified_preclose_q_physical_rad"
                ],
            }
        )
    )
    q, report = _load_runtime_alignment(
        profile, {"target_center_scene_xyz_m": [0.02, 0.0, 0.0]}
    )
    assert q is None
    assert report["accepted"] is False
    assert "moved" in report["reason"]


def test_runtime_alignment_restores_only_safe_local_servo_seed(tmp_path):
    profile = load_profile(PROFILE)
    profile["target_identity"] = dict(profile["target_identity"])
    calibration = tmp_path / "runtime.json"
    profile["target_identity"]["runtime_alignment_calibration"] = str(calibration)
    low_q = profile["trajectory"]["verified_preclose_q_physical_rad"]
    calibration.write_text(
        json.dumps(
            {
                "schema": "piper_robot.thin_object_runtime_alignment/v1",
                    "canonical_hover_goal_uv": profile["target_identity"][
                        "canonical_hover_goal_uv"
                    ],
                    "canonical_preclose_goal_uv": profile["target_identity"][
                        "canonical_preclose_goal_uv"
                    ],
                "target_center_scene_xyz_m": [0.0, 0.0, 0.0],
                "low_q_physical_rad": low_q,
                "servo_seed": {
                    "jacobian": [[8.0, -10.0], [-0.3, 52.0]],
                    "best_error_norm": 0.14,
                    "best_low_q_physical_rad": [9.0] * 6,
                    "last_xy_m": [99.0, 99.0],
                    "last_error_uv": [99.0, 99.0],
                },
            }
        )
    )
    q, report = _load_runtime_alignment(
        profile, {"target_center_scene_xyz_m": [0.001, 0.0, 0.0]}
    )
    np.testing.assert_allclose(q, low_q)
    seed = report["servo_state_seed"]
    assert set(seed) == {"jacobian", "adaptive_trust_from_first_observation"}
    assert seed["adaptive_trust_from_first_observation"] is True
    np.testing.assert_allclose(report["aligned_low_q_physical_rad"], low_q)


def test_runtime_alignment_rejects_wrong_dish_even_when_head_marker_matches(tmp_path):
    profile = copy.deepcopy(load_profile(PROFILE))
    calibration = tmp_path / "poisoned_runtime.json"
    profile["target_identity"]["runtime_alignment_calibration"] = str(calibration)
    scene = profile["head_localization"]["reference_target_center_scene_xyz_m"]
    poisoned_q = [
        -0.024248954773614596,
        0.9331019986079002,
        -0.3654699663647654,
        0.16946897874411404,
        -0.3937091556712801,
        2.2012133668264373,
    ]
    calibration.write_text(
        json.dumps(
            {
                "schema": runner.RUNTIME_ALIGNMENT_SCHEMA,
                "canonical_hover_goal_uv": profile["target_identity"][
                    "canonical_hover_goal_uv"
                ],
                "canonical_preclose_goal_uv": profile["target_identity"][
                    "canonical_preclose_goal_uv"
                ],
                "target_center_scene_xyz_m": scene,
                "low_q_physical_rad": poisoned_q,
            }
        )
    )
    q, report = _load_runtime_alignment(
        profile, {"target_center_scene_xyz_m": scene}
    )
    assert q is None
    assert report["accepted"] is False
    assert report["cross_modal_geometry"]["residual_m"] > 0.10
    assert "geometry" in report["reason"]


def test_runtime_fast_replay_requires_tight_direct_head_identity():
    profile = load_profile(PROFILE)
    low_q = profile["trajectory"]["verified_preclose_q_physical_rad"]
    runtime = {
        "accepted": True,
        "target_scene_displacement_m": 0.001,
        # Stay inside the production profile's strict pre-descent hover gate.
        # Values such as 0.12 used to pass the older 0.18 gate and must no
        # longer authorize a replay descent.
        "normalized_hover_error": 0.08,
        "aligned_low_q_physical_rad": low_q,
    }
    direct = {"runtime_head_continuity": {"proxy_used": False}}
    accepted = _runtime_fast_replay_alignment(profile, runtime, direct)
    assert accepted["accepted"] is True
    assert accepted["closure_authorized"] is False
    assert _runtime_fast_replay_alignment(
        profile,
        {**runtime, "target_scene_displacement_m": 0.004},
        direct,
    ) is None
    strong_fixed_head = {
        "runtime_head_continuity": {
            "accepted": True,
            "proxy_used": True,
            "observed_component_displacement_diagonal_fraction": 0.001,
        }
    }
    noisy_depth = _runtime_fast_replay_alignment(
        profile,
        {**runtime, "target_scene_displacement_m": 0.004},
        strong_fixed_head,
    )
    assert noisy_depth["accepted"] is True
    assert noisy_depth["strong_fixed_head_pixel_identity"] is True
    assert noisy_depth["closure_authorized"] is False
    proxy = _runtime_fast_replay_alignment(
        profile,
        runtime,
        {
            "runtime_head_continuity": {
                "proxy_used": True,
                "observed_component_displacement_diagonal_fraction": 0.008,
            }
        },
    )
    assert proxy["accepted"] is True
    assert _runtime_fast_replay_alignment(
        profile,
        runtime,
        {
            "runtime_head_continuity": {
                "proxy_used": True,
                "observed_component_displacement_diagonal_fraction": 0.015,
            }
        },
    ) is None


def test_verified_preclose_correction_replans_fresh_hover_and_never_authorizes_close():
    profile = load_profile(PROFILE)
    corrected = profile["trajectory"]["verified_preclose_q_physical_rad"]
    attempt = {
        "hover": {
            "low_q_physical_rad": corrected,
            "hover_q_physical_rad": profile["trajectory"][
                "canonical_hover_q_physical_rad"
            ],
        },
        "preclose": {
            "allowed_to_close": False,
            "level": {"accepted": True},
            "observation": {
                "source": "blue_marker",
                "component_touches_border": False,
                "grasp_window": {"normalized_center_error": 0.24},
            },
        },
        "visual_replan": {
            "method": "fixed_camera_physical_xy_trust_region_broyden",
            "selected_delta_xy_m": [-0.0038, -0.0003],
            "target_low_xy_m": [0.074, 0.0705],
            "corrected_q_physical_rad": corrected,
            "ik": {
                "accepted": True,
                "role": "level_yaw_free_low_visual_servo",
            },
        },
    }
    replay = _preclose_correction_replay_alignment(profile, attempt, corrected)
    assert replay["accepted"] is True
    assert replay["fresh_preclose_required"] is True
    assert replay["closure_authorized"] is False
    np.testing.assert_allclose(replay["aligned_low_q_physical_rad"], corrected)
    np.testing.assert_allclose(replay["target_low_xy_m"], [0.074, 0.0705])
    assert replay["source_hover_q_physical_rad"] is None
    np.testing.assert_allclose(
        replay["source_hover_orientation_seed_q_physical_rad"],
        profile["trajectory"]["canonical_hover_q_physical_rad"],
    )
    assert replay["source_tool_frame"] is None
    assert replay["source_preclose_identity_observation"]["source"] == "blue_marker"
    attempt["visual_replan"]["ik"]["role"] = "fixed_orientation_low_visual_servo"
    assert _preclose_correction_replay_alignment(
        profile, attempt, corrected
    )["accepted"] is True
    attempt["preclose_servo_state"] = {
        "enable_runtime_axis_calibration": True,
        "runtime_axis_calibration": {"stage": "x_probe_commanded"},
        "jacobian": [[8.0, -10.0], [-0.3, 52.0]],
        "best_error_norm": 0.01,
        "best_xy_m": [0.09, 0.09],
        "last_xy_m": [0.08, 0.08],
    }
    attempt["visual_replan"]["method"] = (
        "fixed_camera_runtime_cartesian_axis_probe"
    )
    replay = _preclose_correction_replay_alignment(profile, attempt, corrected)
    assert replay["accepted"] is True
    assert replay["method"] == "semantically_verified_preclose_axis_probe"
    assert replay["preclose_servo_state"]["runtime_axis_calibration"] == {
        "stage": "x_probe_commanded"
    }
    assert replay["preclose_servo_state"]["best_error_norm"] == 0.01
    sanitized = runner._fresh_preclose_retry_servo_seed(
        replay["preclose_servo_state"]
    )
    assert set(sanitized) == {
        "jacobian",
        "adaptive_trust_from_first_observation",
        "enable_runtime_axis_calibration",
    }
    np.testing.assert_allclose(
        sanitized["jacobian"], [[8.0, -10.0], [-0.3, 52.0]]
    )
    assert "best_error_norm" not in sanitized
    assert "best_xy_m" not in sanitized
    continued_probe = runner._fresh_preclose_retry_servo_seed(
        replay["preclose_servo_state"],
        preserve_runtime_axis_calibration=True,
    )
    assert continued_probe["runtime_axis_calibration"] == {
        "stage": "x_probe_commanded"
    }
    assert "best_xy_m" not in continued_probe
    completed_seed = runner._fresh_preclose_retry_servo_seed(
        {
            "enable_runtime_axis_calibration": True,
            "jacobian": [[99.0, 0.0], [0.0, 99.0]],
            "runtime_axis_calibration": {
                "completed": True,
                "stage": "completed",
                "baseline_xy_m": [0.10, 0.05],
                "jacobian_uv_per_physical_m": [
                    [-2.0, -7.0],
                    [9.0, 7.0],
                ],
                "jacobian_condition": 3.2,
                "motion_condition": 1.1,
            },
            "best_xy_m": [0.11, 0.06],
        }
    )
    np.testing.assert_allclose(
        completed_seed["jacobian"], [[-2.0, -7.0], [9.0, 7.0]]
    )
    assert completed_seed["runtime_axis_calibration"]["completed"] is True
    assert "best_xy_m" not in completed_seed
    underexcited = runner._fresh_preclose_retry_servo_seed(
        {
            "enable_runtime_axis_calibration": True,
            "jacobian": [[5.0, 7.0], [-0.4, 1.6]],
            "runtime_axis_calibration": {
                "completed": True,
                "baseline_xy_m": [0.10, 0.05],
                "x_delta_xy_m": [0.003, 0.0],
                "y_delta_xy_m": [0.0, 0.003],
                "jacobian_uv_per_physical_m": [[5.0, 7.0], [-0.4, 1.6]],
            },
        },
        preserve_runtime_axis_calibration=True,
        minimum_completed_axis_probe_span_m=0.0045,
    )
    assert underexcited == {"enable_runtime_axis_calibration": True}
    attempt["preclose"]["observation"]["source"] = (
        "local_blue_evidence_continuity"
    )
    assert _preclose_correction_replay_alignment(
        profile, attempt, corrected
    )["accepted"] is True
    attempt["preclose"]["observation"]["source"] = "pyramidal_lk_semantic_dropout_bridge"
    assert _preclose_correction_replay_alignment(profile, attempt, corrected) is None


def test_pending_low_correction_skips_only_high_view_and_still_requires_fresh_low():
    profile = load_profile(PROFILE)
    alignment = {
        "accepted": True,
        "method": "semantically_verified_preclose_level_q_correction",
        "fresh_preclose_required": True,
        "closure_authorized": False,
    }
    assert runner._may_skip_high_view_for_fresh_low_replay(profile, alignment)
    assert runner._may_skip_high_view_for_fresh_low_replay(
        profile,
        {
            **alignment,
            "method": "semantically_verified_same_xy_level_retry",
        },
    )

    # A stable/promoted replay or anything that claims closure authority must
    # still take the ordinary fresh high-view route.
    assert not runner._may_skip_high_view_for_fresh_low_replay(
        profile, {**alignment, "fresh_preclose_required": False}
    )
    assert not runner._may_skip_high_view_for_fresh_low_replay(
        profile, {**alignment, "closure_authorized": True}
    )
    assert not runner._may_skip_high_view_for_fresh_low_replay(
        profile, {**alignment, "method": "unknown_cached_pose"}
    )


def test_moderately_tilted_preclose_can_seed_but_never_authorize_next_close(
    monkeypatch,
):
    profile = load_profile(PROFILE)
    corrected = profile["trajectory"]["verified_preclose_q_physical_rad"]
    attempt = {
        "hover": {
            "low_q_physical_rad": corrected,
            "hover_q_physical_rad": profile["trajectory"][
                "canonical_hover_q_physical_rad"
            ],
        },
        "stages": {
            "descent": {
                "final_right_ee_wxyz_xyz": [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.8]
            }
        },
        "preclose": {
            "allowed_to_close": False,
            "observation": {
                "source": "blue_marker",
                "component_touches_border": False,
                "grasp_window": {"normalized_center_error": 0.26},
            },
        },
        "visual_replan": {
            "method": "fixed_camera_physical_xy_trust_region_broyden",
            "selected_delta_xy_m": [-0.003, 0.002],
            "target_low_xy_m": [0.05, 0.087],
            "corrected_q_physical_rad": corrected,
            "ik": {
                "accepted": True,
                "role": "level_yaw_free_low_visual_servo",
            },
        },
    }
    monkeypatch.setattr(
        runner,
        "assess_jaw_level",
        lambda *_args, **_kwargs: SimpleNamespace(
            accepted=False,
            combined_tilt_deg=2.6,
            tip_height_difference_m=0.0034,
        ),
    )
    replay = _preclose_correction_replay_alignment(profile, attempt, corrected)
    assert replay["source_level_strictly_accepted"] is False
    assert replay["source_level_seed_usable"] is True
    assert replay["fresh_preclose_required"] is True
    assert replay["closure_authorized"] is False


def test_pending_preclose_alignment_round_trips_but_never_authorizes_close(tmp_path):
    profile = load_profile(PROFILE)
    profile["target_identity"] = dict(profile["target_identity"])
    path = tmp_path / "pending.json"
    profile["target_identity"]["pending_runtime_alignment_calibration"] = str(path)
    q = profile["trajectory"]["verified_preclose_q_physical_rad"]
    localization = {
        "target_center_scene_xyz_m": [0.01, 0.02, -0.5],
        "marker": {"center_uv": [0.44, 0.49], "center_px": [844.0, 709.0]},
    }
    alignment = {
        "accepted": True,
        "method": "semantically_verified_preclose_level_q_correction",
        "prior_normalized_hover_error": 0.0,
        "aligned_low_q_physical_rad": q,
        "closure_authorized": False,
        "fresh_preclose_required": True,
    }
    saved = _save_pending_preclose_alignment(
        profile,
        localization,
        alignment,
        source_run="run-a",
        source_attempt="attempt_02",
    )
    assert saved["saved"] is True
    loaded, report = _load_pending_preclose_alignment(profile, localization)
    assert report["accepted"] is True
    assert loaded["closure_authorized"] is False
    assert loaded["fresh_preclose_required"] is True
    np.testing.assert_allclose(loaded["aligned_low_q_physical_rad"], q)

    moved = dict(localization)
    moved["marker"] = {"center_uv": [0.50, 0.49], "center_px": [960.0, 709.0]}
    loaded, report = _load_pending_preclose_alignment(profile, moved)
    assert loaded is None
    assert report["accepted"] is False


def test_pending_hover_progress_resumes_vision_but_never_skips_it(tmp_path):
    profile = load_profile(PROFILE)
    profile["target_identity"] = dict(profile["target_identity"])
    path = tmp_path / "pending_hover.json"
    profile["target_identity"]["pending_hover_alignment_calibration"] = str(path)
    localization = {
        "target_center_scene_xyz_m": profile["head_localization"][
            "reference_target_center_scene_xyz_m"
        ],
        "marker": {"center_uv": [0.44, 0.49], "center_px": [844.0, 709.0]},
    }
    progress = {
        "low_q_physical_rad": profile["trajectory"][
            "verified_preclose_q_physical_rad"
        ],
        "hover_q_physical_rad": profile["trajectory"][
            "canonical_hover_q_physical_rad"
        ],
        "normalized_center_error": 0.23,
        "target_observation": {
            "center_px": [167.0, 278.0],
            "component_pixels": 352,
            "source": "local_blue_evidence_continuity",
            "marker_cross_shaped": False,
        },
        "direct_cross_verified": True,
        "tool_frame": {
            "origin_px": [344.0, 333.0],
            "forward_xy": [-0.80, 0.60],
            "lateral_xy": [0.60, 0.80],
            "scale_px": 153.0,
            "cyan_pixels": 10000,
            "light_pad_pixels": 1600,
        },
    }
    saved = runner._save_pending_hover_progress(
        profile,
        localization,
        progress,
        {"jacobian": [[12.0, -4.0], [7.0, 40.0]], "best_error_norm": 0.23},
        source_run="run-hover",
        source_attempt="attempt_01",
    )
    assert saved["saved"] is True
    loaded, report = runner._load_pending_hover_progress(profile, localization)
    assert report["accepted"] is True
    assert loaded["closure_authorized"] is False
    assert loaded["fresh_hover_observation_required"] is True
    np.testing.assert_allclose(
        loaded["low_q_physical_rad"], progress["low_q_physical_rad"]
    )
    np.testing.assert_allclose(
        loaded["hover_q_physical_rad"], progress["hover_q_physical_rad"]
    )
    np.testing.assert_allclose(
        loaded["servo_seed"]["jacobian"], [[12.0, -4.0], [7.0, 40.0]]
    )
    assert loaded["servo_seed"]["adaptive_trust_from_first_observation"] is True
    assert loaded["target_observation"]["component_pixels"] == 352
    assert loaded["direct_cross_verified"] is True
    assert "best_error_norm" not in loaded["servo_seed"]
    assert "best_low_q_physical_rad" not in loaded["servo_seed"]

    moved = dict(localization)
    moved["target_center_scene_xyz_m"] = [0.04, 0.02, -0.5]
    loaded, report = runner._load_pending_hover_progress(profile, moved)
    assert loaded is None
    assert report["accepted"] is False

    invalidated = runner._invalidate_pending_hover_progress(
        profile, source_run="run-b", reason="fresh camera identity was occluded"
    )
    assert invalidated["invalidated"] is True
    retained = json.loads(path.read_text())
    assert retained["active"] is False
    assert retained["source_run"] == "run-hover"
    assert retained["invalidation_reason"] == "fresh camera identity was occluded"


def test_pending_hover_rejects_head_consistent_but_physically_wrong_pose(tmp_path):
    profile = copy.deepcopy(load_profile(PROFILE))
    path = tmp_path / "poisoned_pending_hover.json"
    profile["target_identity"]["pending_hover_alignment_calibration"] = str(path)
    scene = profile["head_localization"]["reference_target_center_scene_xyz_m"]
    localization = {
        "target_center_scene_xyz_m": scene,
        "marker": {"center_uv": [0.44, 0.49], "center_px": [844.0, 709.0]},
    }
    wrong_low_q = [
        0.11090988481577974,
        0.3174161821745911,
        -0.2771571114861145,
        0.17480453843142973,
        -0.05240102021251402,
        2.1758419613990947,
    ]
    progress = {
        "low_q_physical_rad": wrong_low_q,
        "hover_q_physical_rad": profile["trajectory"][
            "canonical_hover_q_physical_rad"
        ],
        "normalized_center_error": 0.20,
        "target_observation": {
            "center_px": [155.0, 365.0],
            "component_pixels": 900,
            "source": "local_blue_evidence_continuity",
        },
        "direct_cross_verified": True,
        "tool_frame": {
            "origin_px": [344.0, 333.0],
            "forward_xy": [-0.80, 0.60],
            "lateral_xy": [0.60, 0.80],
            "scale_px": 153.0,
            "cyan_pixels": 10000,
            "light_pad_pixels": 1600,
        },
    }
    runner._save_pending_hover_progress(
        profile,
        localization,
        progress,
        {"jacobian": [[12.0, -4.0], [7.0, 40.0]]},
        source_run="wrong-dish-run",
        source_attempt="attempt_01",
    )

    loaded, report = runner._load_pending_hover_progress(profile, localization)

    assert loaded is None
    assert report["accepted"] is False
    assert report["cross_modal_geometry"]["residual_m"] > 0.03
    assert "geometry" in report["reason"]


def test_empty_local_identity_at_exact_camera_replay_forces_geometric_retry():
    attempt = {
        "stages": {"approach": {"precise_camera_replay": True}},
        "hover_iterations": [],
        "hover_perception_or_servo_error": (
            "ValueError: no local blue-evidence target candidate is visible"
        ),
    }
    assert runner._stale_precise_camera_replay_reason(attempt) == (
        "no local blue-evidence target candidate is visible"
    )

    # Once a fresh servo iteration exists, this is an ordinary recoverable
    # tracking loss; retaining its best progress is preferable to discarding it.
    attempt["hover_iterations"] = [{"normalized_error": 0.2}]
    assert runner._stale_precise_camera_replay_reason(attempt) is None

    # Coarse geometric approaches never invalidate a saved exact-camera view.
    attempt["hover_iterations"] = []
    attempt["stages"]["approach"]["precise_camera_replay"] = False
    assert runner._stale_precise_camera_replay_reason(attempt) is None


def test_branch_escape_reuses_only_identity_consistent_hover_progress():
    progress = {
        "direct_cross_verified": True,
        "target_observation": {"source": "local_blue_evidence_continuity"},
    }
    assert runner._identity_consistent_hover_progress(progress) is True

    # A global blue-region selection after the camera branch changes can be
    # the neighbouring dish even if an old boolean was accidentally retained.
    progress["target_observation"]["source"] = "semantic_global_candidate"
    assert runner._identity_consistent_hover_progress(progress) is False

    progress["target_observation"]["source"] = "blue_marker"
    progress["direct_cross_verified"] = False
    assert runner._identity_consistent_hover_progress(progress) is False

def test_rejected_runtime_alignment_does_not_export_a_live_servo_seed():
    # The geometry rejection may retain the old Jacobian in its diagnostic
    # report, but it must not influence a fresh head-geometric attempt.
    report = {"servo_state_seed": {"jacobian": [[1.0, 2.0], [3.0, 4.0]]}}
    assert runner._accepted_runtime_servo_seed(None, report) == {}
    assert runner._accepted_runtime_servo_seed(np.zeros(6), report) == (
        report["servo_state_seed"]
    )


def test_rejected_runtime_pose_keeps_identity_but_drops_stage_parallax():
    report = {
        "servo_state_seed": {
            "jacobian": [[1.0, 0.0], [0.0, 1.0]],
            "vertical_descent_target_pixel_delta_px": [130.0, -51.0],
            "hover_identity_anchor": {
                "center_px": [34.0, 193.0],
                "component_pixels": 700,
            },
        }
    }
    assert runner._accepted_runtime_servo_seed(None, report) == {
        "hover_identity_anchor": report["servo_state_seed"][
            "hover_identity_anchor"
        ]
    }

def test_dark_preclose_pad_is_recovery_visible_but_cannot_close():
    image_path = (
        ROOT
        / "data/runs/pasteur/codexless_fixedhead_levelyaw_grasp_20260805"
        / "attempt_03/preclose.png"
    )
    if not image_path.exists():
        return
    profile = load_profile(PROFILE)
    template = GraspWindowTemplate.from_dict(json.loads(GOAL.read_text())["template"])
    observation = observe_marked_target(
        cv2.imread(str(image_path)),
        template,
        maximum_candidate_distance_tool_units=4.5,
        reference_component_area_per_tool_scale_sq=profile["target_identity"][
            "right_reference_component_area_per_tool_scale_sq"
        ],
        minimum_reference_area_fraction=0.1,
        maximum_reference_area_fraction=3.0,
        proximity_score_weight=2.0,
        fallback_minimum_light_value=45,
    )
    assert observation.tool_frame_source == "light_pad_dark_scene_fallback"
    assert observation.grasp_window.allowed_to_close is False
    assert observation.grasp_window.normalized_center_error > 1.0


def test_bright_hover_prefers_direct_cross_over_merged_translucent_rim():
    image_path = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v23"
        / "attempt_01/hover_00.png"
    )
    if not image_path.exists():
        return
    profile = load_profile(PROFILE)
    template = GraspWindowTemplate.from_dict(json.loads(GOAL.read_text())["template"])
    expected = runner.ToolImageFrame(
        **json.loads(
            (
                image_path.parent / "attempt.json"
            ).read_text()
        )["hover_iterations"][0]["observation"]["tool_frame"]
    )
    observation = observe_marked_target(
        cv2.imread(str(image_path)),
        template,
        reference_component_area_per_tool_scale_sq=profile["target_identity"][
            "right_reference_component_area_per_tool_scale_sq"
        ],
        minimum_reference_area_fraction=profile["target_identity"][
            "minimum_right_reference_area_fraction"
        ],
        maximum_reference_area_fraction=profile["target_identity"][
            "maximum_right_reference_area_fraction"
        ],
        expected_tool_frame=expected,
    )
    # The direct printed cross is near (202, 327).  The permissive translucent
    # component merges it with the lid rim near (194, 331) and has ~932 px.
    np.testing.assert_allclose(observation.center_px, [201.7, 327.1], atol=2.0)
    assert observation.component_pixels < 400
    assert observation.marker_cross_shaped is True


def test_occluded_saved_view_does_not_call_transparent_rim_a_direct_cross():
    image_path = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v111_stale_view_fallback"
        / "attempt_01/hover_00.png"
    )
    if not image_path.exists():
        return
    profile = load_profile(PROFILE)
    template = GraspWindowTemplate.from_dict(json.loads(GOAL.read_text())["template"])
    observation = observe_marked_target(
        cv2.imread(str(image_path)),
        template,
        maximum_candidate_distance_tool_units=profile["perception"][
            "maximum_candidate_distance_tool_units"
        ],
        reference_component_area_per_tool_scale_sq=profile["target_identity"][
            "right_reference_component_area_per_tool_scale_sq"
        ],
        minimum_reference_area_fraction=profile["target_identity"][
            "minimum_right_reference_area_fraction"
        ],
        maximum_reference_area_fraction=profile["target_identity"][
            "maximum_right_reference_area_fraction"
        ],
        expected_tool_frame=runner.ToolImageFrame(
            **profile["target_identity"]["fixed_right_tool_frame"]
        ),
        prefer_expected_tool_frame=True,
        prefer_cross_shape=True,
    )
    assert observation.marker_cross_shaped is False


def test_local_blue_continuity_keeps_transparent_lid_instead_of_cyan_tool():
    attempt_dir = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v113_free_camera_resume"
        / "attempt_01"
    )
    before_path = attempt_dir / "hover_03.png"
    after_path = attempt_dir / "hover_04.png"
    if not before_path.exists() or not after_path.exists():
        return
    profile = load_profile(PROFILE)
    template = GraspWindowTemplate.from_dict(json.loads(GOAL.read_text())["template"])
    expected = runner.ToolImageFrame(
        **json.loads((attempt_dir / "attempt.json").read_text())["hover_iterations"][-1][
            "observation"
        ]["tool_frame"]
    )
    kwargs = dict(
        maximum_candidate_distance_tool_units=profile["perception"][
            "maximum_candidate_distance_tool_units"
        ],
        reference_component_area_per_tool_scale_sq=profile["target_identity"][
            "right_reference_component_area_per_tool_scale_sq"
        ],
        minimum_reference_area_fraction=profile["target_identity"][
            "minimum_right_reference_area_fraction"
        ],
        maximum_reference_area_fraction=profile["target_identity"][
            "maximum_right_reference_area_fraction"
        ],
        expected_tool_frame=expected,
        prefer_expected_tool_frame=True,
        prefer_cross_shape=True,
    )
    previous = observe_marked_target(cv2.imread(str(before_path)), template, **kwargs)
    global_after = observe_marked_target(cv2.imread(str(after_path)), template, **kwargs)
    continued = observe_local_blue_evidence_target(
        cv2.imread(str(after_path)), previous, template
    )
    np.testing.assert_allclose(continued.center_px, [165.5, 281.0], atol=2.0)
    assert np.linalg.norm(
        np.asarray(global_after.center_px) - np.asarray(continued.center_px)
    ) > 50.0
    assert continued.source == "local_blue_evidence_continuity"


def test_audited_tool_frame_rejects_dark_false_white_pad():
    bright_attempt = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v6"
        / "attempt_01/attempt.json"
    )
    dark_image = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v7"
        / "attempt_01/hover_00.png"
    )
    if not bright_attempt.exists() or not dark_image.exists():
        return
    profile = load_profile(PROFILE)
    template = GraspWindowTemplate.from_dict(json.loads(GOAL.read_text())["template"])
    bright = json.loads(bright_attempt.read_text())
    record = next(item for item in bright["hover_iterations"] if item["index"] == 11)
    expected = runner.ToolImageFrame(**record["observation"]["tool_frame"])
    observation = observe_marked_target(
        cv2.imread(str(dark_image)),
        template,
        maximum_candidate_distance_tool_units=profile["perception"][
            "maximum_candidate_distance_tool_units"
        ],
        reference_component_area_per_tool_scale_sq=profile["target_identity"][
            "right_reference_component_area_per_tool_scale_sq"
        ],
        minimum_reference_area_fraction=profile["target_identity"][
            "minimum_right_reference_area_fraction"
        ],
        maximum_reference_area_fraction=profile["target_identity"][
            "maximum_right_reference_area_fraction"
        ],
        fallback_minimum_light_value=profile["perception"][
            "fallback_minimum_light_value"
        ],
        expected_tool_frame=expected,
    )
    assert observation.tool_frame_source == "audited_expected_tool_frame"
    goal = np.asarray(profile["target_identity"]["canonical_hover_goal_uv"])
    error = np.linalg.norm(np.asarray(observation.center_uv) - goal) / template.square_side_u
    assert error < 0.35


def test_rigid_tool_frame_survives_changed_bench_illumination():
    image_path = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v39"
        / "attempt_01/hover_00.png"
    )
    runtime_path = (
        ROOT / "data/runs/pasteur/codexless_thin_object_runtime_alignment.json"
    )
    if not image_path.exists() or not runtime_path.exists():
        return
    profile = load_profile(PROFILE)
    template = GraspWindowTemplate.from_dict(json.loads(GOAL.read_text())["template"])
    expected = runner.ToolImageFrame(
        **json.loads(runtime_path.read_text())["tool_frame"]
    )
    observation = observe_marked_target(
        cv2.imread(str(image_path)),
        template,
        maximum_candidate_distance_tool_units=profile["perception"][
            "maximum_candidate_distance_tool_units"
        ],
        reference_component_area_per_tool_scale_sq=profile["target_identity"][
            "right_reference_component_area_per_tool_scale_sq"
        ],
        minimum_reference_area_fraction=profile["target_identity"][
            "minimum_right_reference_area_fraction"
        ],
        maximum_reference_area_fraction=profile["target_identity"][
            "maximum_right_reference_area_fraction"
        ],
        fallback_minimum_light_value=profile["perception"][
            "fallback_minimum_light_value"
        ],
        expected_tool_frame=expected,
        prefer_expected_tool_frame=True,
    )
    assert observation.tool_frame_source == "rigid_expected_tool_frame"
    np.testing.assert_allclose(observation.tool_frame.origin_px, expected.origin_px)
    assert observation.tool_frame.scale_px == expected.scale_px
    goal = np.asarray(profile["target_identity"]["canonical_hover_goal_uv"])
    error = np.linalg.norm(np.asarray(observation.center_uv) - goal) / template.square_side_u
    # This archived image predates the restored longest-grip goal, so it is no
    # longer expected to be aligned.  The illumination regression protects
    # tool-frame and target detection only.
    assert np.isfinite(error)


def test_preclose_level_yaw_free_replan_stays_on_local_teleop_branch():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    aligned_low = np.asarray(
        [
            0.1568819322,
            0.0561370052,
            -0.1080729213,
            0.1463693396,
            -0.0116750811,
            2.1980521650,
        ]
    )
    measured_xy = np.asarray(fk.pose(aligned_low).parameters()[4:6])
    goal = np.asarray(profile["target_identity"]["canonical_hover_goal_uv"])
    observation = SimpleNamespace(center_uv=goal + np.asarray([-0.04, 0.27]))
    corrected, report = runner._metric_replan(
        profile,
        observation,
        aligned_low,
        fk=fk,
        reference_center_uv=goal,
        servo_state={},
        measured_current_xy_m=measured_xy,
        level_yaw_free=True,
    )
    assert report["ik"]["role"] == "level_yaw_free_low_visual_servo"
    assert report["ik"]["level"]["accepted"] is True
    # The runtime seed is intentionally updated after a newly aligned hover;
    # a local yaw-free correction may move wrist compensation slightly more
    # than the old fixed 0.15 rad snapshot while remaining on the same branch.
    assert np.max(np.abs(corrected - aligned_low)) < 0.20


def test_corrected_hover_preserves_previous_horizontal_camera_yaw():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    first_path = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v2"
        / "attempt_01/attempt.json"
    )
    second_path = (
        ROOT
        / "data/runs/pasteur/codexless_replay_pipeline_20260805_v2"
        / "attempt_02/attempt.json"
    )
    if not first_path.exists() or not second_path.exists():
        return
    source_attempt = json.loads(
        first_path.read_text()
    )
    source_hover_q = np.asarray(source_attempt["hover"]["hover_q_physical_rad"])
    corrected_low_q = np.asarray(
        json.loads(second_path.read_text())["visual_replan"][
            "corrected_q_physical_rad"
        ]
    )
    hover_q, report = runner._plan_level_vertical_offset(
        profile,
        fk,
        corrected_low_q,
        profile["trajectory"]["verification_lift_m"],
        seed_hover_q=source_hover_q,
    )
    source_x = fk.pose(source_hover_q).as_matrix()[:3, 0]
    corrected_x = fk.pose(hover_q).as_matrix()[:3, 0]
    yaw_error = np.degrees(
        np.arccos(np.clip(np.dot(source_x, corrected_x), -1.0, 1.0))
    )
    if report["role"] == "level_fixed_yaw_hover":
        assert yaw_error <= profile["perception"]["maximum_hover_yaw_error_deg"]
    elif report["role"] == "canonical_hover":
        canonical_q = np.asarray(
            profile["trajectory"]["canonical_hover_q_physical_rad"]
        )
        canonical_x = fk.pose(canonical_q).as_matrix()[:3, 0]
        assert np.degrees(
            np.arccos(np.clip(np.dot(canonical_x, corrected_x), -1.0, 1.0))
        ) <= profile["perception"]["maximum_hover_yaw_error_deg"]
    else:
        assert report["role"] == (
            "level_yaw_free_hover_away_from_joint6_limit"
        )
        assert report["orientation_mode_change"] == (
            "horizontal_yaw_released_to_avoid_joint6_limit"
        )
        _, upper = runner._joint_bounds(profile, fk)
        assert upper[5] - hover_q[5] > profile["head_localization"][
            "joint6_hover_limit_avoidance_margin_rad"
        ]
    assert report["level"]["accepted"] is True
    assert report["collision"]["accepted"] is True


def test_unseen_workspace_hover_releases_yaw_only_at_joint_limit():
    profile = load_profile(PROFILE)
    fk = runner.ProductionRightFK(profile["production_model"])
    low_q, _ = runner._plan_level_yaw_free_pose(
        profile,
        fk,
        target_position=np.asarray(
            [0.1098973867, 0.0513988077, 0.8407808198]
        ),
        seed_q=np.asarray(
            [-0.0874780, 0.5143336, -0.1074534, 0.1598888, -0.2008061, 2.1959141]
        ),
        role="unseen_workspace_test_low",
    )
    hover_q, report = runner._plan_level_vertical_offset(
        profile,
        fk,
        low_q,
        profile["trajectory"]["verification_lift_m"],
        seed_hover_q=np.asarray(
            [-0.1851840, 0.5200299, -0.2314192, -0.6409143, 0.1312540, 2.98]
        ),
    )
    assert hover_q.shape == (6,)
    assert report["role"] == "level_yaw_free_hover_after_fixed_yaw_limit"
    assert report["orientation_mode_change"] == (
        "horizontal_yaw_released_at_joint_limit"
    )
    assert report["level"]["accepted"] is True
    assert report["collision"]["accepted"] is True


def test_relocated_target_gets_horizontal_collision_audited_rim_pose(tmp_path):
    profile = load_profile(PROFILE)
    # Runtime alignment is deliberately mutable during live experiments.  A
    # unit test must use a fixed proven anchor instead of whichever failed or
    # successful physical run happened most recently.
    calibration = {
        "schema": runner.RUNTIME_ALIGNMENT_SCHEMA,
        "canonical_hover_goal_uv": profile["target_identity"][
            "canonical_hover_goal_uv"
        ],
        "canonical_preclose_goal_uv": profile["target_identity"][
            "canonical_preclose_goal_uv"
        ],
        "target_center_scene_xyz_m": profile["head_localization"][
            "reference_target_center_scene_xyz_m"
        ],
        "low_q_physical_rad": profile["trajectory"][
            "verified_preclose_q_physical_rad"
        ],
    }
    calibration_path = tmp_path / "runtime_alignment.json"
    calibration_path.write_text(json.dumps(calibration))
    profile["target_identity"]["runtime_alignment_calibration"] = str(
        calibration_path
    )
    q, report = runner._plan_relocated_level_rim(
        profile,
        np.asarray([-0.01965009, 0.88337186, -0.56304601]),
    )
    assert q.shape == (6,)
    assert report["selected"]["level"]["accepted"] is True
    assert report["selected"]["collision"]["accepted"] is True
    assert report["method"] == (
        "fixed_tag_scene_delta_plus_empirical_physical_right_yaw"
    )
    assert report["head_scene_to_production_translation_applied"] is True
    assert report["closure_authorized"] is False
    assert report["fresh_wrist_alignment_required"] is True
    assert not np.allclose(
        q,
        profile["trajectory"]["verified_preclose_q_physical_rad"],
    )


def test_relocation_geometry_survives_stale_runtime_image_goals(tmp_path):
    profile = load_profile(PROFILE)
    calibration = {
        "schema": runner.RUNTIME_ALIGNMENT_SCHEMA,
        "canonical_hover_goal_uv": [99.0, 99.0],
        "canonical_preclose_goal_uv": None,
        "target_center_scene_xyz_m": [0.0, 0.0, 0.0],
        "low_q_physical_rad": [0.0] * 6,
    }
    calibration_path = tmp_path / "stale_image_runtime_alignment.json"
    calibration_path.write_text(json.dumps(calibration))
    profile["target_identity"]["runtime_alignment_calibration"] = str(
        calibration_path
    )
    q, report = runner._plan_relocated_level_rim(
        profile,
        np.asarray([-0.01621217, 0.87460622, -0.56304601]),
    )
    assert report["head_scene_to_production_translation_applied"] is True
    assert report["relocation_calibration"]["accepted"] is True
    assert report["relocation_calibration"]["runtime_anchor_accepted"] is False
    assert report["relocation_calibration"]["anchor_source"] == (
        "configured_reference_scene_and_verified_preclose_pose"
    )
    assert report["selected"]["collision"]["accepted"] is True
    assert not np.allclose(
        q, profile["trajectory"]["verified_preclose_q_physical_rad"]
    )


def test_lift_and_placement_keep_close_command_at_zero(monkeypatch):
    commanded = []

    def cartesian(*_args, **kwargs):
        commanded.append((kwargs["stage"], kwargs["aperture"]))
        return {}

    class RPC:
        def get_right_joint_positions(self):
            return np.zeros(6)

        def get_right_ee_pose(self):
            class Pose:
                def parameters(self):
                    return [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.8]

            return Pose()

    profile = load_profile(PROFILE)
    profile["trajectory"]["support_settle_s"] = 0.0
    monkeypatch.setattr(runner, "_cartesian_move", cartesian)
    monkeypatch.setattr(
        runner,
        "_plan_straight_level_joint_lift",
        lambda *_a, **kwargs: commanded.append(
            ("verification_lift_straight_level", kwargs["aperture"])
        )
        or ([SimpleNamespace()], {"accepted": True}),
    )
    monkeypatch.setattr(
        runner, "_execute_direct_joint_samples", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(runner, "_fixed_pose_gripper_ramp", lambda *_a, **_k: {})
    monkeypatch.setattr(runner, "_retreat_open", lambda *_a, **_k: {})
    runner._straight_lift(profile, RPC(), None)
    runner._place_open_retreat(
        profile,
        RPC(),
        None,
        [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.8],
    )
    assert commanded == [
        ("verification_lift_straight_level", 0.0),
        ("return_object_to_support", 0.0),
    ]


def test_staged_lift_stops_after_short_stage_when_obstruction_disappears(
    monkeypatch,
):
    profile = load_profile(PROFILE)
    calls = []

    class RPC:
        def get_right_gripper_exact(self):
            return 0.005

        def get_right_joint_positions(self):
            return np.zeros(6)

        def set_right_joint_target(self, *_args, **_kwargs):
            raise AssertionError("preload must not be applied after an empty pickup")

    monkeypatch.setattr(
        runner,
        "_straight_lift",
        lambda *_args, **kwargs: calls.append(kwargs) or {"commands_sent": True},
    )
    calibration = runner.ClosureCalibration(
        tuple(profile["closure"]["empty_reference_ratios"]),
        tuple(profile["closure"]["nonempty_reference_ratios"]),
    )
    report = runner._staged_straight_lift(
        profile, RPC(), object(), calibration
    )
    assert len(calls) == 1
    assert calls[0]["distance_m"] == profile["trajectory"][
        "initial_hold_check_lift_m"
    ]
    assert calls[0]["aperture"] == 0.0
    assert report["closure_after_initial_lift"]["nonempty"] is False
    assert report["remaining"] is None
    assert report["completed_full_distance"] is False


def test_staged_lift_continues_vertically_when_obstruction_persists(monkeypatch):
    profile = load_profile(PROFILE)
    calls = []

    class RPC:
        def get_right_gripper_exact(self):
            return 0.58

        def get_right_joint_positions(self):
            return np.zeros(6)

        def set_right_joint_target(self, _q, *, gripper_target, preview_time):
            calls.append(
                {
                    "stage": "post_pickup_preload_command",
                    "aperture": gripper_target,
                    "preview_time": preview_time,
                    "distance_m": 0.0,
                }
            )

    monkeypatch.setattr(
        runner,
        "_straight_lift",
        lambda *_args, **kwargs: calls.append(kwargs) or {"commands_sent": True},
    )
    calibration = runner.ClosureCalibration(
        tuple(profile["closure"]["empty_reference_ratios"]),
        tuple(profile["closure"]["nonempty_reference_ratios"]),
    )
    report = runner._staged_straight_lift(
        profile, RPC(), object(), calibration
    )
    motion_calls = [call for call in calls if call["distance_m"] > 0]
    assert len(motion_calls) == 2
    assert np.isclose(
        sum(call["distance_m"] for call in motion_calls),
        profile["trajectory"]["verification_lift_m"],
    )
    assert motion_calls[0]["aperture"] == 0.0
    assert motion_calls[1]["aperture"] == report["holding_aperture"]
    assert all(
        call["stage"].startswith("verification_lift_")
        for call in motion_calls
    )
    assert report["closure_after_initial_lift"]["nonempty"] is True
    assert report["completed_full_distance"] is True


def test_partial_cartesian_recovery_waits_for_measured_joint_settle():
    samples = iter(
        [
            np.zeros(6),
            np.full(6, 0.20),
            np.full(6, 0.21),
            np.full(6, 0.215),
            np.full(6, 0.217),
        ]
    )

    class RPC:
        def get_right_joint_positions(self):
            return next(samples)

    now = {"value": 0.0}

    def clock():
        now["value"] += 0.01
        return now["value"]

    report = runner._wait_for_right_joint_settle(
        RPC(),
        timeout_s=1.0,
        poll_s=0.01,
        maximum_delta_rad=0.015,
        required_consecutive=3,
        clock=clock,
        sleep=lambda _duration: None,
    )
    assert report["accepted"] is True
    assert report["read_only"] is True
    assert report["sample_count"] == 5
    np.testing.assert_allclose(report["q_physical_rad"], np.full(6, 0.217))


def test_hover_level_failure_converges_audited_endpoint_once(monkeypatch):
    profile = load_profile(PROFILE)
    calls = []

    class RPC:
        q = np.full(6, 0.1)

        def get_right_joint_positions(self):
            return self.q

    class FK:
        def pose(self, q):
            q = np.asarray(q)
            return SimpleNamespace(
                parameters=lambda: np.r_[1.0, 0.0, 0.0, 0.0, q[:3]]
            )

    class Checkpoint:
        def __init__(self):
            self.calls = []

        def require(self, name):
            self.calls.append(name)
            if len(self.calls) == 1:
                raise RuntimeError("fingertip heights differ")
            return SimpleNamespace(accepted=True)

    def execute(_profile, _rpc, _fk, samples, **kwargs):
        calls.append((list(samples), kwargs))
        _rpc.q = np.full(6, 0.2)
        return {"commands_sent": True}

    monkeypatch.setattr(runner, "_execute_direct_joint_samples", execute)
    checkpoint = Checkpoint()
    assessment, report = runner._require_hover_level_after_endpoint_convergence(
        profile,
        RPC(),
        FK(),
        np.full(6, 0.2),
        checkpoint,
    )
    assert assessment.accepted is True
    assert checkpoint.calls == [
        "before_descend",
        "before_descend_after_endpoint_correction",
    ]
    assert report["endpoint_correction_required"] is True
    assert len(calls) == 1
    assert calls[0][1]["final_tolerance_rad"] == 0.008
    assert calls[0][1]["endpoint_correction_gain"] == 1.0


def test_hover_level_pass_sends_no_endpoint_command(monkeypatch):
    profile = load_profile(PROFILE)

    class RPC:
        def get_right_joint_positions(self):
            return np.full(6, 0.2)

    class FK:
        def pose(self, q):
            q = np.asarray(q)
            return SimpleNamespace(
                parameters=lambda: np.r_[1.0, 0.0, 0.0, 0.0, q[:3]]
            )

    class Checkpoint:
        def require(self, name):
            assert name == "before_descend"
            return SimpleNamespace(accepted=True)

    monkeypatch.setattr(
        runner,
        "_execute_direct_joint_samples",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("level endpoint correction must not run")
        ),
    )
    assessment, report = runner._require_hover_level_after_endpoint_convergence(
        profile,
        RPC(),
        FK(),
        np.full(6, 0.2),
        Checkpoint(),
    )
    assert assessment.accepted is True
    assert report["endpoint_correction_required"] is False


def test_level_hover_uses_measured_origin_when_joint_endpoint_is_short(monkeypatch):
    profile = load_profile(PROFILE)
    planned = np.full(6, 0.2)
    calls = []

    class RPC:
        q = np.full(6, 0.18)

        def get_right_joint_positions(self):
            return self.q

    class FK:
        def pose(self, q):
            q = np.asarray(q)
            return SimpleNamespace(
                parameters=lambda: np.r_[1.0, 0.0, 0.0, 0.0, q[:3]]
            )

    class Checkpoint:
        def require(self, _name):
            return SimpleNamespace(accepted=True)

    def execute(_profile, rpc, _fk, _samples, **kwargs):
        calls.append(kwargs)
        rpc.q = planned.copy()
        return {"commands_sent": True}

    monkeypatch.setattr(runner, "_execute_direct_joint_samples", execute)
    assessment, report = runner._require_hover_level_after_endpoint_convergence(
        profile, RPC(), FK(), planned, Checkpoint()
    )
    assert assessment.accepted is True
    assert report["endpoint_correction_required"] is False
    assert report["initial_branch_error"]["joint_error_rad"] > 0.019
    assert report["descent_origin"] == "fresh_measured_level_hover"
    assert calls == []


def test_hover_level_defers_small_joint_proxy_timeout_to_full_descent_audit(
    monkeypatch,
):
    profile = load_profile(PROFILE)

    class RPC:
        def get_right_joint_positions(self):
            return np.zeros(6)

    class FK:
        def pose(self, q):
            q = np.asarray(q)
            return SimpleNamespace(
                parameters=lambda: np.r_[1.0, 0.0, 0.0, 0.0, q[:3]]
            )

    class Checkpoint:
        count = 0

        def require(self, _name):
            self.count += 1
            if self.count == 1:
                raise RuntimeError("initially unlevel")
            return SimpleNamespace(accepted=True)

    monkeypatch.setattr(
        runner,
        "_execute_direct_joint_samples",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            runner.TrajectoryStreamError("joint proxy stopped at 0.014 rad")
        ),
    )
    assessment, report = runner._require_hover_level_after_endpoint_convergence(
        profile, RPC(), FK(), np.zeros(6), Checkpoint()
    )
    assert assessment.accepted is True
    assert "0.014 rad" in report["motion_error"]


def test_hover_level_uses_audited_joint_roll_pitch_correction_before_descent(
    monkeypatch,
):
    profile = load_profile(PROFILE)
    planner_calls = []

    class Pose:
        def parameters(self):
            return [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.8]

    class RPC:
        def get_right_joint_positions(self):
            return np.zeros(6)

        def get_right_ee_pose(self):
            return Pose()

    class FK:
        def pose(self, q):
            q = np.asarray(q)
            return SimpleNamespace(
                parameters=lambda: np.r_[1.0, 0.0, 0.0, 0.0, q[:3]]
            )

    class Checkpoint:
        count = 0
        reference = runner._load_level_reference(profile["level_config"])

        def require(self, _name):
            self.count += 1
            if self.count < 3:
                raise RuntimeError("not level yet")
            return SimpleNamespace(accepted=True)

    monkeypatch.setattr(
        runner, "_execute_direct_joint_samples", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        runner,
        "_plan_same_position_level_joint_samples",
        lambda *_a, **kwargs: planner_calls.append(kwargs)
        or ([SimpleNamespace()], {"accepted": True}),
    )
    assessment, report = runner._require_hover_level_after_endpoint_convergence(
        profile, RPC(), FK(), np.zeros(6), Checkpoint()
    )
    assert assessment.accepted is True
    assert len(planner_calls) == 1
    assert report["joint_level_audit"]["accepted"] is True
    assert report["joint_level_motion"] == {}


def test_partial_cartesian_recovery_retries_from_fresh_measured_pose(monkeypatch):
    profile = load_profile(PROFILE)
    calls = []

    class Streamer:
        def recover_vertical_then_open(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise runner.TrajectoryStreamError("partial IK rejection")
            return {"completed": True, "command_count": 12}

    monkeypatch.setattr(runner, "_streamer", lambda *_args: Streamer())
    monkeypatch.setattr(
        runner,
        "_wait_for_right_joint_settle",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "read_only": True,
            "q_physical_rad": [0.0] * 6,
        },
    )
    report = runner._recover_vertical_then_open(profile, object(), object())
    assert len(calls) == 2
    assert report["completed"] is True
    assert report["method"] == "cartesian_retry_after_partial_rejection"
    assert "partial IK rejection" in report["initial_error"]
    assert report["settle_before_retry"]["read_only"] is True
