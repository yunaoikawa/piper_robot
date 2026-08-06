from pathlib import Path

import json

from rollout.cylindrical_cap_transfer import (
    CapTransferFrame,
    audit_cap_transfer_captures,
    validate_hardware_route_replay,
    validate_lift_transition,
    validate_transport_transition,
    waypoint_route_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "src/configs/pasteur_culture_media_cap_grasp.json"
CAPTURES = ROOT / "data/captures/pasteur/2026-08-06"
BEFORE = CAPTURES / (
    "20260806T090454.207716Z_head_culture_media_cap_"
    "hold_before_lift_ce98d39c"
)
LIFT = CAPTURES / (
    "20260806T090516.450533Z_head_culture_media_cap_"
    "lift_probe10_792a8b3c"
)
TRANSPORT = CAPTURES / (
    "20260806T090638.703002Z_head_culture_media_cap_"
    "transport_home_hold_c01dae62"
)


def _frame(*, xyz, aperture=0.2, white=100, support=(50, 80), jaw_y=50):
    return CapTransferFrame(
        capture="synthetic",
        target_anchor_px=(50.0, 50.0),
        jaw_center_px=(50.0, float(jaw_y)),
        jaw_centers_px=((20.0, 50.0), (80.0, 50.0)),
        jaw_span_px=60.0,
        source_local_white_pixels=white,
        source_local_area_pixels=1000,
        support_center_px=tuple(float(v) for v in support),
        support_scale_px=50.0,
        right_q_physical_rad=(0.0,) * 6,
        right_ee_xyz_m=tuple(float(v) for v in xyz),
        gripper_open_ratio=float(aperture),
    )


def _settings():
    return {
        "maximum_target_perpendicular_span_fraction": 0.25,
        "minimum_measured_lift_m": 0.008,
        "minimum_jaw_motion_spans": 0.05,
        "maximum_support_shift_fraction": 0.10,
        "maximum_source_appearance_ratio": 0.50,
        "minimum_contact_open_ratio": 0.10,
        "maximum_contact_open_ratio": 0.40,
        "maximum_aperture_drift": 0.05,
        "minimum_transport_distance_m": 0.05,
    }


def test_lift_requires_source_clearance_and_persistent_obstruction():
    before = _frame(xyz=(0, 0, 0.7), white=100, jaw_y=50)
    after = _frame(xyz=(0, 0, 0.71), white=35, jaw_y=58)
    assert validate_lift_transition(before, after, _settings())["accepted"]
    empty = _frame(xyz=(0, 0, 0.71), aperture=0.0, white=35, jaw_y=58)
    assert not validate_lift_transition(before, empty, _settings())["accepted"]


def test_transport_requires_motion_and_held_aperture():
    source = _frame(xyz=(0, 0, 0.7), white=100)
    lift = _frame(xyz=(0, 0, 0.71), white=35, jaw_y=58)
    moved = _frame(xyz=(0.08, 0, 0.75), white=30, jaw_y=70)
    assert validate_transport_transition(lift, moved, source, _settings())["accepted"]
    short = _frame(xyz=(0.01, 0, 0.71), white=30, jaw_y=60)
    assert not validate_transport_transition(lift, short, source, _settings())["accepted"]


def test_verified_pasteur_capture_sequence_is_promoted_without_placement():
    if not all(path.exists() for path in (BEFORE, LIFT, TRANSPORT)):
        return
    task = json.loads(TASK.read_text())
    identity = json.loads((ROOT / task["tap_identity"]).read_text())
    result = audit_cap_transfer_captures(
        BEFORE,
        LIFT,
        TRANSPORT,
        target_anchor_uv=identity["tap"]["uv"],
        settings=task["verified_transfer"],
    )
    assert result["accepted"] is True
    assert result["promotion_scope"] == "cap_side_pinch_lift_and_held_transport"
    assert result["placement_promoted"] is False


def test_model_false_positive_override_is_bound_to_exact_route_hash():
    path = [[0.0] * 6, [0.01] * 6]
    waypoints = [{"right_q_physical_rad": [0.0] * 6}]
    digest = waypoint_route_sha256(waypoints)
    accepted = validate_hardware_route_replay(
        path,
        lower=[-1.0] * 6,
        upper=[1.0] * 6,
        model_collision_audit={"accepted": False},
        hardware_evidence_audit={"accepted": True},
        actual_waypoints_sha256=digest,
        expected_waypoints_sha256=digest,
        maximum_route_joint_step_rad=0.02,
        allow_model_false_positive=True,
    )
    assert accepted["accepted"] is True
    assert accepted["model_false_positive_override"] is True
    rejected = validate_hardware_route_replay(
        path,
        lower=[-1.0] * 6,
        upper=[1.0] * 6,
        model_collision_audit={"accepted": False},
        hardware_evidence_audit={"accepted": True},
        actual_waypoints_sha256="changed",
        expected_waypoints_sha256=digest,
        maximum_route_joint_step_rad=0.02,
        allow_model_false_positive=True,
    )
    assert rejected["accepted"] is False
