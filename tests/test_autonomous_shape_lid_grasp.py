from __future__ import annotations

from robot.arm.home import physical_home_q
from src.run_autonomous_shape_lid_grasp import (
    TRAJECTORY_SCHEMA,
    audit_physical_right_level,
    validate_demo_free_trajectory,
)
from rollout.gripper_level import JawLevelReference


PRODUCTION_MODEL = (
    "robot/cone-e-description/robot-welded-base-and-lift.mjcf"
)


def _trajectory():
    q = physical_home_q("right").tolist()
    stages = (
        "home",
        "depart_up",
        "hover_xy",
        "descend",
        "insert",
        "preclose_observe",
        "close",
        "verification_lift",
        "hold",
    )
    return {
        "schema": TRAJECTORY_SCHEMA,
        "commands_sent": False,
        "simulation_only": True,
        "closure_policy": {
            "mode": "close_until_obstructed",
            "demonstration_used": False,
            "physical_gripper_target_open_ratio": 0.0,
        },
        "knots": [
            {
                "stage": stage,
                "right_q_physical_rad": q,
                "right_gripper_open_ratio": (
                    0.0
                    if stage in {"close", "verification_lift", "hold"}
                    else 1.0
                ),
                "minimum_duration_s": 0.1,
            }
            for stage in stages
        ],
    }


def test_demo_free_trajectory_is_accepted():
    validate_demo_free_trajectory(_trajectory())


def test_demonstration_closure_is_rejected():
    payload = _trajectory()
    payload["closure_demonstration"] = {"right_gripper_open_ratio": 0.58}
    try:
        validate_demo_free_trajectory(payload)
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("demonstration-derived plan was accepted")


def test_physical_level_audit_models_the_commanded_low_pose_projection():
    payload = _trajectory()
    report = audit_physical_right_level(
        payload,
        production_model=PRODUCTION_MODEL,
        reference=JawLevelReference(),
    )
    assert report["accepted"]
    bad = [
        0.4228234589,
        1.0378774405,
        -0.1852143407,
        0.8782147765,
        -0.9789202809,
        1.9821355343,
    ]
    for knot in payload["knots"]:
        if knot["stage"] == "descend":
            knot["right_q_physical_rad"] = bad
    corrected = audit_physical_right_level(
        payload,
        production_model=PRODUCTION_MODEL,
        reference=JawLevelReference(),
    )
    assert corrected["accepted"]
    descend = next(
        item for item in corrected["stages"] if item["stage"] == "descend"
    )
    assert descend["maximum_combined_tilt_deg"] < 1e-5
    assert descend["maximum_tip_height_difference_m"] < 1e-9
