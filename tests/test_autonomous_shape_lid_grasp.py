from __future__ import annotations

from robot.arm.home import physical_home_q
from src.run_autonomous_shape_lid_grasp import (
    TRAJECTORY_SCHEMA,
    validate_demo_free_trajectory,
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
