import numpy as np

from rollout.gripper_level import JawLevelReference
from rollout.pose_locked_descent import PhysicalRightPoseLockedPlanner


def test_low_out_of_range_wrist_rebranches_and_descends_level():
    planner = PhysicalRightPoseLockedPlanner(
        "robot/cone-e-description/robot-welded-base-and-lift.mjcf",
        JawLevelReference(
            approach_axis_ee=(0.998811503, 0.048739928, 0.0),
        ),
    )
    start = np.asarray([
        0.49553388, 1.61828673, -0.31408945,
        0.54881877, -1.31109130, 2.26290655,
    ])
    plan = planner.plan(start)
    lift = np.asarray(plan.lift_q_path)
    descent = np.asarray(plan.descent_q_path)
    assert lift[-1, 4] > -1.20
    assert descent[-1, 4] >= -1.195
    assert plan.hover_assessment["tip_height_difference_m"] < 0.0017
    assert plan.final_assessment["tip_height_difference_m"] < 0.0017
    assert plan.final_tip_pitch_deg < 0.0
    np.testing.assert_allclose(
        plan.predicted_final_delta_xyz_m, [0.0, 0.0, 0.0], atol=0.0016
    )


def test_plan_is_deterministic():
    planner = PhysicalRightPoseLockedPlanner(
        "robot/cone-e-description/robot-welded-base-and-lift.mjcf",
        JawLevelReference(
            approach_axis_ee=(0.998811503, 0.048739928, 0.0),
        ),
    )
    start = [0.49553388, 1.61828673, -0.31408945,
             0.54881877, -1.31109130, 2.26290655]
    assert planner.plan(start) == planner.plan(start)
