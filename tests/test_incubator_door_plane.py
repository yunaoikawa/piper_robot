import numpy as np

from rollout.incubator_door_plane import fit_vertical_plane, wrap_degrees


SETTINGS = {
    "minimum_candidate_points": 100,
    "ransac_iterations": 500,
    "maximum_abs_normal_z": 0.2,
    "ransac_distance_m": 0.006,
    "minimum_inlier_points": 80,
}


def test_vertical_plane_yaw_rejects_clutter():
    rng = np.random.default_rng(4)
    yaw = np.deg2rad(-7.0)
    normal = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
    tangent = np.asarray([-normal[1], normal[0], 0.0])
    plane = (
        np.asarray([0.3, 0.1, 0.8])
        + rng.uniform(-0.2, 0.2, (800, 1)) * tangent
        + rng.uniform(-0.2, 0.2, (800, 1)) * np.asarray([0, 0, 1])
        + rng.normal(0, 0.001, (800, 1)) * normal
    )
    clutter = rng.uniform(-0.5, 0.5, (200, 3))
    report = fit_vertical_plane(np.vstack([plane, clutter]), SETTINGS, seed=2)
    assert abs(wrap_degrees(report["normal_yaw_deg"] + 7.0)) < 0.5
    assert report["inlier_count"] > 750


def test_wrap_degrees():
    assert wrap_degrees(181.0) == -179.0
