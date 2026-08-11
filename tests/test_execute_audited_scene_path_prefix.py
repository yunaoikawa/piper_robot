import numpy as np

from src.execute_audited_scene_path_prefix import _reverse_path_from_nearest


def test_reverse_path_includes_zero_endpoint():
    sparse = np.arange(30, dtype=float).reshape(5, 6)
    measured = sparse[4] + 0.01
    selected, nearest, error = _reverse_path_from_nearest(sparse, measured, 0)
    assert nearest == 4
    assert np.isclose(error, 0.01)
    assert np.array_equal(selected[-1], sparse[0])
    assert len(selected) == 6


def test_reverse_path_includes_nonzero_stop_endpoint():
    sparse = np.arange(36, dtype=float).reshape(6, 6)
    selected, nearest, _ = _reverse_path_from_nearest(sparse, sparse[5], 2)
    assert nearest == 5
    assert np.array_equal(selected[-1], sparse[2])
    assert len(selected) == 5
