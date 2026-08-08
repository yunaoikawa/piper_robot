import numpy as np

from src.export_qwen3d_scene import _stratified_indices


def test_stratified_display_sampling_preserves_small_semantic_regions():
    labels = np.r_[np.full(10000, 2), np.full(20, 7), np.full(11, 8)]
    indices = _stratified_indices(labels, 500)
    selected = labels[indices]
    assert len(indices) == 500
    assert set(np.unique(selected)) == {2, 7, 8}
    assert np.sum(selected == 7) == 20
    assert np.sum(selected == 8) == 11


def test_stratified_sampling_is_deterministic():
    labels = np.repeat(np.arange(6), 1000)
    np.testing.assert_array_equal(
        _stratified_indices(labels, 700),
        _stratified_indices(labels, 700),
    )
