"""Tracked-report plotting checks; no robot, raw frames, or private logs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('seaborn')
import matplotlib.pyplot as plt

path = Path(__file__).resolve().parents[1] / 'docs/plot_door_complexity_seaborn.py'
spec = importlib.util.spec_from_file_location('door_seaborn', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_api_counts_and_pose_metrics_are_not_jittered_or_interpolated():
    data = module.load_data()
    assert data['external_api_count'].tolist() == [30, 30, 57, 57]
    fig, axes = module.plot_api_errors(save=False)
    try:
        assert len(fig.axes) == 2
        assert axes[0].get_ylim() == (10, 20)
        assert axes[1].get_ylim() == (0, 12)
        for ax, metric in zip(axes, ('distance_mm', 'orientation_deg')):
            assert not ax.lines
            expected = data[['external_api_count', metric]].to_numpy()
            actual = np.asarray(ax.collections[0].get_offsets())
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual[2], actual[3])
            assert [t.get_text() for t in ax.texts] == ['D1', 'D2', 'D3 / D4']
    finally:
        plt.close(fig)
