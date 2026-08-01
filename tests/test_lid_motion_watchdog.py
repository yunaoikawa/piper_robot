import numpy as np
import pytest

from rollout.lid_motion_watchdog import (
    AsyncMotionWatchdogWorker,
    DualCameraLidMotionGuard,
    RobustLidMotionWatchdog,
)


def _ready(scale=1.0):
    watchdog = RobustLidMotionWatchdog(minimum_baseline_samples=5)
    for offset in ((0, 0), (0.2, -0.1), (-0.1, 0.1), (0.1, 0), (0, -0.2)):
        watchdog.add_baseline(np.asarray(offset) * scale, object_diameter=100 * scale)
    return watchdog


def test_watchdog_is_resolution_scale_invariant_and_rejects_two_frames():
    small = _ready(1.0)
    large = _ready(4.0)
    assert not small.observe((6, 0), object_diameter=100).triggered
    assert not large.observe((24, 0), object_diameter=400).triggered
    assert small.observe((6, 0), object_diameter=100).triggered
    assert large.observe((24, 0), object_diameter=400).triggered
    assert np.isclose(
        small.latest().displacement_scale, large.latest().displacement_scale
    )


def test_single_spike_does_not_abort_and_stationary_frame_resets_strikes():
    watchdog = _ready()
    assert not watchdog.observe((10, 0), object_diameter=100).triggered
    state = watchdog.observe((0, 0), object_diameter=100)
    assert not state.triggered
    assert state.consecutive_exceedances == 0


def test_dual_camera_only_fails_visibility_when_both_are_missing():
    head = _ready()
    wrist = _ready()
    head.missing()
    wrist.observe((0, 0), object_diameter=100)
    assert DualCameraLidMotionGuard(head, wrist).require_motion_safe()
    wrist.missing()
    with pytest.raises(RuntimeError, match="both"):
        DualCameraLidMotionGuard(head, wrist).require_motion_safe()


def test_async_worker_builds_baseline_before_motion_observations():
    values = iter(
        [((0, 0), 100), ((0.1, 0), 100), ((0, 0.1), 100), ((6, 0), 100)]
    )
    watchdog = RobustLidMotionWatchdog(
        minimum_baseline_samples=3,
        consecutive_exceedances=1,
    )
    worker = AsyncMotionWatchdogWorker(watchdog, lambda: next(values))
    assert not worker.sample_once().ready
    assert not worker.sample_once().ready
    assert worker.sample_once().ready
    assert worker.sample_once().triggered
