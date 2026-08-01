import json
import time

import pytest

from rollout.gripper_level import JawLevelReference
from rollout.orientation_monitor_policy import (
    CachedOrientationMonitor,
    OrientationMonitoringPolicyStore,
)


def test_policy_only_escalates_for_observed_contact_orientation_failures(tmp_path):
    store = OrientationMonitoringPolicyStore(tmp_path / "policy.json")
    assert store.load().mode == "checkpoint"
    assert store.record_failure("camera_timeout").mode == "checkpoint"
    escalated = store.record_failure("lid_lateral_motion")
    assert escalated.mode == "continuous_cached"
    assert json.loads((tmp_path / "policy.json").read_text())["reason"] == "lid_lateral_motion"


def test_continuous_monitor_consumes_pushed_cache_not_rpc():
    monitor = CachedOrientationMonitor(JawLevelReference())
    # Identity has local-Y horizontal and must be rejected, but no RPC object
    # exists anywhere in this monitor.
    monitor.update((1, 0, 0, 0, 0, 0, 0))
    with pytest.raises(RuntimeError, match="rejected"):
        monitor.require_level()


def test_continuous_monitor_rejects_stale_cache():
    monitor = CachedOrientationMonitor(JawLevelReference())
    monitor.update(
        (0.5, -0.5, -0.5, -0.5, 0, 0, 0),
        observed_at_monotonic_s=time.monotonic() - 1.0,
    )
    with pytest.raises(RuntimeError, match="stale"):
        monitor.require_level(maximum_age_s=0.1)
