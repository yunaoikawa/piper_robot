from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.recovery_teleop_safety import (
    RecoveryTorqueGuard,
    extend_fallback_threshold_for_stationary_pose,
)


class FakeRPC:
    def __init__(self):
        self.qpos = {
            "left": np.arange(6, dtype=float),
            "right": np.arange(6, dtype=float) + 10.0,
        }
        self.torque = {
            "left": np.zeros(6),
            "right": np.zeros(6),
        }
        self.holds = []

    def __getattr__(self, name):
        for arm in ("left", "right"):
            if name == f"get_{arm}_joint_torque":
                return lambda arm=arm: self.torque[arm]
            if name == f"get_{arm}_joint_positions":
                return lambda arm=arm: self.qpos[arm]
            if name == f"set_{arm}_joint_target":
                return lambda value, arm=arm, **kwargs: self.holds.append(
                    (arm, np.asarray(value).copy(), kwargs)
                )
        raise AttributeError(name)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def guard(rpc, **kwargs):
    return RecoveryTorqueGuard(
        rpc,
        {
            "left": np.ones(6),
            "right": np.ones(6),
        },
        consecutive_samples=2,
        **kwargs,
    )


class RecoveryTorqueGuardTest(unittest.TestCase):
    def test_stationary_extension_only_raises_needed_joints(self):
        rpc = FakeRPC()
        rpc.torque["left"] = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 0.1])
        thresholds = {
            "left": np.array([0.3, 0.5, 0.7, 0.7, 1.5, 0.2]),
            "right": np.ones(6),
        }

        record = extend_fallback_threshold_for_stationary_pose(
            rpc,
            thresholds,
            arm="left",
            sample_count=5,
            sample_interval_s=0.0,
            margin=1.2,
        )

        np.testing.assert_allclose(
            thresholds["left"],
            [0.3, 0.5, 0.72, 0.96, 1.5, 0.2],
        )
        self.assertEqual(record["changed_joints"], [2, 3])

    def test_sustained_excess_latches_only_affected_arm_and_holds(self):
        rpc = FakeRPC()
        clock = FakeClock()
        value = guard(
            rpc,
            clock=clock,
            residual_duration_s=0.1,
            baseline_slew_nm_per_s=0.1,
        )
        rpc.torque["left"][2] = 0.6
        clock.advance(0.05)
        self.assertTrue(value.check("left"))
        clock.advance(0.05)
        self.assertTrue(value.check("left"))
        clock.advance(0.05)
        self.assertFalse(value.check("left"))
        self.assertIn("left", value.latched)
        self.assertNotIn("right", value.latched)
        self.assertEqual(rpc.holds[-1][0], "left")
        self.assertTrue(np.allclose(rpc.holds[-1][1], rpc.qpos["left"]))
        self.assertEqual(rpc.holds[-1][2]["preview_time"], 0.05)

    def test_engagement_reset_clears_strikes_but_not_a_latched_stop(self):
        rpc = FakeRPC()
        value = guard(rpc)
        rpc.torque["right"][:] = 2.1
        self.assertTrue(value.check("right"))
        value.reset("right")
        self.assertTrue(value.check("right"))
        self.assertFalse(value.check("right"))
        value.reset("right")
        self.assertFalse(value.check("right"))

    def test_gradual_pose_dependent_load_is_tracked(self):
        rpc = FakeRPC()
        clock = FakeClock()
        value = guard(
            rpc,
            clock=clock,
            residual_duration_s=0.1,
            baseline_slew_nm_per_s=0.5,
        )
        for sample in range(1, 91):
            clock.advance(1.0 / 30.0)
            rpc.torque["left"][3] = sample / 90.0
            self.assertTrue(value.check("left"))

    def test_observer_only_warns_without_holding_or_latching(self):
        rpc = FakeRPC()
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "guard.jsonl"
            value = guard(
                rpc,
                enforce=False,
                clock=clock,
                audit_path=audit,
                residual_duration_s=0.1,
                baseline_slew_nm_per_s=0.1,
            )
            rpc.torque["left"][2] = 0.8
            for _ in range(4):
                clock.advance(0.05)
                self.assertTrue(value.check("left"))
            text = audit.read_text()
        self.assertIn("torque_warning", text)
        self.assertNotIn("left", value.latched)
        self.assertEqual(rpc.holds, [])

    def test_invalid_torque_fails_closed_and_writes_audit(self):
        rpc = FakeRPC()
        rpc.torque["left"] = np.array([np.nan] * 6)
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "guard.jsonl"
            value = guard(
                rpc,
                audit_path=audit,
                provenance={"fallback": {"arm": "left", "source_arm": "right"}},
            )
            self.assertFalse(value.check("left"))
            text = audit.read_text()
        self.assertIn("torque_guard_initialized", text)
        self.assertIn("torque_stop", text)
        self.assertTrue(value.latched["left"]["hold_sent"])


if __name__ == "__main__":
    unittest.main()
