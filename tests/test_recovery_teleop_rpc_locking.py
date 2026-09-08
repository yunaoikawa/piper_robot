"""Regression checks for recovery teleop's shared robot RPC socket."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from teleop_collect_example import MinimalTeleopCollector


class ObservedLock:
    def __init__(self):
        self.held = False

    def __enter__(self):
        if self.held:
            raise AssertionError("test lock is not re-entrant")
        self.held = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.held = False


class GuardThatRequiresLock:
    def __init__(self, lock):
        self.lock = lock
        self.reset_arms = []

    def reset(self, arm):
        if not self.lock.held:
            raise AssertionError("torque baseline RPC ran outside robot_rpc_lock")
        self.reset_arms.append(arm)


class RecoveryTeleopRPCLockingTest(unittest.TestCase):
    def test_engagement_torque_reset_uses_shared_rpc_lock(self):
        collector = object.__new__(MinimalTeleopCollector)
        collector.robot_rpc_lock = ObservedLock()
        collector.recovery_torque_guard = GuardThatRequiresLock(
            collector.robot_rpc_lock
        )

        collector._reset_recovery_torque_guard("left")
        collector._reset_recovery_torque_guard("right")

        self.assertEqual(
            collector.recovery_torque_guard.reset_arms,
            ["left", "right"],
        )
        self.assertFalse(collector.robot_rpc_lock.held)


if __name__ == "__main__":
    unittest.main()
