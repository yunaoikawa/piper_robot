#!/usr/bin/env python3

from pathlib import Path
import sys

import mink
import numpy as np
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.calibration_keyboard_jog import (
    CalibrationJogController,
    CalibrationJogStop,
    load_torque_thresholds,
)
from rollout.piper_realtime_motion import PiperRealtimeMotionPreparation
from robot.cone_e import (
    LEFT_WORKSPACE_MAX,
    LEFT_WORKSPACE_MIN,
    RIGHT_WORKSPACE_MAX,
    RIGHT_WORKSPACE_MIN,
    clamp_ee_target,
)


class FakeRPC:
    def __init__(self):
        self.pose = {
            "left": mink.SE3(np.array([1.0, 0.0, 0.0, 0.0, 0.20, 0.10, 0.70])),
            "right": mink.SE3(np.array([1.0, 0.0, 0.0, 0.0, 0.30, 0.10, 0.70])),
        }
        self.qpos = {"left": np.zeros(6), "right": np.zeros(6)}
        self.torque = {"left": np.zeros(6), "right": np.zeros(6)}
        self.targets = []
        self.grippers = []
        self.gains = []
        self.torque_reads = {"left": 0, "right": 0}
        self.trip_after_reads = None

    def _get_pose(self, arm):
        return self.pose[arm]

    def _set(self, arm, value, **kwargs):
        self.targets.append((arm, value, kwargs))
        self.pose[arm] = value
        self.qpos[arm] = self.qpos[arm] + 0.001
        return True

    def get_left_ee_pose(self):
        return self._get_pose("left")

    def get_right_ee_pose(self):
        return self._get_pose("right")

    def set_left_ee_target(self, value, **kwargs):
        return self._set("left", value, **kwargs)

    def set_right_ee_target(self, value, **kwargs):
        return self._set("right", value, **kwargs)

    def set_left_joint_target(self, value, **kwargs):
        self.targets.append(("left_joint", np.asarray(value).copy(), kwargs))
        self.qpos["left"] = np.asarray(value).copy()

    def set_right_joint_target(self, value, **kwargs):
        self.targets.append(("right_joint", np.asarray(value).copy(), kwargs))
        self.qpos["right"] = np.asarray(value).copy()

    def get_left_joint_positions(self):
        return self.qpos["left"]

    def get_right_joint_positions(self):
        return self.qpos["right"]

    def get_left_joint_torque(self):
        self.torque_reads["left"] += 1
        if (
            self.trip_after_reads is not None
            and self.torque_reads["left"] >= self.trip_after_reads
        ):
            return np.full(6, 2.0)
        return self.torque["left"]

    def get_right_joint_torque(self):
        self.torque_reads["right"] += 1
        return self.torque["right"]

    def open_left_gripper(self):
        self.grippers.append(("left", "open"))

    def close_left_gripper(self):
        self.grippers.append(("left", "close"))

    def open_right_gripper(self):
        self.grippers.append(("right", "open"))

    def close_right_gripper(self):
        self.grippers.append(("right", "close"))

    def set_left_gain(self, kp, kd):
        self.gains.append(("left", np.asarray(kp).copy(), np.asarray(kd).copy()))

    def set_right_gain(self, kp, kd):
        self.gains.append(("right", np.asarray(kp).copy(), np.asarray(kd).copy()))


def _controller(rpc, **kwargs):
    clock_value = [0.0]

    def clock():
        return clock_value[0]

    def sleep(duration):
        clock_value[0] += max(duration, 1.0 / 30.0)

    return CalibrationJogController(
        rpc,
        torque_thresholds={
            "left": np.ones(6),
            "right": np.ones(6),
        },
        torque_consecutive_samples=2,
        monitor_time_s=0.1,
        preview_time_s=0.05,
        clock=clock,
        sleep=sleep,
        **kwargs,
    )


class CalibrationKeyboardJogTest(unittest.TestCase):
    def test_realtime_preparation_refreshes_mode_ramps_and_restores_hold(self):
        rpc = FakeRPC()
        clock_value = [0.0]
        refreshed = []

        def clock():
            return clock_value[0]

        def sleep(duration):
            clock_value[0] += duration

        preparation = PiperRealtimeMotionPreparation(
            rpc,
            "right",
            mode_refresher=refreshed.append,
            clock=clock,
            sleep=sleep,
        )
        preparation.prepare(lambda: None)
        preparation.finish()
        self.assertEqual(refreshed, ["can_right", "can_right"])
        self.assertGreaterEqual(len(rpc.gains), 12)
        self.assertTrue(np.allclose(rpc.gains[-1][1], np.full(6, 2.5)))
        self.assertEqual(rpc.targets[-1][0], "right_joint")

    def test_cone_e_workspace_y_bounds_are_mirrored_per_arm(self):
        self.assertAlmostEqual(
            LEFT_WORKSPACE_MIN[1], -RIGHT_WORKSPACE_MAX[1]
        )
        self.assertAlmostEqual(
            LEFT_WORKSPACE_MAX[1], -RIGHT_WORKSPACE_MIN[1]
        )
        left_pose = mink.SE3(
            np.array([1.0, 0.0, 0.0, 0.0, 0.29, -0.34, 0.70])
        )
        clamped = clamp_ee_target(
            left_pose,
            workspace_min=LEFT_WORKSPACE_MIN,
            workspace_max=LEFT_WORKSPACE_MAX,
            arm="left",
        )
        self.assertTrue(np.allclose(clamped.translation(), left_pose.translation()))

    def test_staging_and_arm_selection_send_no_command_until_confirm(self):
        rpc = FakeRPC()
        controller = _controller(rpc)
        controller.enable()
        controller.select_arm("right")
        pending = controller.propose("d")
        self.assertEqual(pending.delta_xyz_m, (0.005, 0.0, 0.0))
        self.assertEqual(rpc.targets, [])
        controller.confirm()
        self.assertGreaterEqual(len(rpc.targets), 30)
        self.assertEqual(rpc.targets[0][0], "right")
        self.assertTrue(np.isclose(rpc.pose["right"].translation()[0], 0.305))

    def test_unenabled_controller_and_oversized_steps_fail_closed(self):
        rpc = FakeRPC()
        controller = _controller(rpc)
        controller.propose("w")
        with self.assertRaises(CalibrationJogStop):
            controller.confirm()
        with self.assertRaises(ValueError):
            _controller(rpc, step_m=0.011, maximum_step_m=0.011)

    def test_precommand_torque_trip_latches_without_sending_target(self):
        rpc = FakeRPC()
        controller = _controller(rpc)
        controller.enable()
        controller.propose("r")
        rpc.torque["left"][:] = 2.0
        with self.assertRaises(CalibrationJogStop):
            controller.confirm()
        self.assertIsNotNone(controller.latched_stop)
        self.assertEqual(rpc.targets, [])
        with self.assertRaises(CalibrationJogStop):
            controller.propose("a")

    def test_sustained_postcommand_torque_trip_sends_joint_hold(self):
        rpc = FakeRPC()
        controller = _controller(rpc)
        controller.enable()
        rpc.trip_after_reads = rpc.torque_reads["left"] + 3
        controller.propose("r")
        with self.assertRaises(CalibrationJogStop):
            controller.confirm()
        self.assertEqual(rpc.targets[0][0], "left")
        self.assertEqual(rpc.targets[-1][0], "left_joint")
        self.assertEqual(controller.latched_stop["arm"], "left")

    def test_gripper_action_is_also_confirmed(self):
        rpc = FakeRPC()
        controller = _controller(rpc)
        controller.enable()
        controller.propose("o")
        self.assertEqual(rpc.grippers, [])
        controller.confirm()
        self.assertEqual(rpc.grippers, [("left", "open")])

    def test_joint_jog_is_bounded_slow_and_hold_uses_joint_target(self):
        rpc = FakeRPC()
        controller = _controller(rpc)
        controller.enable()
        controller.select_joint(2)
        pending = controller.propose("+")
        self.assertEqual(pending.joint_index, 2)
        self.assertEqual(pending.joint_delta_rad, 0.005)
        controller.confirm()
        self.assertTrue(np.isclose(rpc.qpos["left"][2], 0.005))
        joint_targets = [item for item in rpc.targets if item[0] == "left_joint"]
        self.assertGreaterEqual(len(joint_targets), 30)
        self.assertAlmostEqual(joint_targets[-1][2]["preview_time"], 0.05)
        joint_values = np.asarray([item[1][2] for item in joint_targets])
        self.assertTrue(np.all(np.diff(joint_values) > 0.0))
        controller.hold()
        self.assertEqual(rpc.targets[-2][0], "left_joint")
        self.assertEqual(rpc.targets[-1][0], "right_joint")

    def test_joint_jog_stops_when_measured_target_does_not_follow(self):
        rpc = FakeRPC()

        def ignore_joint_target(value, **kwargs):
            rpc.targets.append(("left_joint", np.asarray(value).copy(), kwargs))

        rpc.set_left_joint_target = ignore_joint_target
        controller = _controller(rpc)
        controller.enable()
        controller.propose("+")
        with self.assertRaises(CalibrationJogStop):
            controller.confirm()
        self.assertEqual(rpc.targets[-1][0], "left_joint")

    def test_cartesian_target_outside_server_workspace_is_rejected(self):
        rpc = FakeRPC()
        rpc.pose["left"] = mink.SE3(
            np.array([1.0, 0.0, 0.0, 0.0, 0.20, -0.34, 0.70])
        )
        controller = _controller(
            rpc,
            workspace_min=np.array([-0.05, -0.176, 0.549]),
            workspace_max=np.array([0.59, 0.437, 1.102]),
        )
        controller.enable()
        controller.propose("d")
        with self.assertRaises(CalibrationJogStop):
            controller.confirm()
        self.assertEqual(rpc.targets, [])

    def test_cartesian_jog_stops_when_measured_target_does_not_follow(self):
        rpc = FakeRPC()

        def ignore_ee_target(value, **kwargs):
            rpc.targets.append(("right", value, kwargs))
            return True

        rpc.set_right_ee_target = ignore_ee_target
        controller = _controller(rpc)
        controller.enable()
        controller.select_arm("right")
        controller.propose("r")
        with self.assertRaises(CalibrationJogStop):
            controller.confirm()
        self.assertEqual(rpc.targets[-1][0], "right_joint")

    def test_torque_loader_requires_explicit_left_fallback(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "torque.json"
            path.write_text(
                '{"method":"test","consecutive_samples":3,'
                '"thresholds":{"right":[1,2,3,4,5,6]}}'
            )
            with self.assertRaises(ValueError):
                load_torque_thresholds(path)
            thresholds, provenance = load_torque_thresholds(
                path, allow_symmetric_left_fallback=True
            )
            self.assertTrue(np.allclose(thresholds["left"], thresholds["right"]))
            self.assertEqual(provenance["fallback"]["arm"], "left")


if __name__ == "__main__":
    unittest.main()
