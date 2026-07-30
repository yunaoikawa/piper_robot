from pathlib import Path
import sys
import unittest

import mink
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot.cone_e import ConeE


class FakeArm:
    def __init__(self):
        self.target = None

    def set_ee_target(self, target, gripper_target, preview_time):
        self.target = target
        return True


class ConeEUnboundedTargetTest(unittest.TestCase):
    def test_demo_workspace_does_not_modify_either_arm_target(self):
        cone = object.__new__(ConeE)
        cone._initialized = True
        cone.left_arm = FakeArm()
        cone.right_arm = FakeArm()
        left = mink.SE3(
            np.array([1.0, 0.0, 0.0, 0.0, 0.20, -0.76, 0.40])
        )
        right = mink.SE3(
            np.array([1.0, 0.0, 0.0, 0.0, 0.70, -0.50, 1.20])
        )
        self.assertTrue(cone.set_left_ee_target(left))
        self.assertTrue(cone.set_right_ee_target(right))
        self.assertTrue(
            np.allclose(cone.left_arm.target.translation(), left.translation())
        )
        self.assertTrue(
            np.allclose(cone.right_arm.target.translation(), right.translation())
        )


if __name__ == "__main__":
    unittest.main()
