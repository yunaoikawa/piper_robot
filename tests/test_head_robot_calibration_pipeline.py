#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_head_robot_calibration_pipeline import run


class HeadRobotCalibrationPipelineTest(unittest.TestCase):
    def test_dry_run_keeps_capture_and_motion_processes_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(
                capture=str(root / "capture"),
                multiview_report=str(root / "multiview.json"),
                profile=str(root / "profile.json"),
                output_dir=str(root / "output"),
                sam_endpoint="tcp://sam:5562",
                motion_config_template=str(root / "motion.json"),
                robot_mask=[],
                daily_scene=None,
                resume_confirmed=False,
                require_collision_ready=False,
                dry_run=True,
            )
            report = run(args)
            self.assertEqual(report["status"], "dry_run")
            self.assertFalse(report["commands_sent"])
            commands = [" ".join(item["command"]) for item in report["commands"]]
            self.assertIn("calibrate_head_robot_from_cad.py", commands[0])
            self.assertIn("register_fixed_head_scene.py", commands[1])
            self.assertIn("--scene-registration-report", commands[2])
            self.assertTrue(all("calibration_keyboard_jog.py" not in item for item in commands))


if __name__ == "__main__":
    unittest.main()
