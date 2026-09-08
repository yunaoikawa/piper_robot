#!/usr/bin/env python3

from argparse import Namespace
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.run_semantic_scene_pipeline import (
    _commands,
    _run_stage,
    validate_completed_scene,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src" / "configs" / "pasteur_semantic_scene.json"


def _args(**updates):
    values = {
        "capture": None,
        "multiview_report": "/tmp/multiview_report.json",
        "profile": str(PROFILE),
        "output_dir": "/tmp/output",
        "sam_endpoint": "tcp://127.0.0.1:5562",
        "mask": [],
        "attempt": 1,
        "calibration_report": None,
        "daily_scene": None,
        "resume_confirmed": False,
    }
    values.update(updates)
    return Namespace(**values)


def _scene(with_optimization=True):
    names = [
        "robot",
        "incubator",
        "microscope",
        "culture_media_bottle",
        "petri_dish",
        "petri_lid",
    ]
    objects = [{"semantic_name": name} for name in names]
    if with_optimization:
        objects[1]["semantic_volume_fit"] = {
            "attempted": True,
            "accepted": True,
            "method": "semantic_volume_test",
            "initial": {
                "objective": 0.3,
                "known_free_intrusion_fraction": 0.4,
                "yaw_rad": -0.3,
            },
            "optimized": {
                "objective": 0.05,
                "known_free_intrusion_fraction": 0.01,
                "yaw_rad": 0.2,
            },
            "improvement_fraction": 0.83,
        }
    return {
        "schema": "piper_robot.multiview_completed_scene/v1",
        "objects": objects,
        "mujoco_compile": {
            "ok": True,
            "missing_required_nyu_geoms": [],
            "forbidden_stock_bodies": [],
        },
        "readiness": {
            "display_ready": True,
            "collision_ready": False,
            "motion_ready": False,
        },
    }


def test_existing_report_build_command_contains_no_robot_command():
    commands = _commands(
        _args(),
        Path("/tmp/multiview_report.json"),
        Path("/tmp/output/scene"),
    )
    assert [name for name, _ in commands] == ["complete_semantic_mujoco"]
    command = commands[0][1]
    assert "build_semantic_scene.py" in " ".join(command)
    forbidden = {"home", "move", "grasp", "init"}
    assert not forbidden.intersection(command)


def test_capture_command_precedes_semantic_completion():
    commands = _commands(
        _args(capture="/tmp/capture", multiview_report=None),
        Path("/tmp/output/multiview/multiview_report.json"),
        Path("/tmp/output/scene"),
    )
    assert [name for name, _ in commands] == [
        "reconstruct_multiview",
        "complete_semantic_mujoco",
    ]
    assert "reconstruct_multiview_scene.py" in " ".join(commands[0][1])


def test_validation_requires_volume_optimization_for_eligible_box():
    accepted = validate_completed_scene(_scene(), PROFILE)
    rejected = validate_completed_scene(
        _scene(with_optimization=False), PROFILE
    )
    assert accepted["accepted"]
    assert accepted["semantic_volume_optimization"]["incubator"]["accepted"]
    assert not rejected["accepted"]
    assert (
        "semantic_volume_fit_not_attempted:incubator"
        in rejected["reasons"]
    )


def test_failed_stage_preserves_logs_instead_of_losing_diagnostics():
    with tempfile.TemporaryDirectory() as directory:
        record = _run_stage(
            "expected_failure",
            [
                sys.executable,
                "-c",
                "import sys; print('diagnostic', file=sys.stderr); sys.exit(7)",
            ],
            logs_dir=Path(directory),
        )
        assert record["returncode"] == 7
        assert "diagnostic" in record["stderr_tail"]
        assert Path(record["stderr_log"]).is_file()


if __name__ == "__main__":
    test_existing_report_build_command_contains_no_robot_command()
    test_capture_command_precedes_semantic_completion()
    test_validation_requires_volume_optimization_for_eligible_box()
    test_failed_stage_preserves_logs_instead_of_losing_diagnostics()
    print("semantic scene pipeline runner checks passed")
