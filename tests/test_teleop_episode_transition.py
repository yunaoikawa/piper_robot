from contextlib import nullcontext
from pathlib import Path
import re
import threading
from types import SimpleNamespace

import teleop_collect_example as teleop_module
from teleop_collect_example import MinimalTeleopCollector


class _RecordingRobot:
    def __init__(self):
        self.calls = []

    def home_left_arm(self):
        self.calls.append("home_left_arm")

    def home_right_arm(self):
        self.calls.append("home_right_arm")

    def machine_zero_arms(self):
        self.calls.append("machine_zero_arms")


def _collector(*, home_after_episode=False, zero_after_episode=False):
    collector = object.__new__(MinimalTeleopCollector)
    collector.args = SimpleNamespace(
        home_after_episode=home_after_episode,
        zero_after_episode=zero_after_episode,
    )
    collector.robot_rpc_lock = nullcontext()
    collector.robot = _RecordingRobot()
    collector.saved = 0
    collector._end_episode_and_save = lambda: setattr(
        collector, "saved", collector.saved + 1
    )
    return collector


def test_episode_end_holds_final_pose_by_default():
    collector = _collector(home_after_episode=False)

    collector._finish_episode_after_disengage()

    assert collector.robot.calls == []
    assert collector.saved == 1


def test_legacy_home_after_episode_requires_explicit_opt_in():
    collector = _collector(home_after_episode=True)

    collector._finish_episode_after_disengage()

    assert collector.robot.calls == ["home_left_arm", "home_right_arm"]
    assert collector.saved == 1


def test_zero_after_episode_saves_before_machine_zero():
    collector = _collector(zero_after_episode=True)
    events = []
    collector._end_episode_and_save = lambda: events.append("save")
    collector.robot.machine_zero_arms = lambda: events.append("machine_zero_arms")
    collector.robot.home_left_arm = lambda: events.append("home_left_arm")
    collector.robot.home_right_arm = lambda: events.append("home_right_arm")

    collector._finish_episode_after_disengage()

    assert events == [
        "save",
        "machine_zero_arms",
        "home_left_arm",
        "home_right_arm",
    ]


def test_step_episode_filename_is_datetime_only(tmp_path, monkeypatch):
    class _Writers:
        def __init__(self, base_path, fps):
            self.base_path = base_path

        def open(self, _label):
            return None

    monkeypatch.setattr(teleop_module, "VideoWriterSet", _Writers)
    collector = object.__new__(MinimalTeleopCollector)
    collector.mode = "steps"
    collector.steps = ["petri2bench"]
    collector.step_index = 0
    collector.episode_lock = threading.Lock()
    collector.recording_capture_lock = threading.Lock()
    collector.is_recording = False
    collector._step_subdir = lambda _index: Path(tmp_path)

    collector._start_episode()

    stem = Path(collector._current_base_path).name
    assert re.fullmatch(r"\d{8}_\d{6}", stem)
    assert "petri2bench" not in stem
