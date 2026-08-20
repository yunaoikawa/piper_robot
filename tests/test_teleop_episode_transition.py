from contextlib import nullcontext
from types import SimpleNamespace

from teleop_collect_example import MinimalTeleopCollector


class _RecordingRobot:
    def __init__(self):
        self.calls = []

    def home_left_arm(self):
        self.calls.append("home_left_arm")

    def home_right_arm(self):
        self.calls.append("home_right_arm")


def _collector(*, home_after_episode):
    collector = object.__new__(MinimalTeleopCollector)
    collector.args = SimpleNamespace(home_after_episode=home_after_episode)
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
