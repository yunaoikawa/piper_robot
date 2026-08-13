from rollout.episode import EpisodeManager


class RPC:
    def __init__(self):
        self.calls = []

    def home_left_arm(self):
        self.calls.append("left")

    def home_right_arm(self):
        self.calls.append("right")


def test_end_episode_can_defer_home_to_pressure_guarded_orchestrator():
    rpc = RPC()
    manager = EpisodeManager(robot_rpc=rpc)
    manager.start_episode()
    manager.end_episode(reason="cycle_transition", home_after=False)
    assert rpc.calls == []
    assert manager.is_active() is False
