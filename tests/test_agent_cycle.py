import numpy as np
import pytest

from rollout.agent_collection import InterventionState
from rollout.agent_cycle import AlternatingAgentCycle
from rollout.controller import PolicyController


def cycle():
    return AlternatingAgentCycle(
        ("lid_close", "lid_open"), initial_task="lid_open",
        camera_failure_grace_s=.5, inference_failure_grace_s=1,
        maximum_episode_s=5,
        bias_retry_offsets_m={
            "lid_open": ([0, 0, 0], [0, .005, 0], [0, .005, -.005]),
            "lid_close": ([0, 0, 0], [0, .005, -.005]),
        },
    )


def test_release_completes_then_alternates_forever():
    state = cycle()
    state.begin_episode(safety_rejected_count=2, now=0)
    decision = state.evaluate(
        terminal_release=True,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=2, now=.2,
    )
    assert decision.action == "complete_success"
    assert decision.task == "lid_open"
    assert decision.next_task == "lid_close"
    assert state.advance_after_success() == "lid_close"
    state.begin_episode(safety_rejected_count=2, now=1)
    decision = state.evaluate(
        terminal_release=True,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=2, now=2,
    )
    assert decision.next_task == "lid_open"
    assert state.advance_after_success() == "lid_open"


def test_transient_camera_gap_does_not_stop_but_sustained_gap_does():
    state = cycle()
    state.begin_episode(safety_rejected_count=0, now=0)
    assert state.evaluate(
        terminal_release=False,
        camera_ready={"head": False, "right": True},
        safety_rejected_count=0, now=.1,
    ) is None
    decision = state.evaluate(
        terminal_release=False,
        camera_ready={"head": False, "right": True},
        safety_rejected_count=0, now=.7,
    )
    assert decision.action == "stop_failure"
    assert decision.reason == "camera_missing:head"
    assert state.enabled is False


def test_camera_can_recover_inside_grace_period():
    state = cycle()
    state.begin_episode(safety_rejected_count=0, now=0)
    assert state.evaluate(
        terminal_release=False,
        camera_ready={"head": False, "right": True},
        safety_rejected_count=0, now=.1,
    ) is None
    assert state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=.4,
    ) is None
    assert state.evaluate(
        terminal_release=False,
        camera_ready={"head": False, "right": True},
        safety_rejected_count=0, now=.8,
    ) is None


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"safety_rejected_count": 4}, "safety_rejected"),
        ({"pressure_stop": True}, "pressure_stop"),
        ({"explicit_failure": "drop"}, "drop"),
    ],
)
def test_physical_or_explicit_failure_stops_without_advancing(kwargs, reason):
    state = cycle()
    state.begin_episode(safety_rejected_count=3, now=0)
    values = dict(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=3,
        now=.1,
    )
    values.update(kwargs)
    decision = state.evaluate(**values)
    assert decision.action == "stop_failure"
    assert decision.reason == reason
    assert state.current_task == "lid_open"


def test_drop_evidence_beats_stale_terminal_release_and_never_retries():
    state = cycle()
    state.begin_episode(safety_rejected_count=0, now=0)
    decision = state.evaluate(
        terminal_release=True,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0,
        explicit_failure="drop",
        now=.1,
    )
    assert decision.action == "stop_failure"
    assert decision.reason == "drop"
    assert state.enabled is False
    assert state.current_task == "lid_open"


def test_inference_and_episode_deadlines_fail_closed():
    state = cycle()
    state.begin_episode(safety_rejected_count=0, now=0)
    state.note_inference(False, now=.1)
    assert state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=.9,
    ) is None
    assert state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=1.2,
    ).reason == "inference_unavailable"

    state = cycle()
    state.begin_episode(safety_rejected_count=0, now=0)
    decision = state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=5.1,
    )
    assert decision.action == "retry_failure"
    assert decision.reason == "episode_timeout"
    assert decision.task == decision.next_task == "lid_open"
    assert decision.bias_attempt_index == 1
    assert decision.bias_offset_m == pytest.approx((0, .005, 0))


def test_timeout_advances_finite_bias_schedule_without_changing_task():
    state = cycle()
    state.begin_episode(safety_rejected_count=0, now=0)
    first = state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=5.1,
    )
    assert state.advance_after_retry(first) == 1
    assert state.current_task == "lid_open"
    assert state.current_bias_offset_m == pytest.approx((0, .005, 0))

    state.begin_episode(safety_rejected_count=0, now=6)
    second = state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=12,
    )
    assert state.advance_after_retry(second) == 2
    assert state.current_bias_offset_m == pytest.approx((0, .005, -.005))

    state.begin_episode(safety_rejected_count=0, now=13)
    exhausted = state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=19,
    )
    assert exhausted.action == "stop_retry_exhausted"
    assert state.enabled is True  # save + pressure-home must happen first
    stopped = state.stop_after_retry_exhausted(exhausted)
    assert stopped.action == "stop_failure"
    assert stopped.reason == "bias_schedule_exhausted:episode_timeout"
    assert state.enabled is False


def test_grasp_miss_retries_but_pressure_and_drop_still_fail_closed():
    state = cycle()
    state.begin_episode(safety_rejected_count=0, now=0)
    retry = state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, explicit_failure="grasp_miss", now=.1,
    )
    assert retry.action == "retry_failure"
    assert retry.bias_attempt_index == 1

    for explicit in ("drop", "jam"):
        state = cycle()
        state.begin_episode(safety_rejected_count=0, now=0)
        stopped = state.evaluate(
            terminal_release=False,
            camera_ready={"head": True, "right": True},
            safety_rejected_count=0, explicit_failure=explicit, now=.1,
        )
        assert stopped.action == "stop_failure"
        assert stopped.reason == explicit


def test_success_resets_attempt_and_alternates_task():
    state = cycle()
    state.begin_episode(safety_rejected_count=0, now=0)
    retry = state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, explicit_failure="grasp_miss", now=.1,
    )
    state.advance_after_retry(retry)
    state.begin_episode(safety_rejected_count=0, now=1)
    success = state.evaluate(
        terminal_release=True,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=2,
    )
    assert success.action == "complete_success"
    assert state.advance_after_success() == "lid_close"
    assert state.bias_attempt_index["lid_open"] == 0
    assert state.current_bias_attempt_index == 0


@pytest.mark.parametrize(
    "schedule",
    [
        {"unknown": [[0, 0, 0]]},
        {"lid_open": []},
        {"lid_open": [[0, 0]]},
        {"lid_open": [[0, .001, 0]]},
        {"lid_open": [[0, 0, 0], [0, float("nan"), 0]]},
    ],
)
def test_invalid_bias_retry_schedule_is_rejected(schedule):
    with pytest.raises(ValueError):
        AlternatingAgentCycle(
            ("lid_open", "lid_close"), initial_task="lid_open",
            bias_retry_offsets_m=schedule,
        )


class _Recorder:
    def __init__(self, order):
        self.order = order

    def log_event(self, event, **payload):
        self.order.append(("log", event, payload))

    def finalize(self, outcome, *, reason=None):
        self.order.append(("finalize", outcome, reason))
        return "/tmp/failure-episode"


class _EpisodeManager:
    def __init__(self, order):
        self.order = order

    def end_episode(self, *, reason, home_after):
        self.order.append(("end", reason, home_after))

    def clear_action_queue(self):
        self.order.append(("clear",))


def _retry_controller(state):
    order = []
    controller = PolicyController.__new__(PolicyController)
    controller.agent_cycle = state
    controller._agent_cycle_transition_active = False
    controller._agent_task_bias = {
        "lid_open": np.array([.01, .02, -.03]),
        "lid_close": np.zeros(3),
    }
    controller.xyz_bias = {
        "left": np.zeros(3), "right": np.array([.01, .02, -.03])
    }
    controller.recorder = _Recorder(order)
    controller.episode_manager = _EpisodeManager(order)
    controller.intervention = InterventionState()
    controller._agent_home_both_arms = lambda: (
        order.append(("pressure_home",)) or {"completed": True}
    )

    def set_bias(arm, value):
        value = np.asarray(value, dtype=float)
        controller.xyz_bias[arm] = value.copy()
        order.append(("set_bias", arm, value.copy()))

    controller.set_bias = set_bias
    controller._start_cycle_episode = lambda task: order.append(("start", task))
    controller._stop_agent_cycle = lambda reason: pytest.fail(
        f"unexpected hard stop: {reason}"
    )
    return controller, order


def test_controller_saves_homes_and_applies_next_bias_before_retry_start():
    state = cycle()
    state.begin_episode(safety_rejected_count=0, now=0)
    decision = state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=5.1,
    )
    controller, order = _retry_controller(state)
    controller._retry_or_exhaust_agent_cycle(decision)

    names = [entry[0] for entry in order]
    assert names.index("finalize") < names.index("pressure_home")
    assert names.index("pressure_home") < names.index("set_bias")
    assert names.index("set_bias") < names.index("start")
    assert ("finalize", "failure", "episode_timeout") in order
    set_entry = next(entry for entry in order if entry[0] == "set_bias")
    assert set_entry[2] == pytest.approx([.01, .025, -.03])
    assert state.current_task == "lid_open"
    assert state.current_bias_attempt_index == 1


def test_controller_saves_and_pressure_homes_before_exhausted_stop():
    state = AlternatingAgentCycle(
        ("lid_open", "lid_close"), initial_task="lid_open",
        maximum_episode_s=1,
        bias_retry_offsets_m={
            "lid_open": [[0, 0, 0]],
            "lid_close": [[0, 0, 0]],
        },
    )
    state.begin_episode(safety_rejected_count=0, now=0)
    decision = state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=2,
    )
    controller, order = _retry_controller(state)
    controller._retry_or_exhaust_agent_cycle(decision)

    names = [entry[0] for entry in order]
    assert names.index("finalize") < names.index("pressure_home")
    assert "set_bias" not in names and "start" not in names
    assert state.enabled is False
    assert state.stop_reason == "bias_schedule_exhausted:episode_timeout"
    assert controller.intervention.mode == "cycle_stopped"
