import pytest

from rollout.agent_cycle import AlternatingAgentCycle


def cycle():
    return AlternatingAgentCycle(
        ("lid_close", "lid_open"), initial_task="lid_open",
        camera_failure_grace_s=.5, inference_failure_grace_s=1,
        maximum_episode_s=5,
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
    assert state.evaluate(
        terminal_release=False,
        camera_ready={"head": True, "right": True},
        safety_rejected_count=0, now=5.1,
    ).reason == "episode_timeout"
