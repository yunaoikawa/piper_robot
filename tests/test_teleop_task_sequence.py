import pytest

from teleop_collect_example import (
    STEPS,
    advance_step_index,
    resolve_start_step,
    resolve_task_sequence,
)


def test_resolve_three_task_sequence_preserves_requested_order():
    tasks = resolve_task_sequence("lid_open,petri2incubator,lid_close")

    assert tasks == ["lid_open", "petri2incubator", "lid_close"]
    assert resolve_start_step("petri2incubator", tasks) == 1


def test_sequence_rejects_unknown_or_duplicate_tasks():
    with pytest.raises(SystemExit, match="Unknown task"):
        resolve_task_sequence("lid_open,not_a_task")
    with pytest.raises(SystemExit, match="duplicate"):
        resolve_task_sequence("lid_open,lid_open")


def test_loop_sequence_wraps_after_last_task():
    assert advance_step_index(
        2, 3, repeat_step=False, loop_sequence=True
    ) == (0, False)


def test_non_loop_sequence_finishes_and_repeat_step_stays_put():
    assert advance_step_index(
        2, 3, repeat_step=False, loop_sequence=False
    ) == (3, True)
    assert advance_step_index(
        1, 3, repeat_step=True, loop_sequence=False
    ) == (1, False)


def test_default_sequence_remains_all_tasks():
    assert resolve_task_sequence(None) == STEPS
