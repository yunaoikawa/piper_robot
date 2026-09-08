import json
from pathlib import Path

import pytest

from rollout.grasp_orchestration import (
    CheckpointStore,
    ControllerLease,
    PrimitiveResult,
    allowed_next_actions,
    validate_seating_preconditions,
)
from src.run_codexless_thin_object_grasp import load_profile


def _result(tmp_path, *, primitive="descend-bottom", accepted=True, sequence=0):
    profile = tmp_path / "profile.json"
    profile.write_text("{}\n")
    return PrimitiveResult(
        run_id="test-run",
        sequence=sequence,
        primitive=primitive,
        accepted=accepted,
        profile_path=str(profile),
        profile_sha256="0" * 64,
        input_state_sha256=None,
        summary="test",
    )


def test_success_and_failure_transitions_are_fail_closed():
    assert allowed_next_actions("descend-bottom", True) == (
        "seat-2mm",
        "recover",
        "stop",
    )
    assert allowed_next_actions("descend-bottom", False) == ("recover", "stop")
    assert "close" not in allowed_next_actions("seat-2mm", False)


def test_checkpoint_decision_is_bound_to_immutable_result(tmp_path):
    store = CheckpointStore(tmp_path / "run")
    path, value = store.publish(_result(tmp_path))
    decision = store.write_decision(
        path, "seat-2mm", reason="normal descent evidence accepted"
    )
    loaded = json.loads(decision.read_text())
    assert loaded["result_state_sha256"] == value["state_sha256"]
    assert store.read_decision(path)["action"] == "seat-2mm"
    with pytest.raises(ValueError, match="not allowed"):
        store.write_decision(path, "close", reason="skip seating")


def test_checkpoint_is_append_only(tmp_path):
    store = CheckpointStore(tmp_path / "run")
    result = _result(tmp_path)
    path, _ = store.publish(result)
    changed = PrimitiveResult(
        **{
            **result.__dict__,
            "summary": "different",
        }
    )
    with pytest.raises(FileExistsError, match="immutable"):
        store.publish(changed)
    assert path.exists()


def test_seating_is_exactly_once_after_accepted_normal_descent(tmp_path):
    predecessor = _result(tmp_path).to_dict()
    assert validate_seating_preconditions(predecessor, already_applied=False) == 0.002
    with pytest.raises(ValueError, match="single-shot"):
        validate_seating_preconditions(predecessor, already_applied=True)
    rejected = _result(tmp_path, accepted=False).to_dict()
    with pytest.raises(ValueError, match="accepted normal descent"):
        validate_seating_preconditions(rejected, already_applied=False)


def test_controller_lease_rejects_a_second_process_owner(tmp_path):
    path = tmp_path / "right.lock"
    first = ControllerLease(path, owner={"run_id": "first"}).acquire()
    try:
        with pytest.raises(RuntimeError, match="already leased"):
            ControllerLease(path, owner={"run_id": "second"}).acquire()
    finally:
        first.release()
    with ControllerLease(path, owner={"run_id": "third"}):
        assert Path(path).exists()


def test_pasteur_profile_configures_exact_single_seating_command():
    root = Path(__file__).resolve().parents[1]
    profile = load_profile(
        root / "src/configs/pasteur_codexless_thin_object_grasp.json"
    )
    assert profile["execution"]["final_seating_extra_down_m"] == 0.002
