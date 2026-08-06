#!/usr/bin/env python3
"""Create and approve auditable thin-object grasp checkpoints.

The command is intentionally robot-I/O free.  Physical primitives use the
same result contract, but are executed by the reviewed real-time controller.
This utility lets Codex and a future Codexless state machine share exactly the
same state and transition validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.grasp_orchestration import (
    CheckpointStore,
    PrimitiveResult,
    file_hash,
)
from src.run_codexless_thin_object_grasp import audit_profile, load_profile


def _load_json_argument(value: str | None) -> dict:
    if not value:
        return {}
    path = Path(value)
    if path.is_file():
        return json.loads(path.read_text())
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("measurements must be a JSON object")
    return parsed


def _next_sequence(store: CheckpointStore) -> int:
    latest = store.root / "latest.json"
    if not latest.exists():
        return 0
    return int(store.load_result(latest)["sequence"]) + 1


def _predecessor_hash(store: CheckpointStore, value: str | None) -> str | None:
    if value:
        return str(store.load_result(value)["state_sha256"])
    latest = store.root / "latest.json"
    if latest.exists():
        return str(store.load_result(latest)["state_sha256"])
    return None


def _emit(args) -> int:
    store = CheckpointStore(args.run_dir)
    profile = Path(args.config).resolve()
    result = PrimitiveResult(
        run_id=args.run_id or store.root.name,
        sequence=(args.sequence if args.sequence is not None else _next_sequence(store)),
        primitive=args.primitive,
        accepted=bool(args.accepted),
        profile_path=str(profile),
        profile_sha256=file_hash(profile),
        input_state_sha256=_predecessor_hash(store, args.input_result),
        summary=args.summary,
        measurements=_load_json_argument(args.measurements),
        evidence=tuple(str(Path(item).resolve()) for item in args.evidence),
        commands_sent=bool(args.commands_sent),
    )
    path, value = store.publish(result)
    print(json.dumps({"result_path": str(path), **value}, indent=2, ensure_ascii=False))
    return 0 if result.accepted else 2


def _audit(args) -> int:
    """Emit observe and plan-hover results without camera or robot commands."""

    store = CheckpointStore(args.run_dir)
    profile_path = Path(args.config).resolve()
    profile = load_profile(profile_path)
    profile_digest = file_hash(profile_path)
    run_id = args.run_id or store.root.name
    observe = PrimitiveResult(
        run_id=run_id,
        sequence=_next_sequence(store),
        primitive="observe",
        accepted=True,
        profile_path=str(profile_path),
        profile_sha256=profile_digest,
        input_state_sha256=_predecessor_hash(store, None),
        summary="profile identity adapter and fixed-head inputs parsed",
        measurements={
            "physical_arm": profile["physical_arm"],
            "feature_adapter": profile["target_identity"]["feature_adapter"],
            "live_observation": False,
            "closure_authorized": False,
        },
        commands_sent=False,
    )
    observe_path, observe_value = store.publish(observe)
    audit = audit_profile(profile)
    plan = PrimitiveResult(
        run_id=run_id,
        sequence=observe.sequence + 1,
        primitive="plan-hover",
        accepted=bool(audit["accepted"]),
        profile_path=str(profile_path),
        profile_sha256=profile_digest,
        input_state_sha256=observe_value["state_sha256"],
        summary=(
            "MuJoCo route audit accepted"
            if audit["accepted"]
            else "MuJoCo route audit rejected"
        ),
        measurements={"audit": audit, "live_observation": False},
        evidence=(str(observe_path),),
        commands_sent=False,
    )
    plan_path, plan_value = store.publish(plan)
    print(
        json.dumps(
            {
                "observe_result": str(observe_path),
                "plan_result": str(plan_path),
                "latest": plan_value,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if plan.accepted else 2


def _decide(args) -> int:
    store = CheckpointStore(args.run_dir)
    result_path = Path(args.result).resolve()
    path = store.write_decision(
        result_path, args.action, reason=args.reason, actor=args.actor
    )
    print(path)
    return 0


def _status(args) -> int:
    store = CheckpointStore(args.run_dir)
    path = Path(args.result).resolve() if args.result else store.root / "latest.json"
    result = store.load_result(path)
    decision = store.read_decision(path)
    print(json.dumps({"result": result, "decision": decision}, indent=2, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="src/configs/pasteur_codexless_thin_object_grasp.json",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit")
    audit.set_defaults(func=_audit)

    emit = subparsers.add_parser("emit")
    emit.add_argument("--primitive", required=True)
    accepted = emit.add_mutually_exclusive_group(required=True)
    accepted.add_argument("--accepted", action="store_true")
    accepted.add_argument("--rejected", dest="accepted", action="store_false")
    emit.add_argument("--summary", required=True)
    emit.add_argument("--measurements")
    emit.add_argument("--evidence", action="append", default=[])
    emit.add_argument("--commands-sent", action="store_true")
    emit.add_argument("--input-result")
    emit.add_argument("--sequence", type=int)
    emit.set_defaults(func=_emit)

    decide = subparsers.add_parser("decide")
    decide.add_argument("--result", required=True)
    decide.add_argument("--action", required=True)
    decide.add_argument("--reason", required=True)
    decide.add_argument("--actor", default="codex")
    decide.set_defaults(func=_decide)

    status = subparsers.add_parser("status")
    status.add_argument("--result")
    status.set_defaults(func=_status)

    args = parser.parse_args(argv)
    started = time.time()
    try:
        return int(args.func(args))
    finally:
        _ = started


if __name__ == "__main__":
    raise SystemExit(main())
