"""Auditable checkpoint contracts for thin-object grasp orchestration.

This module contains no camera or robot I/O.  It defines the small semantic
steps that Codex, a human operator, or a deterministic state machine may
sequence while the existing real-time controller remains responsible for
motion streaming and pressure safety.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence


RESULT_SCHEMA = "piper_robot.thin_object_primitive_result/v1"
DECISION_SCHEMA = "piper_robot.thin_object_primitive_decision/v1"

PRIMITIVES = (
    "observe",
    "plan-hover",
    "move-hover",
    "align-hover",
    "descend-bottom",
    "seat-2mm",
    "close",
    "verify-lift",
    "recover",
    "home",
    "stop",
)

_SUCCESSORS = {
    "observe": ("plan-hover", "stop"),
    "plan-hover": ("move-hover", "stop"),
    "move-hover": ("align-hover", "recover", "stop"),
    "align-hover": ("descend-bottom", "recover", "stop"),
    "descend-bottom": ("seat-2mm", "recover", "stop"),
    "seat-2mm": ("close", "recover", "stop"),
    "close": ("verify-lift", "recover", "stop"),
    "verify-lift": ("home", "recover", "stop"),
    "recover": ("observe", "home", "stop"),
    "home": ("observe", "stop"),
    "stop": (),
}

_FAILURE_SUCCESSORS = {
    "observe": ("observe", "stop"),
    "plan-hover": ("observe", "stop"),
    "move-hover": ("recover", "stop"),
    "align-hover": ("recover", "stop"),
    "descend-bottom": ("recover", "stop"),
    "seat-2mm": ("recover", "stop"),
    "close": ("recover", "stop"),
    "verify-lift": ("recover", "stop"),
    "recover": ("home", "stop"),
    "home": ("stop",),
    "stop": (),
}


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allowed_next_actions(primitive: str, accepted: bool) -> tuple[str, ...]:
    if primitive not in PRIMITIVES:
        raise ValueError(f"unknown grasp primitive: {primitive}")
    table = _SUCCESSORS if accepted else _FAILURE_SUCCESSORS
    return table[primitive]


@dataclass(frozen=True)
class PrimitiveResult:
    run_id: str
    sequence: int
    primitive: str
    accepted: bool
    profile_path: str
    profile_sha256: str
    input_state_sha256: str | None
    summary: str
    measurements: Mapping[str, Any] = field(default_factory=dict)
    evidence: Sequence[str] = field(default_factory=tuple)
    commands_sent: bool = False
    allowed_next_actions: Sequence[str] = field(default_factory=tuple)
    created_at_s: float = field(default_factory=time.time)
    schema: str = RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"unknown grasp primitive: {self.primitive}")
        expected = allowed_next_actions(self.primitive, self.accepted)
        supplied = tuple(self.allowed_next_actions)
        if supplied and supplied != expected:
            raise ValueError(
                f"allowed actions {supplied} do not match contract {expected}"
            )
        object.__setattr__(self, "allowed_next_actions", expected)
        if self.sequence < 0:
            raise ValueError("primitive sequence must be non-negative")

    def unsigned_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence"] = list(self.evidence)
        value["allowed_next_actions"] = list(self.allowed_next_actions)
        return value

    def to_dict(self) -> dict[str, Any]:
        value = self.unsigned_dict()
        value["state_sha256"] = canonical_hash(value)
        return value


class CheckpointStore:
    """Write append-only primitive results and hash-bound decisions."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{time.monotonic_ns()}.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n"
        )
        temporary.replace(path)

    def publish(self, result: PrimitiveResult) -> tuple[Path, dict[str, Any]]:
        value = result.to_dict()
        stage = self.root / f"{result.sequence:03d}_{result.primitive}"
        path = stage / "result.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old != value:
                raise FileExistsError(f"checkpoint is immutable: {path}")
            return path, old
        self._atomic_json(path, value)
        self._atomic_json(self.root / "latest.json", value)
        return path, value

    def load_result(self, path: str | Path) -> dict[str, Any]:
        value = json.loads(Path(path).resolve().read_text())
        if value.get("schema") != RESULT_SCHEMA:
            raise ValueError("primitive result schema mismatch")
        state_hash = value.pop("state_sha256", None)
        expected = canonical_hash(value)
        value["state_sha256"] = state_hash
        if state_hash != expected:
            raise ValueError("primitive result hash mismatch")
        return value

    def write_decision(
        self,
        result_path: str | Path,
        action: str,
        *,
        reason: str,
        actor: str = "codex",
    ) -> Path:
        result_path = Path(result_path).resolve()
        result = self.load_result(result_path)
        if action not in result["allowed_next_actions"]:
            raise ValueError(
                f"{action!r} is not allowed after {result['primitive']!r}"
            )
        decision = {
            "schema": DECISION_SCHEMA,
            "run_id": result["run_id"],
            "sequence": result["sequence"],
            "result_state_sha256": result["state_sha256"],
            "action": action,
            "reason": str(reason),
            "actor": str(actor),
            "created_at_s": time.time(),
        }
        path = result_path.with_name("decision.json")
        if path.exists():
            old = json.loads(path.read_text())
            if old != decision:
                raise FileExistsError(f"decision already exists: {path}")
            return path
        self._atomic_json(path, decision)
        return path

    def read_decision(self, result_path: str | Path) -> dict[str, Any] | None:
        result_path = Path(result_path).resolve()
        path = result_path.with_name("decision.json")
        if not path.exists():
            return None
        result = self.load_result(result_path)
        decision = json.loads(path.read_text())
        if decision.get("schema") != DECISION_SCHEMA:
            raise ValueError("primitive decision schema mismatch")
        if decision.get("result_state_sha256") != result["state_sha256"]:
            raise ValueError("decision was issued for a different result state")
        if decision.get("action") not in result["allowed_next_actions"]:
            raise ValueError("decision action is not allowed by the result")
        return decision


class ControllerLease:
    """Process-wide, non-blocking lease for one physical arm controller."""

    def __init__(self, path: str | Path, *, owner: Mapping[str, Any] | None = None):
        self.path = Path(path).resolve()
        self.owner = dict(owner or {})
        self._stream = None

    def acquire(self) -> "ControllerLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.seek(0)
            holder = stream.read().strip() or "unknown holder"
            stream.close()
            raise RuntimeError(f"physical controller is already leased: {holder}") from error
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "acquired_at_s": time.time(),
                    **self.owner,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stream.flush()
        self._stream = stream
        return self

    def release(self) -> None:
        if self._stream is None:
            return
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._stream = None

    def __enter__(self) -> "ControllerLease":
        return self.acquire()

    def __exit__(self, *_exc) -> None:
        self.release()


def seating_distance_m(profile: Mapping[str, Any]) -> float:
    """Return the single post-bottom seating command from the profile."""

    execution = profile.get("execution", {})
    distance = float(execution.get("final_seating_extra_down_m", 0.0))
    if not 0.0 < distance <= 0.002:
        raise ValueError(
            "final_seating_extra_down_m must be configured in (0, 0.002]"
        )
    return distance


def validate_seating_preconditions(
    predecessor: Mapping[str, Any], *, already_applied: bool
) -> float:
    """Fail closed unless normal descent completed exactly once before seating."""

    if predecessor.get("primitive") != "descend-bottom":
        raise ValueError("seat-2mm requires a descend-bottom predecessor")
    if predecessor.get("accepted") is not True:
        raise ValueError("seat-2mm requires accepted normal descent")
    if already_applied:
        raise ValueError("seat-2mm is single-shot and has already been applied")
    return 0.002
