"""Versioned, operator-confirmed daily bench state.

The store is intentionally independent of Codex and robot RPC.  A browser,
chat adapter, perception process, or autonomous runner can all use the same
small state machine.  Plans bind to one confirmed revision; any reported or
observed change invalidates that revision before new motion is allowed.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


class SceneNotConfirmed(RuntimeError):
    pass


@dataclass(frozen=True)
class SceneObject:
    instance_id: str
    semantic_name: str
    geometry: dict
    role: str | None = None
    pose_robot: list[list[float]] | None = None
    confidence: float = 0.0
    status: str = "uncertain"
    source: str = "sam_rgbd"
    transparent: bool = False
    mask_path: str | None = None
    depth_quality: str = "unknown"

    @classmethod
    def from_dict(cls, value: dict) -> "SceneObject":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: val for key, val in value.items() if key in allowed})


@dataclass
class DailyScene:
    scene_id: str
    revision: int
    local_date: str
    status: str
    captured_at_s: float
    calibration_id: str
    camera_ids: dict[str, str]
    objects: list[SceneObject] = field(default_factory=list)
    unknown_regions: list[dict] = field(default_factory=list)
    images: dict[str, str] = field(default_factory=dict)
    confirmed_at_s: float | None = None
    confirmed_by: str | None = None
    change_reason: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "DailyScene":
        payload = dict(value)
        payload["objects"] = [
            SceneObject.from_dict(item) for item in payload.get("objects", [])
        ]
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)


def _today(timestamp_s: float | None = None) -> str:
    timestamp_s = time.time() if timestamp_s is None else float(timestamp_s)
    return datetime.fromtimestamp(timestamp_s).astimezone().date().isoformat()


class DailySceneStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> DailyScene | None:
        if not self.path.exists():
            return None
        return DailyScene.from_dict(json.loads(self.path.read_text()))

    def save(self, scene: DailyScene) -> DailyScene:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
        )
        encoded = (json.dumps(scene.to_dict(), indent=2) + "\n").encode()
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return scene

    def propose(
        self,
        *,
        objects: Iterable[SceneObject],
        calibration_id: str,
        camera_ids: dict[str, str],
        images: dict[str, str] | None = None,
        unknown_regions: list[dict] | None = None,
        timestamp_s: float | None = None,
        reason: str = "daily_scan",
    ) -> DailyScene:
        now = time.time() if timestamp_s is None else float(timestamp_s)
        previous = self.load()
        revision = 1 if previous is None else previous.revision + 1
        scene = DailyScene(
            scene_id=f"{_today(now)}-bench",
            revision=revision,
            local_date=_today(now),
            status="pending_confirmation",
            captured_at_s=now,
            calibration_id=str(calibration_id),
            camera_ids=dict(camera_ids),
            objects=list(objects),
            unknown_regions=list(unknown_regions or []),
            images=dict(images or {}),
            change_reason=reason,
        )
        return self.save(scene)

    def confirm(self, *, revision: int, operator: str, timestamp_s=None) -> DailyScene:
        scene = self._require_revision(revision)
        now = time.time() if timestamp_s is None else float(timestamp_s)
        if scene.local_date != _today(now):
            raise SceneNotConfirmed("daily scene belongs to a different local date")
        if scene.unknown_regions:
            raise SceneNotConfirmed("unknown regions must be resolved before confirmation")
        if any(item.status not in {"confirmed", "absent"} for item in scene.objects):
            raise SceneNotConfirmed("every proposed object must be confirmed or absent")
        scene.status = "confirmed"
        scene.confirmed_at_s = now
        scene.confirmed_by = str(operator)
        scene.change_reason = None
        return self.save(scene)

    def report_change(self, reason: str, *, timestamp_s=None) -> DailyScene:
        scene = self.load()
        if scene is None:
            raise SceneNotConfirmed("no daily scene exists")
        scene.revision += 1
        scene.status = "change_reported"
        scene.confirmed_at_s = None
        scene.confirmed_by = None
        scene.change_reason = str(reason)
        scene.captured_at_s = (
            time.time() if timestamp_s is None else float(timestamp_s)
        )
        return self.save(scene)

    def replace_objects(
        self, objects: Iterable[SceneObject], *, revision: int
    ) -> DailyScene:
        scene = self._require_revision(revision)
        scene.objects = list(objects)
        scene.status = "pending_confirmation"
        scene.confirmed_at_s = None
        scene.confirmed_by = None
        return self.save(scene)

    def require_confirmed(
        self,
        *,
        revision: int | None = None,
        calibration_id: str | None = None,
        now_s: float | None = None,
    ) -> DailyScene:
        scene = self.load()
        if scene is None or scene.status != "confirmed":
            raise SceneNotConfirmed("today's bench scene is not operator-confirmed")
        if scene.local_date != _today(now_s):
            raise SceneNotConfirmed("today's bench scene has not been confirmed")
        if revision is not None and scene.revision != int(revision):
            raise SceneNotConfirmed("planned scene revision is stale")
        if calibration_id is not None and scene.calibration_id != str(calibration_id):
            raise SceneNotConfirmed("scene was captured with another calibration")
        return scene

    def _require_revision(self, revision: int) -> DailyScene:
        scene = self.load()
        if scene is None:
            raise SceneNotConfirmed("no daily scene exists")
        if scene.revision != int(revision):
            raise SceneNotConfirmed("scene revision changed; refresh before editing")
        return scene
