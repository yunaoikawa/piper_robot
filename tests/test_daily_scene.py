from __future__ import annotations

from datetime import datetime
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.daily_scene import DailySceneStore, SceneNotConfirmed, SceneObject


def item(status="uncertain"):
    return SceneObject(
        instance_id="petri-body",
        semantic_name="petri dish body",
        geometry={
            "type": "cylinder",
            "radius_m": 0.045,
            "height_m": 0.015,
        },
        status=status,
        transparent=True,
    )


def raises_scene_not_confirmed(call, contains=None):
    try:
        call()
    except SceneNotConfirmed as error:
        if contains is not None:
            assert contains in str(error)
    else:
        raise AssertionError("SceneNotConfirmed was not raised")


def check_daily_confirmation_and_change_invalidate_bound_revision(tmp_path):
    now = datetime.now().astimezone().timestamp()
    store = DailySceneStore(tmp_path / "daily.json")
    draft = store.propose(
        objects=[item()],
        calibration_id="cal-1",
        camera_ids={"head": "head-1"},
        timestamp_s=now,
    )
    raises_scene_not_confirmed(lambda: store.require_confirmed(now_s=now))
    raises_scene_not_confirmed(
        lambda: store.confirm(
            revision=draft.revision, operator="test", timestamp_s=now
        )
    )

    draft = store.replace_objects(
        [item("confirmed")], revision=draft.revision
    )
    confirmed = store.confirm(
        revision=draft.revision, operator="test", timestamp_s=now
    )
    assert store.require_confirmed(
        revision=confirmed.revision,
        calibration_id="cal-1",
        now_s=now,
    ).confirmed_by == "test"

    changed = store.report_change("moved dish", timestamp_s=now)
    assert changed.revision == confirmed.revision + 1
    raises_scene_not_confirmed(
        lambda: store.require_confirmed(
            revision=confirmed.revision, now_s=now
        )
    )


def check_unknown_regions_cannot_be_operator_confirmed(tmp_path):
    now = datetime.now().astimezone().timestamp()
    store = DailySceneStore(tmp_path / "daily.json")
    draft = store.propose(
        objects=[item("confirmed")],
        calibration_id="cal-1",
        camera_ids={},
        unknown_regions=[{"reason": "unobserved behind arm"}],
        timestamp_s=now,
    )
    raises_scene_not_confirmed(
        lambda: store.confirm(
            revision=draft.revision, operator="test", timestamp_s=now
        ),
        "unknown regions",
    )


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    check_daily_confirmation_and_change_invalidate_bound_revision(root / "a")
    check_unknown_regions_cannot_be_operator_confirmed(root / "b")

print("daily scene checks passed")
