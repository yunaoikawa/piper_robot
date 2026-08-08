import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import rollout.articulated_appliance as appliance


SETTINGS = {
    "tag_family": "DICT_APRILTAG_36h11",
    "fixed_tag_ids": [3, 12],
    "minimum_fixed_tags": 2,
    "closed_state_marker_tag_id": 1,
    "maximum_registration_error_tag_lengths": 0.2,
    "roi_horizontal_marker_lengths": 4.0,
    "roi_vertical_marker_lengths": 5.2,
    "minimum_depth_m": 0.1,
    "maximum_depth_m": 2.0,
    "minimum_endpoint_depth_change_m": 0.025,
    "minimum_dynamic_tag_areas": 2.0,
    "minimum_dynamic_depth_fraction": 0.7,
    "maximum_endpoint_relative_error": 0.45,
    "minimum_endpoint_relative_margin": 0.2,
    "maximum_closed_marker_error_tag_lengths": 0.75,
    "maximum_marker_confirmed_relative_error": 0.65,
}


def _tag(tag_id, center, length=20.0):
    x, y = center
    half = length / 2
    return SimpleNamespace(
        tag_id=tag_id,
        center=np.asarray(center, dtype=float),
        corners=np.asarray(
            [
                [x - half, y - half],
                [x + half, y - half],
                [x + half, y + half],
                [x - half, y + half],
            ],
            dtype=float,
        ),
    )


def _fake_detect(image, _family, scales):
    assert tuple(scales) == (1, 2)
    result = [_tag(3, (40, 40)), _tag(12, (260, 160))]
    if int(image[0, 0, 0]) == 2:
        result.append(_tag(1, (150, 80)))
    return result


def _observation(code, depth, source):
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    image[0, 0, 0] = code
    return appliance.EndpointObservation(image, depth, source)


def _endpoints():
    opened = np.full((20, 30), 1.0, dtype=float)
    closed = opened.copy()
    yy, xx = np.indices(closed.shape)
    changed = ((xx - 15) / 8) ** 2 + ((yy - 8) / 8) ** 2 <= 1
    closed[changed] = 0.75
    return (
        _observation(0, opened, "open"),
        _observation(2, closed, "closed"),
    )


def test_verified_depth_endpoints_classify_and_missing_marker_is_not_proof(monkeypatch):
    monkeypatch.setattr(appliance, "detect_tags", _fake_detect)
    opened, closed = _endpoints()
    model = appliance.build_endpoint_model(opened, closed, SETTINGS)

    open_report = appliance.classify_endpoint_state(opened, model, SETTINGS)
    closed_report = appliance.classify_endpoint_state(closed, model, SETTINGS)

    assert open_report["state"] == "open"
    assert not open_report["closed_marker_visible"]
    assert open_report["reason"] == "registered depth matches verified open endpoint"
    assert closed_report["state"] == "closed"
    assert closed_report["closed_marker_confirmed"]


def test_between_endpoints_stays_unknown(monkeypatch):
    monkeypatch.setattr(appliance, "detect_tags", _fake_detect)
    opened, closed = _endpoints()
    model = appliance.build_endpoint_model(opened, closed, SETTINGS)
    between = _observation(0, (opened.depth_m + closed.depth_m) / 2, "between")
    report = appliance.classify_endpoint_state(between, model, SETTINGS)
    assert report["state"] == "unknown"


def test_registration_fails_closed_when_fixed_tags_disappear(monkeypatch):
    opened, closed = _endpoints()
    monkeypatch.setattr(appliance, "detect_tags", _fake_detect)
    model = appliance.build_endpoint_model(opened, closed, SETTINGS)
    monkeypatch.setattr(appliance, "detect_tags", lambda *_args, **_kwargs: [])
    with pytest.raises(RuntimeError, match="fixed tags"):
        appliance.classify_endpoint_state(opened, model, SETTINGS)


def test_close_workflow_uses_dedicated_demo_not_reverse_opening():
    stages = appliance.workflow_stages("closed")
    assert stages == (
        "observe-open",
        "dedicated-open-jaw-close-demo",
        "verify-closed",
    )
    assert not any("reverse" in stage for stage in stages)


def test_current_pasteur_endpoint_references_regression():
    profile_path = Path("src/configs/pasteur_incubator_door_demo.json")
    # Endpoint recordings are deliberately excluded from git.  Exercise the
    # deployed Pasteur calibration when the local data volume is mounted.
    profile = json.loads(profile_path.read_text())
    settings = profile["state_detection"]
    opened_paths = settings["references"]["open"]
    closed_paths = settings["references"]["closed"]
    paths = [
        Path(opened_paths["image"]),
        Path(opened_paths["depth"]),
        Path(closed_paths["image"]),
        Path(closed_paths["depth"]),
    ]
    if not all(path.exists() for path in paths):
        pytest.skip("Pasteur endpoint calibration volume is not mounted")
    opened = appliance.load_endpoint(paths[0], paths[1])
    closed = appliance.load_endpoint(paths[2], paths[3])
    model = appliance.build_endpoint_model(opened, closed, settings)
    assert appliance.classify_endpoint_state(opened, model, settings)["state"] == "open"
    assert appliance.classify_endpoint_state(closed, model, settings)["state"] == "closed"
