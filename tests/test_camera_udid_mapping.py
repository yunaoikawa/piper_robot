#!/usr/bin/env python3
"""Camera enumeration reorders must not swap autonomous views."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import robot.camera_id as camera_id


with tempfile.TemporaryDirectory() as directory:
    camera_id.CAM_MAP_FILE = Path(directory) / "camera_map.json"
    camera_id.CAM_MAP_FILE.write_text(
        json.dumps({"head": 0, "right": 1, "left": 2})
    )
    camera_id.Record3DStream.get_connected_devices = staticmethod(
        lambda: [
            SimpleNamespace(udid="right-id"),
            SimpleNamespace(udid="left-id"),
            SimpleNamespace(udid="head-id"),
        ]
    )
    mapping, live = camera_id.configure_camera_map_by_udid(
        {"head": "head-id", "right": "right-id", "left": "left-id"}
    )
    assert mapping == {"head": 2, "right": 0, "left": 1}
    assert live == {0: "right-id", 1: "left-id", 2: "head-id"}
    assert json.loads(camera_id.CAM_MAP_FILE.read_text()) == mapping

    try:
        camera_id.configure_camera_map_by_udid({"head": "missing"})
    except RuntimeError as error:
        assert "not connected" in str(error)
    else:
        raise AssertionError("missing calibrated camera was accepted")

print("camera UDID mapping checks passed")
