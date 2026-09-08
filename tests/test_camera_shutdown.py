#!/usr/bin/env python3

import sys
import threading
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout import camera


class FakeThread:
    def __init__(self):
        self.join_calls = []

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class FakeSession:
    def __init__(self):
        self.on_new_frame = lambda: None
        self.on_stream_stopped = lambda: None
        self.disconnect_calls = 0

    def disconnect(self):
        self.disconnect_calls += 1
        # Record3D 1.4 calls once in the public disconnect method and once
        # again as its detached native receive loop returns.
        self.on_stream_stopped()
        self.on_stream_stopped()


with mock.patch.object(camera.time, "sleep", return_value=None):
    stop_event = threading.Event()
    head = camera.CameraFeedManager(
        stop_event,
        display=False,
        head_stream=False,
    )
    head.iphone_thread = FakeThread()
    head.display_thread = None
    head.session = FakeSession()
    head.session.on_stream_stopped = head._on_stream_stopped
    head_session = head.session
    head.stop()
    assert stop_event.is_set()
    assert head.iphone_thread.join_calls == [2.0]
    assert head_session.disconnect_calls == 1
    assert head._native_stop_count == 2
    assert head.session is None

    stop_event = threading.Event()
    wrist = camera.USBWristCameraFeedManager(
        stop_event,
        device_index=2,
        label="right wrist",
    )
    wrist.thread = FakeThread()
    wrist.session = FakeSession()
    wrist.session.on_stream_stopped = wrist._on_stream_stopped
    wrist_session = wrist.session
    wrist.stop()
    assert stop_event.is_set()
    assert wrist.thread.join_calls == [2.0]
    assert wrist_session.disconnect_calls == 1
    assert wrist._native_stop_count == 2
    assert wrist.session is None

print("Record3D camera shutdown checks passed")
