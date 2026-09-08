"""Headless Record3D RGB-D capture for live SAM observations.

This module is deliberately independent of ``rollout.camera``.  The latter is
the interactive teleoperation camera manager and may create display or stream
server resources; a SAM observation only needs an atomic, fresh RGB-D sample.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from record3d import Record3DStream

from robot.camera_id import load_camera_map


RECORD3D_SHUTDOWN_TIMEOUT_S = 5.0
RECORD3D_SHUTDOWN_GRACE_S = 0.10


def _note_exception(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)


class SamHeadlessRecord3DManager:
    """Capture atomic head-camera RGB-D frames without UI or an HTTP server.

    A frame timestamp is assigned inside Record3D's ``on_new_frame`` callback.
    Merely polling :meth:`get_latest_frame` can therefore never make an old
    native buffer appear fresh.
    """

    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.session = None
        self._started = False
        self._lifecycle_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_timestamp = None
        self._capture_error = None
        self._native_stop_lock = threading.Lock()
        self._native_stop_count = 0
        self._native_stopped = threading.Event()

    def start(self) -> None:
        """Connect to the configured Record3D head camera.

        ``Record3DStream.connect`` starts the binding's native capture thread,
        so no Python polling, display, or server thread is needed here.
        """

        with self._lifecycle_lock:
            if self._started:
                return
            if self.stop_event.is_set():
                raise RuntimeError(
                    "cannot start SAM head camera after stop was requested"
                )

            devices = Record3DStream.get_connected_devices()
            camera_map = load_camera_map()
            try:
                device_index = int(camera_map.get("head", 0))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Record3D head camera index is not an integer"
                ) from exc
            if device_index < 0 or device_index >= len(devices):
                raise RuntimeError(
                    "Record3D head camera unavailable: "
                    f"configured index {device_index}, "
                    f"found {len(devices)} device(s)"
                )

            session = Record3DStream()
            session.on_new_frame = self._on_new_frame
            session.on_stream_stopped = self._on_stream_stopped
            # Publish the session before connect: some bindings can invoke the
            # first callback synchronously from connect().
            self.session = session
            self._started = True

            try:
                session.connect(devices[device_index])
            except BaseException as connect_error:
                self._started = False
                self.session = None
                session.on_new_frame = lambda: None
                session.on_stream_stopped = lambda: None
                try:
                    session.disconnect()
                except BaseException as cleanup_error:
                    _note_exception(
                        connect_error,
                        "Record3D cleanup after failed connect also failed: "
                        f"{cleanup_error!r}",
                    )
                raise

    def _on_new_frame(self) -> None:
        """Copy one native RGB-D frame and publish it atomically."""

        with self._lifecycle_lock:
            session = self.session
            if (
                not self._started
                or session is None
                or self.stop_event.is_set()
            ):
                return
            try:
                # Timestamp at callback entry, not when a consumer polls.
                callback_timestamp = float(time.time())
                rgb = np.array(session.get_rgb_frame(), copy=True)
                depth = np.array(session.get_depth_frame(), copy=True)
                if rgb.ndim != 3 or rgb.size == 0:
                    raise RuntimeError(
                        "Record3D returned an invalid head RGB frame"
                    )
                if depth.ndim != 2 or depth.size == 0:
                    raise RuntimeError(
                        "Record3D returned an invalid head depth frame"
                    )
            except Exception as exc:
                with self._frame_lock:
                    self._capture_error = exc
                return

            # stop_event may have been set independently while native buffers
            # were copied.  Never publish a post-stop frame.
            if self.stop_event.is_set():
                return
            with self._frame_lock:
                self._latest_rgb = rgb
                self._latest_depth = depth
                self._latest_timestamp = callback_timestamp
                self._capture_error = None

    def _on_stream_stopped(self) -> None:
        with self._native_stop_lock:
            self._native_stop_count += 1
            if self._native_stop_count >= 2:
                self._native_stopped.set()
        with self._lifecycle_lock:
            if not self._started:
                return
            self._started = False
            with self._frame_lock:
                self._capture_error = RuntimeError(
                    "Record3D head-camera stream stopped unexpectedly"
                )

    def get_latest_frame(self):
        """Return independent RGB/depth copies and their callback timestamp."""

        with self._frame_lock:
            if self._capture_error is not None:
                raise RuntimeError(
                    "Record3D head-camera capture failed"
                ) from self._capture_error
            if self._latest_rgb is None:
                return None, None, None
            depth = (
                None
                if self._latest_depth is None
                else self._latest_depth.copy()
            )
            return (
                self._latest_rgb.copy(),
                self._latest_timestamp,
                depth,
            )

    def stop(self) -> None:
        """Wait for Record3D's detached native thread before releasing it."""

        with self._lifecycle_lock:
            session = self.session
            self._started = False
            if session is None:
                return
            self.stop_event.set()

        # Record3D 1.4 invokes the stopped callback once synchronously here,
        # then a second time immediately before its detached receive thread
        # returns. Releasing the session before that second callback can
        # segfault the interpreter.
        session.disconnect()
        if not self._native_stopped.wait(RECORD3D_SHUTDOWN_TIMEOUT_S):
            raise RuntimeError(
                "head camera Record3D native thread did not stop"
            )
        time.sleep(RECORD3D_SHUTDOWN_GRACE_S)
        session.on_new_frame = lambda: None
        session.on_stream_stopped = lambda: None
        with self._lifecycle_lock:
            if self.session is session:
                self.session = None
