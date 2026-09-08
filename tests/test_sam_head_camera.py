#!/usr/bin/env python3

import inspect
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout import sam_head_camera


class FakeRecord3DStream:
    devices = [object()]
    instances = []
    fail_connect = False
    fail_disconnect = False

    @classmethod
    def get_connected_devices(cls):
        return list(cls.devices)

    def __init__(self):
        self.on_new_frame = None
        self.on_stream_stopped = None
        self.rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        self.depth = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.connected_device = None
        self.disconnect_calls = 0
        type(self).instances.append(self)

    def connect(self, device):
        self.connected_device = device
        if type(self).fail_connect:
            raise RuntimeError("synthetic connect failure")

    def disconnect(self):
        self.disconnect_calls += 1
        if type(self).fail_disconnect:
            raise RuntimeError("synthetic disconnect failure")
        self.on_stream_stopped()
        self.on_stream_stopped()

    def get_rgb_frame(self):
        return self.rgb

    def get_depth_frame(self):
        return self.depth


class SamHeadCameraTest(unittest.TestCase):
    def setUp(self):
        FakeRecord3DStream.devices = [object()]
        FakeRecord3DStream.instances = []
        FakeRecord3DStream.fail_connect = False
        FakeRecord3DStream.fail_disconnect = False
        self.stream_patch = mock.patch.object(
            sam_head_camera, "Record3DStream", FakeRecord3DStream
        )
        self.map_patch = mock.patch.object(
            sam_head_camera,
            "load_camera_map",
            return_value={"head": 0},
        )
        self.stream_patch.start()
        self.map_patch.start()

    def tearDown(self):
        self.map_patch.stop()
        self.stream_patch.stop()

    def test_constructor_is_clean_one_argument_api(self):
        signature = inspect.signature(
            sam_head_camera.SamHeadlessRecord3DManager
        )
        self.assertEqual(list(signature.parameters), ["stop_event"])

        manager = sam_head_camera.SamHeadlessRecord3DManager(
            threading.Event()
        )
        self.assertEqual(manager.get_latest_frame(), (None, None, None))
        manager.stop()

    def test_only_callbacks_publish_fresh_timestamps_and_frame_copies(self):
        timestamps = iter([100.0, 101.0])
        with mock.patch.object(
            sam_head_camera.time,
            "time",
            side_effect=lambda: next(timestamps),
        ):
            manager = sam_head_camera.SamHeadlessRecord3DManager(
                threading.Event()
            )
            manager.start()
            session = FakeRecord3DStream.instances[-1]

            self.assertEqual(
                manager.get_latest_frame(), (None, None, None)
            )
            session.on_new_frame()
            rgb, timestamp, depth = manager.get_latest_frame()
            self.assertEqual(timestamp, 100.0)
            self.assertTrue(np.array_equal(rgb, session.rgb))
            self.assertTrue(np.array_equal(depth, session.depth))

            # Neither native buffers nor consumer-owned copies can mutate the
            # manager's snapshot or invent a new timestamp without a callback.
            rgb[:] = 255
            depth[:] = 255
            session.rgb[:] = 42
            session.depth[:] = 42
            unchanged_rgb, unchanged_timestamp, unchanged_depth = (
                manager.get_latest_frame()
            )
            self.assertEqual(unchanged_timestamp, 100.0)
            self.assertFalse(np.all(unchanged_rgb == 255))
            self.assertFalse(np.all(unchanged_rgb == 42))
            self.assertFalse(np.all(unchanged_depth == 255))
            self.assertFalse(np.all(unchanged_depth == 42))

            session.on_new_frame()
            new_rgb, new_timestamp, new_depth = (
                manager.get_latest_frame()
            )
            self.assertEqual(new_timestamp, 101.0)
            self.assertTrue(np.all(new_rgb == 42))
            self.assertTrue(np.all(new_depth == 42))
            manager.stop()

    def test_stop_disconnects_once_and_late_callback_cannot_publish(self):
        timestamps = iter([100.0, 101.0])
        with mock.patch.object(
            sam_head_camera.time,
            "time",
            side_effect=lambda: next(timestamps),
        ):
            manager = sam_head_camera.SamHeadlessRecord3DManager(
                threading.Event()
            )
            manager.start()
            session = FakeRecord3DStream.instances[-1]
            callback = session.on_new_frame
            callback()
            _, timestamp, _ = manager.get_latest_frame()
            self.assertEqual(timestamp, 100.0)

            manager.stop()
            manager.stop()
            self.assertEqual(session.disconnect_calls, 1)
            session.rgb[:] = 77
            callback()
            latest_rgb, latest_timestamp, _ = (
                manager.get_latest_frame()
            )
            self.assertEqual(latest_timestamp, 100.0)
            self.assertFalse(np.all(latest_rgb == 77))

    def test_failed_connect_is_cleaned_up_and_stop_remains_safe(self):
        FakeRecord3DStream.fail_connect = True
        manager = sam_head_camera.SamHeadlessRecord3DManager(
            threading.Event()
        )

        with self.assertRaisesRegex(
            RuntimeError, "synthetic connect failure"
        ):
            manager.start()

        session = FakeRecord3DStream.instances[-1]
        self.assertEqual(session.disconnect_calls, 1)
        self.assertIsNone(manager.session)
        manager.stop()
        self.assertEqual(session.disconnect_calls, 1)

    def test_callback_failure_is_contained_and_next_frame_recovers(self):
        manager = sam_head_camera.SamHeadlessRecord3DManager(
            threading.Event()
        )
        manager.start()
        session = FakeRecord3DStream.instances[-1]
        session.rgb = np.zeros((2, 3), dtype=np.uint8)

        # A native callback must not throw across the binding boundary.
        session.on_new_frame()
        with self.assertRaisesRegex(
            RuntimeError, "head-camera capture failed"
        ):
            manager.get_latest_frame()

        session.rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        session.on_new_frame()
        rgb, timestamp, depth = manager.get_latest_frame()
        self.assertEqual(rgb.shape, (2, 3, 3))
        self.assertIsInstance(timestamp, float)
        self.assertEqual(depth.shape, (2, 3))
        manager.stop()

    def test_disconnect_failure_leaves_manager_safely_stopped(self):
        manager = sam_head_camera.SamHeadlessRecord3DManager(
            threading.Event()
        )
        manager.start()
        session = FakeRecord3DStream.instances[-1]
        FakeRecord3DStream.fail_disconnect = True

        with self.assertRaisesRegex(
            RuntimeError, "synthetic disconnect failure"
        ):
            manager.stop()

        self.assertIs(manager.session, session)
        FakeRecord3DStream.fail_disconnect = False
        manager.stop()
        self.assertIsNone(manager.session)
        self.assertEqual(session.disconnect_calls, 2)

    def test_live_sam_uses_manager_with_clean_constructor(self):
        from src import run_realtime_sam_grasp as grasp

        created = []

        class FakeHeadlessManager:
            def __init__(self, stop_event):
                created.append(stop_event)

            def stop(self):
                pass

        class FakeSam:
            def __init__(self, *_args, **_kwargs):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                torque_config="src/configs/pasteur_lid_torque.json",
                scene_config="src/configs/pasteur_lid_scene3d.json",
                output_dir=str(Path(directory) / "artifacts"),
                sam_endpoint="tcp://unused",
            )
            with (
                mock.patch.object(
                    grasp,
                    "SamHeadlessRecord3DManager",
                    FakeHeadlessManager,
                ),
                mock.patch.object(
                    grasp, "SamSegmentationClient", FakeSam
                ),
                mock.patch.object(
                    grasp,
                    "RPCClient",
                    side_effect=lambda *_args, **_kwargs: (
                        SimpleNamespace()
                    ),
                ),
            ):
                runner = grasp.LiveSamGrasp(args)
                self.assertEqual(created, [runner.stop_event])
                self.assertIsInstance(
                    runner.camera, FakeHeadlessManager
                )
                runner.stop()


if __name__ == "__main__":
    unittest.main()
