#!/usr/bin/env python3

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.run_realtime_sam_grasp as grasp
from rollout.sam_segmentation import (
    MaskCandidate,
    SamSegmentationClient,
    SegmentationResult,
    compute_candidate_roi,
    decode_request,
    encode_request,
    encode_response,
)


class FakeSamSocket:
    def __init__(self, candidate, *, receive_error=None, events=None):
        self.candidate = candidate
        self.receive_error = receive_error
        self.events = [] if events is None else events
        self.sent = None

    def send_multipart(self, parts):
        self.events.append("send")
        self.sent = tuple(bytes(part) for part in parts)

    def recv_multipart(self):
        self.events.append("receive")
        if self.receive_error is not None:
            raise self.receive_error
        metadata, _ = decode_request(list(self.sent))
        return encode_response(
            frame_id=metadata["frame_id"],
            source_timestamp=metadata["timestamp"],
            model="test-sam",
            inference_ms=3.25,
            candidates=[self.candidate],
        )


def _wire_fixture():
    image = np.zeros((72, 96, 3), np.uint8)
    image[:, :, 0] = np.arange(96, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], bool)
    mask[20:50, 30:65] = True
    candidate = MaskCandidate(
        mask=mask,
        box_xyxy=np.array([30.0, 20.0, 65.0, 50.0]),
        score=0.91,
    )
    socket = FakeSamSocket(candidate)
    client = object.__new__(SamSegmentationClient)
    client.socket = socket
    client.last_frame_id = -1
    captured = []

    def observe_request(parts):
        assert socket.sent is None
        socket.events.append("observe")
        captured.append(parts)

    result = client.segment(
        image,
        frame_id=17,
        timestamp=123.125,
        prompt="test lid",
        confidence_threshold=0.05,
        jpeg_quality=grasp.SAM_REQUEST_JPEG_QUALITY,
        request_observer=observe_request,
    )
    assert len(captured) == 1
    wire_request = captured[0]
    return image, mask, client, socket, result, wire_request


def test_exact_wire_jpeg_is_returned():
    image, _, client, socket, result, wire_request = _wire_fixture()
    assert wire_request == socket.sent
    assert socket.events == ["observe", "send", "receive"]
    metadata, decoded = decode_request(list(socket.sent))
    assert metadata == {
        "version": 1,
        "frame_id": 17,
        "timestamp": 123.125,
        "prompt": "test lid",
        "confidence_threshold": 0.05,
    }
    assert decoded.shape == image.shape
    assert result.frame_id == 17
    assert result.source_timestamp == 123.125
    assert client.last_frame_id == 17


def test_request_observer_runs_before_send_and_survives_timeout():
    image, mask, _, _, _, _ = _wire_fixture()
    candidate = MaskCandidate(
        mask=mask,
        box_xyxy=np.array([30.0, 20.0, 65.0, 50.0]),
        score=0.91,
    )
    events = []
    socket = FakeSamSocket(
        candidate,
        receive_error=TimeoutError("simulated SAM timeout"),
        events=events,
    )
    client = object.__new__(SamSegmentationClient)
    client.socket = socket
    client.last_frame_id = -1
    captured = []

    def observer(parts):
        assert socket.sent is None
        events.append("observe")
        captured.append(parts)

    try:
        client.segment(
            image,
            frame_id=18,
            timestamp=124.0,
            request_observer=observer,
        )
        raise AssertionError("SAM receive timeout was ignored")
    except TimeoutError as exc:
        assert "simulated SAM timeout" in str(exc)
    assert events == ["observe", "send", "receive"]
    assert len(captured) == 1
    assert captured[0] == socket.sent
    assert all(isinstance(part, bytes) for part in captured[0])
    assert client.last_frame_id == -1


def test_request_observer_failure_prevents_send():
    image, mask, _, _, _, _ = _wire_fixture()
    candidate = MaskCandidate(
        mask=mask,
        box_xyxy=np.array([30.0, 20.0, 65.0, 50.0]),
        score=0.91,
    )
    socket = FakeSamSocket(candidate)
    client = object.__new__(SamSegmentationClient)
    client.socket = socket
    client.last_frame_id = -1

    def observer(_):
        raise RuntimeError("durable journal failed")

    try:
        client.segment(
            image,
            frame_id=19,
            timestamp=125.0,
            request_observer=observer,
        )
        raise AssertionError("request was sent without durable evidence")
    except RuntimeError as exc:
        assert "durable journal failed" in str(exc)
    assert socket.sent is None
    assert socket.events == []
    assert client.last_frame_id == -1


def _write_bundle(directory, image, mask, wire_request):
    directory.mkdir()
    writer = grasp._ObservationArtifactWriter.reserve(
        directory,
        run_id="test-run",
        attempt_id="test-attempt",
    )
    prefix = f"{writer.sequence:03d}"
    writer.add_image("raw_image", f"{prefix}_head_raw.png", image)
    writer.add_image(
        "sam_input_png", f"{prefix}_head_sam_input.png", image
    )
    writer.add_bytes(
        "sam_request_jpeg_q90",
        f"{prefix}_head_sam_request_q90.jpg",
        wire_request[1],
        media_type="image/jpeg",
    )
    metadata_path = writer.add_bytes(
        "sam_request_000_wire_metadata",
        f"{prefix}_head_sam_request_000_metadata.json",
        wire_request[0],
        media_type="application/json",
    )
    writer.add_image(
        "lid_mask",
        f"{prefix}_head_lid_mask.png",
        mask.astype(np.uint8) * 255,
    )
    writer.add_image(
        "gripper_mask",
        f"{prefix}_head_gripper_mask.png",
        np.flip(mask, axis=1).astype(np.uint8) * 255,
    )
    depth = np.linspace(
        0.7, 1.1, image.shape[0] * image.shape[1], dtype=np.float64
    ).reshape(image.shape[:2])
    camera_matrix = np.array(
        [[500.0, 0.0, 48.0], [0.0, 501.0, 36.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    timestamps = np.array([123.125, 123.158, 123.191], dtype=np.float64)
    writer.add_npz(
        "depth_npz",
        f"{prefix}_head_depth.npz",
        depth_m=depth,
        camera_matrix=camera_matrix,
        source_timestamps=timestamps,
        image_timestamp=np.asarray(123.125, dtype=np.float64),
        native_depth_shape_hw=np.asarray(image.shape[:2], dtype=np.int64),
    )
    writer.add_image("overlay_image", f"{prefix}.png", image)
    request_metadata = json.loads(wire_request[0])
    document, manifest_path, manifest_sha256 = writer.finish(
        {
            "source_timestamp": 123.125,
            "sam_transport": {
                "request_image_format": "jpeg",
                "jpeg_quality": 90,
                "requests": [
                    {
                        "wire_metadata": request_metadata,
                        "wire_metadata_artifact_role": (
                            "sam_request_000_wire_metadata"
                        ),
                        "wire_metadata_artifact_path": metadata_path.name,
                        "wire_metadata_sha256": hashlib.sha256(
                            wire_request[0]
                        ).hexdigest(),
                    }
                ],
            },
            "depth": {
                "source_timestamps": timestamps.tolist(),
                "camera_matrix": camera_matrix.tolist(),
            },
        }
    )
    return (
        document,
        manifest_path,
        manifest_sha256,
        depth,
        camera_matrix,
        timestamps,
    )


def test_bundle_round_trip_and_canonical_manifest():
    image, mask, _, _, _, wire_request = _wire_fixture()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first = _write_bundle(root / "first", image, mask, wire_request)
        second = _write_bundle(root / "second", image, mask, wire_request)
        first_document, first_manifest, first_hash, depth, matrix, times = first
        second_document, second_manifest, second_hash, *_ = second

        first_bytes = first_manifest.read_bytes()
        assert first_bytes == second_manifest.read_bytes()
        assert first_document == second_document
        assert first_hash == second_hash
        assert hashlib.sha256(first_bytes).hexdigest() == first_hash
        assert json.loads(first_bytes) == first_document
        assert first_document["schema"] == "sam_head_observation/v2"
        assert first_document["run_id"] == "test-run"
        assert first_document["attempt_id"] == "test-attempt"

        files = first_document["files"]
        request_path = first_manifest.parent / files[
            "sam_request_jpeg_q90"
        ]["path"]
        assert request_path.read_bytes() == wire_request[1]
        assert (
            hashlib.sha256(request_path.read_bytes()).hexdigest()
            == files["sam_request_jpeg_q90"]["sha256"]
        )
        metadata_path = first_manifest.parent / files[
            "sam_request_000_wire_metadata"
        ]["path"]
        assert metadata_path.read_bytes() == wire_request[0]
        assert (
            hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            == first_document["sam_transport"]["requests"][0][
                "wire_metadata_sha256"
            ]
        )

        lid_path = first_manifest.parent / files["lid_mask"]["path"]
        saved_mask = cv2.imread(str(lid_path), cv2.IMREAD_GRAYSCALE)
        assert np.array_equal(saved_mask > 0, mask)

        depth_path = first_manifest.parent / files["depth_npz"]["path"]
        with np.load(depth_path) as saved:
            assert np.array_equal(saved["depth_m"], depth)
            assert np.array_equal(saved["camera_matrix"], matrix)
            assert np.array_equal(saved["source_timestamps"], times)
            assert float(saved["image_timestamp"]) == 123.125
            assert np.array_equal(
                saved["native_depth_shape_hw"], image.shape[:2]
            )


def test_reservation_is_collision_free_and_final_files_never_overwrite():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "000.png").write_bytes(b"legacy zero")
        (root / "002_head_observation.json").write_bytes(b"legacy two")
        first = grasp._ObservationArtifactWriter.reserve(
            root,
            run_id="run-a",
            attempt_id="attempt-a",
        )
        second = grasp._ObservationArtifactWriter.reserve(
            root,
            run_id="run-b",
            attempt_id="attempt-b",
        )
        assert first.sequence == 3
        assert second.sequence == 4
        assert first.reservation_path.exists()
        assert second.reservation_path.exists()
        first_reservation = json.loads(first.reservation_path.read_bytes())
        assert first_reservation["run_id"] == "run-a"
        assert first_reservation["attempt_id"] == "attempt-a"

        first.add_bytes(
            "payload",
            "003_payload.bin",
            b"new payload",
            media_type="application/octet-stream",
        )
        target = root / "004_payload.bin"
        target.write_bytes(b"someone else's payload")
        try:
            second.add_bytes(
                "payload",
                target.name,
                b"replacement",
                media_type="application/octet-stream",
            )
            raise AssertionError("existing final artifact was overwritten")
        except FileExistsError:
            pass
        assert target.read_bytes() == b"someone else's payload"

        first_document, first_manifest, _ = first.finish(
            {
                "status": "ok",
                "run_id": "forged-run",
                "attempt_id": "forged-attempt",
                "sequence": 999,
            }
        )
        second_failure = second.fail(RuntimeError("artifact collision"))
        assert first_document["run_id"] == "run-a"
        assert first_document["attempt_id"] == "attempt-a"
        assert first_document["sequence"] == 3
        assert first_manifest.exists()
        assert not first.reservation_path.exists()
        assert second_failure.exists()
        assert not second.reservation_path.exists()
        assert not (root / "004_head_observation.json").exists()

        third = grasp._ObservationArtifactWriter.reserve(
            root,
            run_id="run-c",
            attempt_id="attempt-c",
        )
        assert third.sequence == 5
        third.fail(RuntimeError("test cleanup"))


def test_exclusive_publish_failure_is_loud_and_preserves_target():
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "artifact.bin"
        target.write_bytes(b"existing")
        try:
            grasp._publish_new_bytes(target, b"replacement")
            raise AssertionError("existing artifact was overwritten")
        except FileExistsError:
            pass
        assert target.read_bytes() == b"existing"
        assert not list(target.parent.glob(f".{target.name}.*.tmp"))

        missing = target.parent / "missing.bin"
        original_link = grasp.os.link

        def fail_link(source, destination):
            raise OSError("simulated disk full")

        grasp.os.link = fail_link
        try:
            try:
                grasp._publish_new_bytes(missing, b"payload")
                raise AssertionError("artifact publication failure was ignored")
            except RuntimeError as exc:
                assert "simulated disk full" in str(exc)
                assert str(missing) in str(exc)
        finally:
            grasp.os.link = original_link
        assert not missing.exists()
        assert not list(missing.parent.glob(f".{missing.name}.*.tmp"))


def test_timeout_writes_failure_journal_with_exact_wire_evidence():
    image, mask, _, _, _, _ = _wire_fixture()
    candidate = MaskCandidate(
        mask=mask,
        box_xyxy=np.array([30.0, 20.0, 65.0, 50.0]),
        score=0.91,
    )
    socket = FakeSamSocket(
        candidate,
        receive_error=TimeoutError("SAM did not reply"),
    )
    client = object.__new__(SamSegmentationClient)
    client.socket = socket
    client.last_frame_id = -1

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        writer = grasp._ObservationArtifactWriter.reserve(
            root,
            run_id="timeout-run",
            attempt_id="timeout-attempt",
        )
        prefix = f"{writer.sequence:03d}"
        requests = []
        writer.failure_context["sam_transport"] = {"requests": requests}

        def observer(parts):
            metadata, jpeg = (bytes(part) for part in parts)
            path = writer.add_bytes(
                "sam_request_jpeg_q90",
                f"{prefix}_head_sam_request_q90.jpg",
                jpeg,
                media_type="image/jpeg",
            )
            metadata_path = writer.add_bytes(
                "sam_request_000_wire_metadata",
                f"{prefix}_head_sam_request_000_metadata.json",
                metadata,
                media_type="application/json",
            )
            requests.append(
                {
                    "request_artifact_path": path.name,
                    "wire_metadata_artifact_path": metadata_path.name,
                    "wire_metadata": json.loads(metadata),
                    "wire_metadata_sha256": hashlib.sha256(
                        metadata
                    ).hexdigest(),
                    "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                    "outcome": "prepared_before_send",
                }
            )

        try:
            client.segment(
                image,
                frame_id=20,
                timestamp=126.0,
                jpeg_quality=grasp.SAM_REQUEST_JPEG_QUALITY,
                request_observer=observer,
            )
            raise AssertionError("SAM timeout was ignored")
        except TimeoutError as exc:
            requests[0]["outcome"] = "segment_failed"
            journal_path = writer.fail(exc)

        assert socket.sent is not None
        request_path = root / requests[0]["request_artifact_path"]
        assert request_path.read_bytes() == socket.sent[1]
        metadata_path = root / requests[0]["wire_metadata_artifact_path"]
        assert metadata_path.read_bytes() == socket.sent[0]
        assert not (root / f"{prefix}_head_observation.json").exists()
        assert journal_path.exists()
        assert not writer.reservation_path.exists()
        journal = json.loads(journal_path.read_bytes())
        assert journal["schema"] == "sam_head_observation_failure/v1"
        assert journal["status"] == "failed"
        assert journal["run_id"] == "timeout-run"
        assert journal["attempt_id"] == "timeout-attempt"
        assert (
            journal["context"]["sam_transport"]["requests"][0][
                "jpeg_sha256"
            ]
            == hashlib.sha256(socket.sent[1]).hexdigest()
        )


class _FailingResource:
    def __init__(self, events, label, error=None):
        self.events = events
        self.label = label
        self.error = error

    def _run(self):
        self.events.append(self.label)
        if self.error is not None:
            raise self.error

    def set(self):
        self._run()

    def stop(self):
        self._run()

    def close(self, linger=0):
        assert linger == 0
        self._run()

    def term(self):
        self._run()


def test_stop_attempts_every_resource_and_preserves_primary_failure():
    events = []
    cleanup_notes = []
    original_note_exception = grasp._note_exception
    grasp._note_exception = lambda error, note: cleanup_notes.append(
        (error, note)
    )
    camera_error = RuntimeError("camera stop failed")
    runner = object.__new__(grasp.LiveSamGrasp)
    runner.stop_event = _FailingResource(events, "event")
    runner.camera = _FailingResource(
        events, "camera", error=camera_error
    )
    runner.sam = _FailingResource(
        events, "sam", error=RuntimeError("SAM close failed")
    )
    runner.rpc = SimpleNamespace(
        socket=_FailingResource(events, "rpc socket"),
        context=_FailingResource(
            events,
            "rpc context",
            error=RuntimeError("RPC context failed"),
        ),
    )
    try:
        try:
            runner.stop()
            raise AssertionError("cleanup failures were ignored")
        except RuntimeError as exc:
            assert exc is camera_error
        assert events == [
            "event",
            "camera",
            "sam",
            "rpc socket",
            "rpc context",
        ]
        note_text = "\n".join(note for _, note in cleanup_notes)
        assert "SAM client cleanup also failed" in note_text
        assert "RPC context cleanup also failed" in note_text

        class FailingRunner:
            def stop(self):
                raise RuntimeError("final stop failed")

        primary = ValueError("primary observation failure")
        grasp._stop_runner_without_masking(FailingRunner(), primary)
        assert cleanup_notes[-1][0] is primary
        assert "final stop failed" in cleanup_notes[-1][1]
        try:
            grasp._stop_runner_without_masking(FailingRunner(), None)
            raise AssertionError("sole cleanup failure was ignored")
        except RuntimeError as exc:
            assert "final stop failed" in str(exc)
    finally:
        grasp._note_exception = original_note_exception


def test_candidate_image_margin_rejects_mask_or_box_clipping():
    shape = (100, 120)
    interior = np.zeros(shape, bool)
    interior[20:81, 20:81] = True
    accepted = MaskCandidate(
        interior,
        np.array([20.0, 20.0, 81.0, 81.0]),
        0.9,
    )
    report = grasp._require_candidate_image_margin(
        accepted, shape, label="lid"
    )
    assert report["required_margin_px"] == 10
    assert report["mask_margin_px"] >= 10
    assert report["bbox_margin_px"] >= 10

    clipped_mask = interior.copy()
    clipped_mask[99, 50] = True
    mask_clipped = MaskCandidate(
        clipped_mask,
        np.array([20.0, 20.0, 81.0, 81.0]),
        0.9,
    )
    try:
        grasp._require_candidate_image_margin(
            mask_clipped, shape, label="lid"
        )
        raise AssertionError("boundary-touching lid mask was accepted")
    except RuntimeError as exc:
        assert "clipped" in str(exc)
        assert "mask_margin=0px" in str(exc)

    box_clipped = MaskCandidate(
        interior,
        np.array([20.0, 20.0, 119.0, 99.0]),
        0.9,
    )
    try:
        grasp._require_candidate_image_margin(
            box_clipped, shape, label="gripper"
        )
        raise AssertionError("boundary-touching gripper box was accepted")
    except RuntimeError as exc:
        assert "selected gripper is clipped" in str(exc)
        assert "clipped" in str(exc)
        assert "bbox_margin=1.0px" in str(exc)

    observed = np.zeros((1440, 1920), bool)
    observed[1317:1440, 1469:1656] = True
    observed_candidate = MaskCandidate(
        observed,
        np.array([1469.0, 1317.0, 1655.0, 1439.0]),
        0.9,
    )
    try:
        grasp._require_candidate_image_margin(
            observed_candidate,
            observed.shape,
            label="lid",
        )
        raise AssertionError("observed clipped lid was accepted")
    except RuntimeError as exc:
        assert "mask_margin=0px" in str(exc)


def test_roi_margin_and_gripper_association_fail_closed():
    shape = (120, 160)
    coarse_mask = np.zeros(shape, dtype=bool)
    coarse_mask[45:75, 100:140] = True
    coarse = MaskCandidate(
        coarse_mask,
        np.array([100.0, 45.0, 140.0, 75.0]),
        0.8,
    )
    roi = compute_candidate_roi(
        [coarse],
        full_shape_hw=shape,
        padding_px=25,
        scale=2,
    )

    interior_mask = np.zeros(shape, dtype=bool)
    interior_mask[45:75, 102:138] = True
    interior = MaskCandidate(
        interior_mask,
        np.array([102.0, 45.0, 138.0, 75.0]),
        0.9,
    )
    margin = grasp._require_candidate_roi_margin(
        interior, roi, label="gripper"
    )
    assert margin["mask_margin_px"] >= 10

    edge_mask = interior_mask.copy()
    edge_mask[:, :] = False
    edge_mask[45:75, roi.crop_xyxy[0] : 110] = True
    edge = MaskCandidate(
        edge_mask,
        np.array(
            [float(roi.crop_xyxy[0]), 45.0, 110.0, 75.0]
        ),
        0.95,
    )
    try:
        grasp._require_candidate_roi_margin(edge, roi, label="gripper")
        raise AssertionError("ROI-edge-truncated gripper was accepted")
    except RuntimeError as exc:
        assert "ROI boundary" in str(exc)

    decoy_mask = np.zeros(shape, dtype=bool)
    decoy_mask[45:75, 50:90] = True
    decoy = MaskCandidate(
        decoy_mask,
        np.array([50.0, 45.0, 90.0, 75.0]),
        0.99,
    )
    associated, report = grasp._associate_fine_gripper(
        coarse, (decoy, interior)
    )
    assert associated is interior
    assert report["fine_candidate_index"] == 1
    assert report["mask_iou"] > 0.5
    try:
        grasp._associate_fine_gripper(coarse, (decoy,))
        raise AssertionError("unassociated fine gripper was accepted")
    except RuntimeError as exc:
        assert "coarse gripper instance" in str(exc)


def test_post_sam_roi_consistency_metrics_reject_scene_change():
    before = np.zeros((80, 100, 3), dtype=np.uint8)
    nearly_same = before.copy()
    nearly_same[10, 10] = 10
    report = grasp._compare_roi_images(before, nearly_same)
    assert report["accepted"] is True

    changed = before.copy()
    changed[:, 40:] = 255
    report = grasp._compare_roi_images(before, changed)
    assert report["accepted"] is False
    assert report["mean_abs_difference"] > 0.5


def test_pre_sam_depth_burst_tolerates_startup_polling_but_gates_scene():
    source_rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    source_depth = np.ones((12, 16), dtype=np.float32)
    frames = [
        (source_rgb.copy(), 100.01 + index * 0.01, source_depth.copy())
        for index in range(3)
    ]
    for index in range(7):
        fresh_depth = source_depth.copy()
        fresh_depth[index, index] += (index + 1) * 0.001
        frames.append(
            (
                source_rgb.copy(),
                100.38 + index * 0.38,
                fresh_depth,
            )
        )

    class Camera:
        def __init__(self, values):
            self.values = list(values)

        def get_latest_frame(self):
            if not self.values:
                return frames[-1]
            return self.values.pop(0)

    runner = object.__new__(grasp.LiveSamGrasp)
    runner.args = SimpleNamespace(depth_frames=15)
    runner.camera = Camera(frames)
    depth_frames, timestamps, _, timing = runner._collect_depth_burst(
        source_rgb,
        100.0,
        source_depth,
    )
    assert len(depth_frames) == 7
    assert 2.2 < timestamps[-1] - timestamps[0] < 2.5
    assert timing["maximum_timestamp_span_s"] == 2.5
    assert timing["rgb_consistency"]["accepted"] is True
    assert timing["repeated_rgb_digest_count"] == 9
    assert timing["repeated_depth_digest_count"] == 3
    assert timing["timestamp_advanced_count"] == 9
    assert timing["timestamp_beyond_span_count"] == 1
    assert len(timing["source_depth_sha256"]) == len(depth_frames)

    changed_rgb = source_rgb.copy()
    changed_rgb[:, 12:] = 255
    changed_depth = source_depth.copy()
    changed_depth[0, 0] = 2.0
    runner.args = SimpleNamespace(depth_frames=3)
    runner.camera = Camera(
        [(changed_rgb, 100.1, changed_depth)]
    )
    try:
        runner._collect_depth_burst(source_rgb, 100.0, source_depth)
        raise AssertionError("changed pre-SAM scene was accepted")
    except RuntimeError as exc:
        assert "scene changed" in str(exc)

    class RepeatedDepthCamera:
        def __init__(self):
            self.calls = 0

        def get_latest_frame(self):
            self.calls += 1
            return (
                source_rgb.copy(),
                100.0 + self.calls * 0.1,
                source_depth.copy(),
            )

    diagnostics = {}
    runner.args = SimpleNamespace(depth_frames=15)
    runner.camera = RepeatedDepthCamera()
    try:
        runner._collect_depth_burst(
            source_rgb,
            100.0,
            source_depth,
            diagnostics=diagnostics,
        )
        raise AssertionError("repeated stale depth was accepted")
    except RuntimeError as exc:
        message = str(exc)
        assert "accepted=1/7" in message
        assert "repeated_rgb=" in message
        assert "repeated_depth=" in message
        assert "accepted_timestamps=[100.0]" in message
    assert diagnostics["frames_accepted"] == 1
    assert diagnostics["repeated_rgb_digest_count"] >= 1
    assert diagnostics["repeated_depth_digest_count"] >= 1
    assert diagnostics["accepted_timestamps"] == [100.0]
    json.dumps(diagnostics, allow_nan=False)

    failing_runner = object.__new__(grasp.LiveSamGrasp)
    failing_runner.args = SimpleNamespace(depth_frames=15)
    failing_runner.camera = RepeatedDepthCamera()
    failing_runner.run_id = "depth-failure-run"
    failing_runner.sequence = 0
    failing_runner.last_observation_artifacts = None
    failing_runner._await_fresh_head_frame = lambda **_: (
        source_rgb.copy(),
        100.0,
        source_depth.copy(),
    )
    with tempfile.TemporaryDirectory() as temporary:
        failing_runner.output_dir = Path(temporary)
        try:
            failing_runner.observe(0.04)
            raise AssertionError("depth failure journal was not written")
        except RuntimeError:
            pass
        journal_path = (
            failing_runner.output_dir
            / "000_head_observation.failed.json"
        )
        journal = json.loads(journal_path.read_bytes())
        failure_diagnostics = journal["context"]["depth_burst"]
        assert failure_diagnostics["frames_accepted"] == 1
        assert failure_diagnostics["repeated_depth_digest_count"] >= 1
        assert failure_diagnostics["accepted_timestamps"] == [100.0]


def test_mocked_live_observation_runs_coarse_to_fine_with_provenance():
    image_shape = (160, 200)
    camera_rgb_shape = (200, 160, 3)
    source_rgb = np.zeros(camera_rgb_shape, dtype=np.uint8)
    post_rgb = source_rgb.copy()
    post_rgb[10, 10, 0] = 1
    depth = np.full(camera_rgb_shape[:2], 0.8, dtype=np.float32)

    def full_candidate(x0, y0, x1, y1, score):
        mask = np.zeros(image_shape, dtype=bool)
        mask[y0:y1, x0:x1] = True
        return MaskCandidate(
            mask,
            np.array([x0, y0, x1, y1], dtype=float),
            score,
        )

    coarse_lid = full_candidate(40, 50, 80, 90, 0.90)
    coarse_gripper = full_candidate(130, 60, 170, 100, 0.88)

    class FakeSam:
        def __init__(self):
            self.calls = 0

        def segment(
            self,
            image_bgr,
            *,
            frame_id,
            timestamp,
            prompt,
            confidence_threshold,
            jpeg_quality,
            request_observer,
        ):
            if self.calls == 0:
                assert runner.camera.frames == []
            request_observer(
                tuple(
                    encode_request(
                        image_bgr,
                        frame_id=frame_id,
                        timestamp=timestamp,
                        prompt=prompt,
                        confidence_threshold=confidence_threshold,
                        jpeg_quality=jpeg_quality,
                    )
                )
            )
            call = self.calls
            self.calls += 1
            if call == 0:
                candidates = (coarse_lid,)
            elif call == 1:
                candidates = (coarse_gripper,)
            else:
                height, width = image_bgr.shape[:2]
                assert (height, width) == (640, 800)

                def roi_candidate(x0, y0, x1, y1, score):
                    mask = np.zeros((height, width), dtype=bool)
                    mask[y0 * 4 : y1 * 4, x0 * 4 : x1 * 4] = True
                    return MaskCandidate(
                        mask,
                        np.array(
                            [x0 * 4, y0 * 4, x1 * 4, y1 * 4],
                            dtype=float,
                        ),
                        score,
                    )

                if call == 2:
                    candidates = (roi_candidate(42, 52, 78, 88, 0.96),)
                elif call == 3:
                    candidates = (
                        roi_candidate(80, 60, 110, 95, 0.99),
                        roi_candidate(132, 62, 168, 98, 0.94),
                    )
                else:
                    raise AssertionError("unexpected SAM request")
            return SegmentationResult(
                frame_id=frame_id,
                source_timestamp=timestamp,
                model="fake-sam",
                inference_ms=1.0,
                candidates=candidates,
            )

    burst_frames = []
    for index in range(2):
        rgb = source_rgb.copy()
        rgb[index, index, 1] = index + 1
        next_depth = depth.copy()
        next_depth[index, index] += (index + 1) * 0.001
        burst_frames.append(
            (rgb, 100.01 + index * 0.01, next_depth)
        )

    class BurstCamera:
        def __init__(self):
            self.frames = list(burst_frames)

        def get_latest_frame(self):
            if not self.frames:
                return burst_frames[-1]
            return self.frames.pop(0)

    runner = object.__new__(grasp.LiveSamGrasp)
    runner.args = SimpleNamespace(depth_frames=3)
    runner.camera = BurstCamera()
    runner.sam = FakeSam()
    runner.run_id = "mock-live-run"
    runner.sequence = 0
    runner.frame_id = 1000
    runner.previous_lid_center = None
    runner.previous_gripper_center = None
    runner.last_head_timestamp = None
    runner.last_head_rgb_sha256 = None
    runner.head_camera_matrix = np.array(
        [[300.0, 0.0, 100.0], [0.0, 300.0, 80.0], [0.0, 0.0, 1.0]]
    )
    runner.head_camera_reference_shape = image_shape
    runner.support_plane_config = {}
    runner.geometry_quality_config = {}
    runner.proximity_warning_m = 0.02
    runner.last_target_3d = None
    runner.last_geometry_quality = None
    runner.last_depth = None
    runner.last_camera_matrix = None
    runner.last_proximity_m = None
    runner.last_observation_artifacts = None
    fresh_frames = iter(
        [
            (source_rgb.copy(), 100.0, depth.copy()),
            (post_rgb.copy(), 101.0, depth.copy()),
        ]
    )
    runner._await_fresh_head_frame = lambda **_: next(fresh_frames)

    original = {
        "choose_lid_candidate": grasp.choose_lid_candidate,
        "lid_left_grasp_px": grasp.lid_left_grasp_px,
        "estimate_target_on_support_plane": (
            grasp.estimate_target_on_support_plane
        ),
        "assess_target_geometry": grasp.assess_target_geometry,
        "scene_feature": grasp.scene_feature,
        "nearest_scene_distance": grasp.nearest_scene_distance,
        "render_scene": grasp.render_scene,
    }

    def choose_lid(candidates, **_):
        candidate = tuple(candidates)[0]
        yy, xx = np.nonzero(candidate.mask)
        geometry = SimpleNamespace(
            center_px=np.array([np.mean(xx), np.mean(yy)]),
            radius_px=20.0,
            contour=np.column_stack((xx, yy))[:20],
            area_px=float(len(xx)),
            circularity=0.8,
        )
        return candidate, geometry

    observed_gripper_candidate_counts = []

    def fake_scene_feature(
        *,
        gripper_candidates,
        selected_lid,
        **_,
    ):
        gripper_candidates = tuple(gripper_candidates)
        observed_gripper_candidate_counts.append(len(gripper_candidates))
        lid_candidate, lid_geometry = selected_lid
        return SimpleNamespace(
            lid_candidate=lid_candidate,
            lid_geometry=lid_geometry,
            gripper_candidate=gripper_candidates[0],
            lid_grasp_feature=np.array([40.0, 70.0, 760.0]),
            gripper_feature=np.array([168.0, 80.0, 800.0]),
        )

    grasp.choose_lid_candidate = choose_lid
    grasp.lid_left_grasp_px = lambda *_: np.array([42.0, 70.0])
    grasp.estimate_target_on_support_plane = lambda *_args, **_kwargs: (
        SimpleNamespace(
            confidence=0.9,
            pixel_xy=np.array([42.0, 70.0]),
            point_camera_xyz_m=np.array([0.0, 0.0, 0.8]),
            plane=SimpleNamespace(
                normal=np.array([0.0, 0.0, 1.0]),
                offset=-0.8,
                inlier_fraction=0.9,
                rms_m=0.001,
            ),
            support_sample_count=100,
        )
    )
    grasp.assess_target_geometry = lambda *_args, **_kwargs: SimpleNamespace(
        accepted=True,
        view_angle_deg=10.0,
        native_pixel_footprint_x_m=0.001,
        native_pixel_footprint_y_m=0.001,
        reasons=(),
    )
    grasp.scene_feature = fake_scene_feature
    grasp.nearest_scene_distance = lambda *_: 0.1
    grasp.render_scene = lambda image, *_: image.copy()

    try:
        with tempfile.TemporaryDirectory() as temporary:
            runner.output_dir = Path(temporary)
            feature, _, path, returned_timestamp = runner.observe(0.04)
            assert returned_timestamp == 100.0
            assert Path(path).exists()
            assert feature.gripper_candidate.box_xyxy[0] == 132.0
            assert observed_gripper_candidate_counts == [1]
            assert runner.sam.calls == 4

            manifest_path = Path(
                runner.last_observation_artifacts["manifest"]
            )
            document = json.loads(manifest_path.read_bytes())
            refinement = document["roi_refinement"]
            association = refinement["gripper_selected_candidate"][
                "coarse_association"
            ]
            assert association["fine_candidate_index"] == 1
            assert association["mask_iou"] > 0.5
            assert refinement["post_inference_scene_consistency"][
                "accepted"
            ]
            assert document["depth"]["captured_before_sam"] is True
            assert document["depth"]["timestamp_span_s"] <= 0.5
            assert len(document["sam_transport"]["requests"]) == 4
            for request in document["sam_transport"]["requests"]:
                metadata_path = manifest_path.parent / request[
                    "wire_metadata_artifact_path"
                ]
                assert (
                    hashlib.sha256(metadata_path.read_bytes()).hexdigest()
                    == request["wire_metadata_sha256"]
                )
            assert (
                refinement["lid_selected_candidate"][
                    "raw_roi_mask_artifact_role"
                ]
                in document["files"]
            )
            assert (
                refinement["gripper_selected_candidate"][
                    "raw_roi_mask_artifact_role"
                ]
                in document["files"]
            )
    finally:
        for name, value in original.items():
            setattr(grasp, name, value)


if __name__ == "__main__":
    test_exact_wire_jpeg_is_returned()
    test_request_observer_runs_before_send_and_survives_timeout()
    test_request_observer_failure_prevents_send()
    test_bundle_round_trip_and_canonical_manifest()
    test_reservation_is_collision_free_and_final_files_never_overwrite()
    test_exclusive_publish_failure_is_loud_and_preserves_target()
    test_timeout_writes_failure_journal_with_exact_wire_evidence()
    test_stop_attempts_every_resource_and_preserves_primary_failure()
    test_candidate_image_margin_rejects_mask_or_box_clipping()
    test_roi_margin_and_gripper_association_fail_closed()
    test_post_sam_roi_consistency_metrics_reject_scene_change()
    test_pre_sam_depth_burst_tolerates_startup_polling_but_gates_scene()
    test_mocked_live_observation_runs_coarse_to_fine_with_provenance()
    print("SAM observation artifact checks passed")
