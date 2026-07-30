import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.capture_record3d_multiview import _load_resume_manifest


def record(path: Path, root: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(root)),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def interrupted_capture(root: Path) -> Path:
    session = root / "session"
    frame_dir = session / "raw" / "head" / "pose_0" / "000000"
    derived = session / "derived" / "pose_0"
    frame_dir.mkdir(parents=True)
    derived.mkdir(parents=True)
    files = {}
    for name in ("rgb.png", "depth.npy", "confidence.npy", "meta.json"):
        path = frame_dir / name
        path.write_bytes((name + "-payload").encode())
        files[name] = record(path, session)
    index = derived / "frames.jsonl"
    preview = derived / "rgb_landscape.png"
    index.write_text("{}\n")
    preview.write_bytes(b"preview")
    view = {
        "name": "pose_0",
        "pose_stability": {"accepted": True},
        "robot_state": {"stability": {"accepted": True}},
        "frames_index": record(index, session),
        "preview": record(preview, session),
        "frames": [{"sequence": 0, "files": files}],
    }
    manifest = {
        "schema": "piper_robot.rgbd_multiview_capture/v1",
        "status": "collecting",
        "commands_sent": False,
        "session_id": "test",
        "purpose": "fixed_head_robot_calibration",
        "operator_action": "move-robot",
        "frames_per_view": 7,
        "view_order": ["pose_0", "pose_1", "pose_holdout"],
        "completed_view_names": ["pose_0"],
        "views": [view],
    }
    (session / "manifest.partial.json").write_text(json.dumps(manifest))
    return session


class CaptureRecord3DResumeTest(unittest.TestCase):
    def test_valid_prefix_is_loaded_without_overwriting_saved_view(self):
        with tempfile.TemporaryDirectory() as directory:
            session = interrupted_capture(Path(directory))
            manifest, views, saved = _load_resume_manifest(
                session,
                requested_views=None,
                frames_per_view=None,
                condition=None,
                operator_action=None,
            )
        self.assertEqual(manifest["session_id"], "test")
        self.assertEqual(views, ["pose_0", "pose_1", "pose_holdout"])
        self.assertEqual([item["name"] for item in saved], ["pose_0"])

    def test_view_order_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            session = interrupted_capture(Path(directory))
            with self.assertRaisesRegex(ValueError, "requested views"):
                _load_resume_manifest(
                    session,
                    requested_views=["pose_1", "pose_0", "pose_holdout"],
                    frames_per_view=None,
                    condition=None,
                    operator_action=None,
                )

    def test_tampered_saved_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            session = interrupted_capture(Path(directory))
            target = session / "raw" / "head" / "pose_0" / "000000" / "rgb.png"
            target.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "byte count changed"):
                _load_resume_manifest(
                    session,
                    requested_views=None,
                    frames_per_view=None,
                    condition=None,
                    operator_action=None,
                )


if __name__ == "__main__":
    unittest.main()
