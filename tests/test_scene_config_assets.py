#!/usr/bin/env python3
"""Check that fixed-camera placement references survive a fresh clone."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_CONFIGS = (
    REPO_ROOT / "src" / "configs" / "pasteur_lid_scene3d.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for config_path in SCENE_CONFIGS:
    profile = json.loads(config_path.read_text(encoding="utf-8"))
    placement = profile["head_camera_placement_reference"]
    relative_path = Path(placement["rgb_path"])
    assert not relative_path.is_absolute()

    reference_path = (REPO_ROOT / relative_path).resolve()
    reference_path.relative_to(REPO_ROOT.resolve())
    assert reference_path.is_file(), reference_path

    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            relative_path.as_posix(),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    assert tracked.returncode == 0, (
        f"placement reference is not tracked by git: {relative_path}"
    )
    assert sha256_file(reference_path) == placement["rgb_sha256"]

    image = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    assert image is not None, reference_path
    assert list(image.shape[:2]) == profile["head_camera_reference_shape_hw"]

print("scene config placement assets checks passed")
