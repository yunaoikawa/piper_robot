"""WetRobo fiducial/calibration command line tools.

Examples:
  python -m wetrobo.fiducial_cli generate-markers --manifest ... --out markers/
  python -m wetrobo.fiducial_cli calibrate-intrinsics --images 'calib/head/*.png' ...
  python -m wetrobo.fiducial_cli validate-calibration --image bench.png ...
  python -m wetrobo.fiducial_cli author-daily-cad --image bench.png ...
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from wetrobo.perception.calibrate import (
    calibrate_hand_eye, calibrate_intrinsics_charuco, generate_charuco_assets,
)
from wetrobo.perception.fiducials import (
    CalibrationProfile, MarkerManifest, detect_tags, estimate_camera_pose,
    estimate_registered_objects, generate_marker_assets,
)


def _image(path):
    path = Path(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None and path.suffix.lower() in {".heic", ".heif"}:
        with tempfile.NamedTemporaryFile(suffix=".png") as converted:
            try:
                subprocess.run(["heif-convert", str(path), converted.name], check=True,
                               stdout=subprocess.DEVNULL)
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                raise ValueError("HEIC input requires the `heif-convert` command") from exc
            bgr = cv2.imread(converted.name, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + "\n")


def _quality_dict(q):
    return asdict(q)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate-markers")
    g.add_argument("--manifest", required=True); g.add_argument("--out", required=True)

    inspect = sub.add_parser("inspect-image")
    inspect.add_argument("--image", required=True)
    inspect.add_argument("--out", required=True)

    gc = sub.add_parser("generate-charuco")
    gc.add_argument("--out", required=True)

    ci = sub.add_parser("calibrate-intrinsics")
    ci.add_argument("--images", required=True, help="quoted glob")
    ci.add_argument("--camera-id", required=True); ci.add_argument("--out", required=True)

    rm = sub.add_parser("register-mounts")
    rm.add_argument("--manifest", required=True)
    rm.add_argument("--transforms", required=True,
                    help="JSON object: marker id -> measured T_parent_tag 4x4")
    rm.add_argument("--out", required=True)

    he = sub.add_parser("calibrate-hand-eye")
    he.add_argument("--pairs", required=True,
                    help="NPZ with T_base_gripper and T_camera_target arrays")
    he.add_argument("--profile", required=True); he.add_argument("--out", required=True)

    for name in ("validate-calibration", "author-daily-cad"):
        p = sub.add_parser(name)
        p.add_argument("--image", required=True); p.add_argument("--profile", required=True)
        p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
        if name == "author-daily-cad":
            p.add_argument("--catalog", default=None)
            p.add_argument("--mjcf", default=None,
                           help="optional measured daily-CAD MJCF output")

    args = ap.parse_args(argv)
    if args.command == "generate-markers":
        made = generate_marker_assets(MarkerManifest.load(args.manifest), args.out)
        print("\n".join(map(str, made))); return 0
    if args.command == "inspect-image":
        image = _image(args.image)
        detections = detect_tags(image)
        tags = []
        for marker_id, corners in sorted(detections.items()):
            edges = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
            tags.append({"marker_id": marker_id,
                         "center_px": corners.mean(axis=0).round(2).tolist(),
                         "mean_edge_px": float(edges.mean()),
                         "min_edge_px": float(edges.min()),
                         "corners_px": corners.round(2).tolist()})
        artifact = {"schema_version": 1,
                    "inspected_at": datetime.now(timezone.utc).isoformat(),
                    "image": str(Path(args.image).resolve()),
                    "image_sha256": hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
                    "width": image.shape[1], "height": image.shape[0], "tags": tags}
        _write_json(args.out, artifact)
        print(f"detected ids: {[t['marker_id'] for t in tags]}")
        return 0
    if args.command == "generate-charuco":
        made = generate_charuco_assets(args.out)
        print("\n".join(map(str, made)))
        return 0
    if args.command == "calibrate-intrinsics":
        paths = sorted(glob.glob(args.images))
        profile = calibrate_intrinsics_charuco([_image(p) for p in paths], args.camera_id)
        profile.save(args.out); print(f"wrote {args.out} rms={profile.intrinsic_rms_px:.3f}px")
        return 0
    if args.command == "register-mounts":
        manifest = MarkerManifest.load(args.manifest)
        transforms = json.loads(Path(args.transforms).read_text())
        unknown = set(transforms) - {str(m.marker_id) for m in manifest.markers}
        if unknown:
            raise ValueError(f"unknown marker ids: {sorted(unknown)}")
        manifest.markers = [replace(m, T_parent_tag=transforms.get(str(m.marker_id), m.T_parent_tag))
                            for m in manifest.markers]
        manifest.save(args.out); print(f"wrote {args.out} hash={manifest.digest()}")
        return 0
    if args.command == "calibrate-hand-eye":
        pairs = np.load(args.pairs)
        profile = calibrate_hand_eye(pairs["T_base_gripper"], pairs["T_camera_target"],
                                     CalibrationProfile.load(args.profile))
        profile.save(args.out); print(f"wrote {args.out}")
        return 0

    image = _image(args.image); profile = CalibrationProfile.load(args.profile)
    manifest = MarkerManifest.load(args.manifest)
    q = estimate_camera_pose(image, profile, manifest)
    if args.command == "validate-calibration":
        _write_json(args.out, _quality_dict(q)); print(q.reason)
        return 0 if q.accepted else 2

    objects = estimate_registered_objects(image, profile, manifest, q) if q.accepted else []
    image_hash = hashlib.sha256(Path(args.image).read_bytes()).hexdigest()
    artifact = {"schema_version": 1, "condition": "tag_assisted_deployment",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "image": str(Path(args.image).resolve()), "image_sha256": image_hash,
                "calibration_id": profile.calibration_id,
                "marker_manifest_sha256": manifest.digest(), "quality": _quality_dict(q),
                "objects": objects}
    if args.mjcf:
        from wetrobo.perception.catalog import LabwareCatalog
        from bench_verify.scene_graph import BenchState
        catalog = LabwareCatalog.load(args.catalog)
        items = []
        for i, obj in enumerate(objects):
            T = np.asarray(obj["T_robot_object"])
            items.append(catalog.to_item(obj["container"], f"tag-{obj['marker_id']}",
                                         T[:3, 3], T[:3, :3], 1.0))
        state = BenchState("measured_daily_cad", items,
                           captured_by="fiducial_cli.author-daily-cad")
        catalog.to_mjcf(state, args.mjcf)
        artifact["mjcf"] = str(Path(args.mjcf).resolve())
    _write_json(args.out, artifact); print(q.reason)
    return 0 if q.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
