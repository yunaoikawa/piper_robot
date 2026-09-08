#!/usr/bin/env python3
"""Run a labelled, offline ablation of tool-relative grasp-window methods."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.grasp_ablation import AblationSampleResult, METHODS, select_method
from rollout.grasp_window import (
    assess_grasp_window,
    calibrate_grasp_window,
    render_grasp_window,
)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_mask(spec, image_shape, base):
    if "mask_path" in spec:
        mask = cv2.imread(
            str(_resolve(base, spec["mask_path"])), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise ValueError(f"could not read mask {spec['mask_path']}")
        if mask.shape != image_shape[:2]:
            raise ValueError("mask shape does not match its image")
        return mask > 0
    ellipse = spec.get("ellipse")
    if ellipse is None:
        raise ValueError("sample requires mask_path or ellipse")
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.ellipse(
        mask,
        tuple(int(value) for value in ellipse["center_px"]),
        tuple(int(value) for value in ellipse["axes_px"]),
        float(ellipse.get("angle_deg", 0.0)),
        0,
        360,
        255,
        -1,
    )
    return mask > 0


def _load_sample(spec, base):
    if "image_path" in spec:
        image_path = _resolve(base, spec["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image {image_path}")
        source_path = image_path
    else:
        video_path = _resolve(base, spec["video_path"])
        frame_index = int(spec["frame_index"])
        capture = cv2.VideoCapture(str(video_path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
        finally:
            capture.release()
        if not ok or image is None:
            raise ValueError(
                f"could not read frame {frame_index} from {video_path}"
            )
        source_path = video_path
    return source_path, image, _load_mask(spec, image.shape, base)


def _sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    manifest = _load_json(manifest_path)
    base = manifest_path.parent
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference_path, reference_image, reference_mask = _load_sample(
        manifest["reference"], base
    )
    template, _ = calibrate_grasp_window(
        reference_image, reference_mask, method="HYBRID"
    )

    results = []
    rows = []
    source_paths = [manifest_path, reference_path]
    for sample in manifest["samples"]:
        image_path, image, target_mask = _load_sample(sample, base)
        source_paths.append(image_path)
        predictions = {}
        inference_ms = {}
        hybrid = hybrid_frame = None
        for method in METHODS:
            started = time.perf_counter()
            assessment, frame = assess_grasp_window(
                image, target_mask, template, method=method
            )
            inference_ms[method] = 1000.0 * (time.perf_counter() - started)
            predictions[method] = assessment.allowed_to_close
            if method == "HYBRID":
                hybrid, hybrid_frame = assessment, frame
        sample_id = str(sample["id"])
        overlay_path = output / f"{sample_id}_overlay.png"
        cv2.imwrite(
            str(overlay_path),
            render_grasp_window(
                image, target_mask, template, hybrid, hybrid_frame
            ),
        )
        results.append(
            AblationSampleResult(
                sample_id=sample_id,
                expected_window_ready=bool(sample["expected_window_ready"]),
                predictions=predictions,
                inference_ms=inference_ms,
            )
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(sample_id)}</td>"
            f"<td>{bool(sample['expected_window_ready'])}</td>"
            f"<td>{html.escape(json.dumps(predictions))}</td>"
            f"<td><img src='{html.escape(overlay_path.name)}'></td>"
            "</tr>"
        )

    selection = select_method(results)
    artifact = {
        "schema": "piper_robot.grasp_window_ablation/v1",
        "selected_method": selection.selected_method,
        "selection_reason": selection.selection_reason,
        "template": template.to_dict(),
        "dataset_sha256": _sha256(source_paths),
        "sample_count": len(results),
        "metrics": [
            {
                key: value
                for key, value in metric.items()
                if key != "mean_inference_ms"
            }
            for metric in selection.to_dict()["metrics"]
        ],
        "independent_required_gates": [
            "tool_horizontal",
            "tip_at_support",
            "stable_nonempty_closure",
            "target_follows_verification_lift",
        ],
    }
    artifact_path = Path(args.artifact) if args.artifact else output / "selection.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    (output / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>img{width:480px}td{vertical-align:top}</style>"
        f"<h1>Selected: {html.escape(selection.selected_method)}</h1>"
        "<table border='1'><tr><th>sample</th><th>expected window</th>"
        "<th>predictions</th><th>overlay</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
