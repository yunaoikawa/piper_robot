#!/usr/bin/env python3
"""Retrospectively score preserved Door configurations with one frozen evaluator.

The evaluator uses an independent open observation and the final closed
calibration capture.  It deliberately leaves D0 and D5 missing: D0 has no
identified post-action RGB-D result, and no new physical opening was executed
after the D5 evidence-only hardening change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY))

from rollout.articulated_appliance import (  # noqa: E402
    build_endpoint_model,
    classify_endpoint_state,
    load_bundle_endpoint,
    load_endpoint,
)
from rollout.incubator_door_plane import estimate_frame  # noqa: E402


PROFILE = "src/configs/pasteur_incubator_door_demo.json"
OPEN_REFERENCE = {
    "image": "data/runs/pasteur/incubator_door_close_20260808T082919Z_observe/head.png",
    "depth": "data/runs/pasteur/incubator_door_close_20260808T082919Z_observe/head_depth.npy",
}
CLOSED_REFERENCE = {
    "bundle": "data/runs/pasteur/incubator_auto_close_20260808_demo/captures/2026-08-08/20260808T100604.838020Z_head_close_result_722eb26e",
}
CONFIGURATIONS = (
    {
        "stage": "D0",
        "label": "generic replay",
        "missing_reason": "no identified post-action RGB-D result for the generic baseline",
    },
    {
        "stage": "D1",
        "label": "relative demo",
        "description": "first uninterrupted contact-relative long-pull result",
        "image": "data/runs/pasteur/incubator_door_20260808T043620Z_open_door_demo_contact/after/head.png",
        "depth": "data/runs/pasteur/incubator_door_20260808T043620Z_open_door_demo_contact/after/head_depth.npy",
    },
    {
        "stage": "D2",
        "label": "checkpointed pull",
        "description": "retry-2 checkpointed-pull slip observation",
        "image": "data/runs/pasteur/incubator_door_20260808T044541Z_retry2_slip_observe/head.png",
        "depth": "data/runs/pasteur/incubator_door_20260808T044541Z_retry2_slip_observe/head_depth.npy",
    },
    {
        "stage": "D3",
        "label": "metric alignment",
        "description": "yaw-aligned pull result after slip recovery",
        "image": "data/runs/pasteur/incubator_door_20260808_retry6_yaw_aligned_slip_observe/head.png",
        "depth": "data/runs/pasteur/incubator_door_20260808_retry6_yaw_aligned_slip_observe/head_depth.npy",
    },
    {
        "stage": "D4",
        "label": "autonomous endpoints",
        "description": "autonomous opening final observation",
        "bundle": "data/runs/pasteur/incubator_auto_open_20260808_demo_retry2/captures/2026-08-08/20260808T100427.298699Z_head_open_attempt_1_result_95dc7b74",
    },
    {
        "stage": "D5",
        "label": "evidence hardening",
        "missing_reason": "no independent physical opening was executed after the evidence-only change",
    },
)


def load_json(path: str | Path) -> dict:
    return json.loads((REPOSITORY / path).read_text())


def digest(path: str | Path) -> str:
    return hashlib.sha256((REPOSITORY / path).read_bytes()).hexdigest()


def source_record(specification: dict, observation) -> dict:
    if "bundle" in specification:
        frame = Path(observation.source)
        paths = [frame / "rgb.png", frame / "depth.npy", frame / "meta.json"]
        return {
            "bundle": specification["bundle"],
            "selected_frame": str(frame.relative_to(REPOSITORY)),
            "files": {
                str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
        }
    return {
        "image": specification["image"],
        "image_sha256": digest(specification["image"]),
        "depth": specification["depth"],
        "depth_sha256": digest(specification["depth"]),
    }


def load_observation(specification: dict):
    if "bundle" in specification:
        return load_bundle_endpoint(REPOSITORY / specification["bundle"])
    return load_endpoint(
        REPOSITORY / specification["image"],
        REPOSITORY / specification["depth"],
    )


def main() -> None:
    profile = load_json(PROFILE)
    settings = profile["state_detection"]
    opened = load_observation(OPEN_REFERENCE)
    closed = load_observation(CLOSED_REFERENCE)
    closed_frame = Path(closed.source)
    metadata = json.loads((closed_frame / "meta.json").read_text())
    tag_config_path = profile["door_plane"]["tag_config"]
    tag_config = load_json(tag_config_path)
    plane = estimate_frame(
        closed.image_bgr,
        closed.depth_m,
        metadata["intrinsics"]["K_rgb_rotated_clockwise"],
        tag_config,
        profile["door_plane"],
    )
    model = build_endpoint_model(
        opened,
        closed,
        settings,
        candidate_mask=plane["depth_inlier_mask"],
    )

    measurements = []
    for specification in CONFIGURATIONS:
        row = {
            "stage": specification["stage"],
            "label": specification["label"],
        }
        if "missing_reason" in specification:
            row.update({"status": "not_measured", "reason": specification["missing_reason"]})
            measurements.append(row)
            continue
        observation = load_observation(specification)
        assessment = classify_endpoint_state(observation, model, settings)
        d_open = float(assessment["relative_open_error"])
        d_closed = float(assessment["relative_closed_error"])
        row.update(
            {
                "status": "measured",
                "description": specification["description"],
                "source": source_record(specification, observation),
                "classified_state": assessment["state"],
                "relative_open_error": d_open,
                "relative_closed_error": d_closed,
                "open_reference_median_absolute_depth_error_mm": d_open * model.endpoint_separation_m * 1000.0,
                "goal_conditioned_endpoint_score": d_closed / (d_open + d_closed),
                "registration_error_tag_lengths": assessment["registration"][
                    "registration_error_tag_lengths"
                ],
                "dynamic_point_count": assessment["dynamic_point_count"],
            }
        )
        measurements.append(row)

    report = {
        "schema": "piper_robot.door_configuration_curve/v1",
        "goal": "open",
        "score_formula": "d_closed / (d_open + d_closed)",
        "plotted_metric": "open_reference_median_absolute_depth_error_mm",
        "depth_error_formula": "1000 * median(abs(registered_depth_m - open_reference_depth_m)) over valid dynamic mask",
        "depth_error_interpretation": "Camera-depth mismatch to the open reference, not door travel distance or a measured opening angle; lower is closer to the reference.",
        "evaluation_policy": (
            "one frozen hardened endpoint evaluator and an independent reference "
            "pair; no interpolation or zero imputation"
        ),
        "evaluator": {
            "profile": PROFILE,
            "profile_sha256": digest(PROFILE),
            "endpoint_code": "rollout/articulated_appliance.py",
            "endpoint_code_sha256": digest("rollout/articulated_appliance.py"),
            "plane_code": "rollout/incubator_door_plane.py",
            "plane_code_sha256": digest("rollout/incubator_door_plane.py"),
            "tag_config": tag_config_path,
            "tag_config_sha256": digest(tag_config_path),
            "candidate_mask": settings.get("candidate_mask"),
            "endpoint_separation_m": model.endpoint_separation_m,
            "dynamic_point_count": int(model.dynamic_mask.sum()),
        },
        "references": {
            "open": source_record(OPEN_REFERENCE, opened),
            "closed": source_record(CLOSED_REFERENCE, closed),
        },
        "configurations": measurements,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
