#!/usr/bin/env python3
"""Audit historical Door logs and plot measured aperture, never pressure.

Default: plot the tracked, source-hashed report (no hardware/private logs).
--rebuild-report: re-extract from preserved local August 8 run JSON files.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/runs/pasteur"
ASSETS = ROOT / "docs/assets/code_as_learning_machine"
REPORT = ASSETS / "door_grasp_signal_report.json"
AUTO = "incubator_auto_open_20260808_demo_retry2"
# These are physical pull attempts, NOT the paper's D1--D4 code snapshots.
TRIALS = [
    ("T1", "First demo pull",
     "incubator_door_20260808T043325Z_close_verify_demo_contact",
     "incubator_door_20260808T043548Z_proof_pull_demo_contact",
     "incubator_door_20260808T043620Z_open_door_demo_contact/after/observation.json",
     "Open not supported"),
    ("T2", "Retry 2",
     "incubator_door_20260808T044420Z_retry2_close_verify",
     "incubator_door_20260808T044442Z_retry2_proof_pull",
     "incubator_door_20260808T044541Z_retry2_slip_observe/observation.json",
     "Open not supported"),
    ("T3", "Retry 3",
     "incubator_door_20260808T044900Z_retry3_close_verify",
     "incubator_door_20260808T044921Z_retry3_proof_pull",
     None, "Endpoint not included"),
    ("T4", "Retry 4",
     "incubator_door_20260808T045946Z_retry4_close_verify",
     "incubator_door_20260808T050021Z_retry4_proof_pull",
     "incubator_door_20260808T050128Z_retry4_slip_observe/observation.json",
     "Slip observation"),
    ("T5", "Retry 5",
     "incubator_door_20260808T050457Z_retry5_close_verify",
     "incubator_door_20260808T050518Z_retry5_proof_pull",
     None, "Endpoint not included"),
    ("T6", "Yaw-aligned retry",
     "incubator_door_20260808_retry6_yaw_aligned_close2",
     "incubator_door_20260808_retry6_yaw_aligned_proof",
     "incubator_door_20260808_retry6_yaw_aligned_slip_observe/observation.json",
     "Open observed; grasp lost"),
    ("T7", "Autonomous opening",
     AUTO + "/06_close-verify/motion", AUTO + "/07_proof-pull/motion",
     AUTO + "/10_recover-empty-close/motion/before/observation.json",
     "Open verified; grasp lost"),
]


def checked_samples(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Invalid or missing measured aperture samples")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Aperture outside calibrated open-ratio range")
    return values


def pose_error(pose, reference):
    pose, reference = np.asarray(pose, dtype=float), np.asarray(reference, dtype=float)
    if pose.shape != (7,) or reference.shape != (7,) or not np.isfinite([pose, reference]).all():
        raise ValueError("Expected finite wxyz_xyz poses")
    q, r = pose[:4], reference[:4]
    if min(np.linalg.norm(q), np.linalg.norm(r)) <= 0:
        raise ValueError("Zero quaternion")
    cosine = np.clip(abs(q @ r) / (np.linalg.norm(q) * np.linalg.norm(r)), 0, 1)
    return float(np.linalg.norm(pose[4:] - reference[4:]) * 1000), float(np.degrees(2 * np.arccos(cosine)))


def build_demo_comparison():
    """Compare saved measured contact poses, without fitting any alignment."""
    import h5py  # Only required to re-audit local reference recordings.

    path = ROOT / "data/reference/pasteur/incubator/compiled_door_open_v1.json"
    compiled = json.loads(path.read_text())
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()}
    reference = compiled["medoid"]["contact_pose_wxyz_xyz"]
    gripper_audit = []
    for episode in compiled["successes"]:
        path = ROOT / "data/reference/pasteur/incubator/incoming/door_open" / (episode["stem"] + ".hdf5")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != episode["hdf5_sha256"]:
            raise ValueError(f"Demo source changed: {path}")
        sources[str(path.relative_to(ROOT))] = digest
        with h5py.File(path) as recording:
            gripper_audit.append({"stem": episode["stem"],
                                  "unique_gripper_values": np.unique(recording["right_gripper"][:]).tolist()})
    labels = {close + "/contact_state.json": ident for ident, _, close, *_ in TRIALS}
    rows = []
    for path in sorted(RUNS.glob("incubator*20260808*/**/contact_state.json")):
        raw = path.read_bytes()
        state = json.loads(raw)
        relative = str(path.relative_to(RUNS))
        position, orientation = pose_error(state["contact_pose_wxyz_xyz"], reference)
        rows.append({"trial": labels.get(relative), "timestamp_s": state["before"]["timestamp_s"],
                     "source": relative, "contact_pose_wxyz_xyz": state["contact_pose_wxyz_xyz"],
                     "position_difference_mm": position, "orientation_difference_deg": orientation})
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    rows.sort(key=lambda row: row["timestamp_s"])
    early = 0
    for row in rows:
        if row["trial"] is None:
            early += 1
            row["trial"] = f"E{early}"
    return {
        "reference_stem": compiled["medoid"]["stem"],
        "reference_contact_pose_wxyz_xyz": reference,
        "metric": "Euclidean EE-origin position difference and full SO(3) angular difference at saved contact pose",
        "frame": "Saved robot base coordinates; no registration for physical door displacement",
        "limitations": [
            "Not object-relative handle error, not pressure, not full-trajectory similarity.",
            "The reference medoid was also used to generate the controller trajectory; this is not held-out evaluation.",
            "All ten preserved August 8 contact records are shown chronologically, with no monotonic smoothing.",
            "Later successful trials need not minimize this absolute-demo discrepancy after live alignment.",
            "Demo gripper values are binary (all twelve recordings); they cannot be compared as measured aperture curves.",
        ],
        "demo_gripper_audit": gripper_audit, "trials": rows, "source_sha256": sources,
    }


def build_report():
    sources = {}

    def read(path):
        path = RUNS / path
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    rows = []
    for ident, label, close_dir, proof_dir, post_path, outcome in TRIALS:
        close = read(close_dir + "/contact_state.json")
        proof = read(proof_dir + "/proof_state.json")
        close_samples = checked_samples(close["aperture_samples"])
        proof_samples = checked_samples(proof["proof_aperture_samples"])
        # Assert that the explicit trial pairing links the same contact state.
        np.testing.assert_allclose(close["contact_pose_wxyz_xyz"],
                                   proof["contact_pose_wxyz_xyz"], atol=1e-12)
        post = read(post_path) if post_path else None
        rows.append({
            "trial": ident, "label": label, "outcome_note": outcome,
            "close_source": close_dir + "/contact_state.json",
            "proof_source": proof_dir + "/proof_state.json",
            "post_pull_source": post_path,
            "close_samples": close_samples.tolist(),
            "preclose_measured_aperture": float(close["before"]["right_gripper"]),
            "post_proof_samples": proof_samples.tolist(),
            "closed_aperture": float(close["closed_aperture"]),
            "post_proof_median": float(np.median(proof_samples)),
            "post_pull_aperture": None if post is None else float(post["right_gripper"]),
            "post_pull_timestamp_s": None if post is None else post["timestamp_s"],
            "stable_nonempty": bool(close["stable_nonempty"]),
            "proof_retained": bool(proof["proof_retained"]),
        })

    # Include excluded early contact-only attempts in the audit, not as pull trials.
    included = {row["close_source"] for row in rows}
    contact_only = []
    for path in sorted(RUNS.glob("incubator*20260808*/**/contact_state.json")):
        relative = str(path.relative_to(RUNS))
        if relative not in included:
            d = read(relative)
            contact_only.append({"source": relative,
                                 "closed_aperture": d["closed_aperture"],
                                 "aperture_samples": checked_samples(d["aperture_samples"]).tolist(),
                                 "sample_count": len(d["aperture_samples"]),
                                 "stable_nonempty": d["stable_nonempty"]})

    fields = Counter()

    def inspect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if any(word in key.lower() for word in
                       ("pressure", "force", "torque", "current", "effort")):
                    fields[key] += 1
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    files = sorted(RUNS.glob("incubator*20260808*/**/*.json"))
    for path in files:
        inspect(json.loads(path.read_text()))
    # Keep endpoint outcome provenance distinct from aperture-based evidence.
    read(AUTO + "/12_state_open_attempt_1_result/state.json")
    result = {
        "schema": "door_grasp_signal_audit/v1",
        "measurement": "measured right-gripper normalized opening (0 closed, 1 open)",
        "is_pressure": False,
        "units": "dimensionless open ratio; no force or pressure conversion",
        "scope": "Seven preserved August 8 physical pull attempts, not D1--D4 configurations or teleop demos",
        "limitations": [
            "No continuous calibrated contact-pressure or gripper-current trace found in scoped run JSON.",
            "Arm joint torque snapshots/warning summaries are not contact pressure.",
            "Within-block samples have no individual timestamps: plots use sample index, not elapsed seconds.",
            "Proof samples were collected AFTER the 5 mm motion, not continuously during that motion.",
            "Full-pull continuous samples and complete checkpoint histories are unavailable here.",
            "Post-pull observations do not identify the exact slip time; missing values are not filled.",
            "Opening can succeed even if the gripper later loses the door; aperture is not the door-state evaluator.",
        ],
        "signal_field_audit": {"json_file_count": len(files),
                               "matching_key_occurrences": dict(sorted(fields.items()))},
        "empty_aperture_reference": 0.02,
        "empty_aperture_reference_note": "Controller's configured empty-aperture bound, not pressure threshold or universal grasp-success criterion",
        "trials": rows,
        "contact_only_attempts_not_in_pull_plot": contact_only,
        "endpoint_context": "docs/PASTEUR_INCUBATOR_DOOR_OPENING_RETROSPECTIVE.md and door_configuration_curve_report.json",
        "source_sha256": sources,
        "demo_comparison": build_demo_comparison(),
    }
    result["three_patterns"] = build_three_patterns(result)
    result["successful_opening"] = build_successful_opening(result)
    result["outcome_overlay"] = build_outcome_overlay(result)
    return result


def build_outcome_overlay(report):
    from PIL import Image

    path = ASSETS / "door_configuration_curve_report.json"
    endpoint_report = json.loads(path.read_text())
    endpoints = {r["stage"]: r for r in endpoint_report["configurations"]}
    trials = {r["trial"]: r for r in report["trials"]}
    cases = []
    for trial, historical_stage in [("T1", "D1"), ("T2", "D2"), ("T6", "D3"), ("T7", "D4")]:
        source = endpoints[historical_stage]["source"]
        raw = "selected_frame" in source
        image_path = source["selected_frame"] + "/rgb.png" if raw else source["image"]
        expected = source["files"][image_path] if raw else source["image_sha256"]
        if hashlib.sha256((ROOT / image_path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Image hash mismatch: {image_path}")
        target = ASSETS / f"door_grasp_overlay_{trial}_head.jpg"
        with Image.open(ROOT / image_path) as image:
            image = image.convert("RGB")
            if raw:
                image = image.transpose(Image.Transpose.ROTATE_270)
            image.thumbnail((768, 576))
            image.save(target, quality=92)
        row = trials[trial]
        cases.append({
            "trial": trial, "door_state": endpoints[historical_stage]["classified_state"],
            "aperture": [row["preclose_measured_aperture"], row["closed_aperture"],
                         row["post_proof_median"], row["post_pull_aperture"]],
            "image_source": image_path, "image_sha256": expected,
            "display_image": str(target.relative_to(ROOT)),
            "display_image_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "rotation_clockwise_deg": 90 if raw else 0,
        })
    return {"cases": cases,
            "endpoint_report": str(path.relative_to(ROOT)),
            "endpoint_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selection": "All four pull attempts with endpoints re-evaluated by the frozen classifier: two closed, two open.",
            "limitations": ["Start values are measured, not imputed ones.",
                            "Dashed connections between observation phases are visual guides, not sampled trajectories.",
                            "No line crosses the missing full-pull interval. Endpoint dots overlap at their actual values.",
                            "Observed open endpoints are task successes, not continuously retained grasps."]}


def build_successful_opening(report):
    from PIL import Image

    initial_path = RUNS / AUTO / "02_state_initial/state.json"
    final_path = RUNS / AUTO / "12_state_open_attempt_1_result/state.json"
    initial, final = json.loads(initial_path.read_text()), json.loads(final_path.read_text())
    if (initial["state"], final["state"]) != ("closed", "open"):
        raise ValueError("Historical success endpoints changed")
    source = Path(initial["source"]) / "rgb.png"
    target = ASSETS / "door_grasp_success_T7_initial.jpg"
    with Image.open(source) as image:
        image = image.convert("RGB").transpose(Image.Transpose.ROTATE_270)
        image.thumbnail((768, 576))
        image.save(target, quality=92)
    end_case = next(c for c in report["three_patterns"]["cases"] if c["trial"] == "T7")
    return {
        "trial": "T7", "task_goal": "Open the door", "task_success": True,
        "continuous_grasp_success": False,
        "initial_state": initial["state"], "final_state": final["state"],
        "initial_display_image": str(target.relative_to(ROOT)),
        "final_display_image": end_case["display_image"],
        "initial_source_image": str(source.relative_to(ROOT)),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [initial_path, final_path, source, target]},
        "display_note": "Both full-frame RGB images rotated clockwise 90 degrees and downscaled, not mirrored.",
    }


def build_three_patterns(report):
    """Preserve measured cases and display thumbnails, not synthetic curves."""
    from PIL import Image

    endpoint_path = ASSETS / "door_configuration_curve_report.json"
    endpoints = json.loads(endpoint_path.read_text())
    by_stage = {row["stage"]: row for row in endpoints["configurations"]}
    by_trial = {row["trial"]: row for row in report["trials"]}
    early = report["demo_comparison"]["trials"][1]
    assert early["trial"] == "E2"
    early_samples = next(row for row in report["contact_only_attempts_not_in_pull_plot"]
                         if row["source"] == early["source"])
    cases = [
        {"trial": "E2", "title": "A. Empty close",
         "samples": early_samples["aperture_samples"],
         "closed": early_samples["closed_aperture"], "proof": None, "post_pull": None,
         "door_state": "Not evaluated here; stopped before pull",
         "image_source": str(Path("data/runs/pasteur") / early["source"]).replace("contact_state.json", "after/head.png"),
         "rotation_clockwise_deg": 0,
         "interpretation": "Gripper approaches full closure without a retained object."},
    ]
    for trial, historical_stage, title in [
        ("T1", "D1", "B. Grasp lost; door still closed"),
        ("T7", "D4", "C. Door open; grasp lost"),
    ]:
        row = by_trial[trial]
        endpoint = by_stage[historical_stage]
        source = endpoint["source"]
        raw_bundle = "selected_frame" in source
        image_source = source["selected_frame"] + "/rgb.png" if raw_bundle else source["image"]
        expected_hash = source["files"][image_source] if raw_bundle else source["image_sha256"]
        if hashlib.sha256((ROOT / image_source).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Endpoint image changed: {image_source}")
        cases.append({
            "trial": trial, "title": title, "samples": row["close_samples"],
            "closed": row["closed_aperture"], "proof": row["post_proof_median"],
            "post_pull": row["post_pull_aperture"],
            "door_state": endpoint["classified_state"],
            "endpoint_evaluation_source": str(endpoint_path.relative_to(ROOT)),
            "endpoint_evaluation_sha256": hashlib.sha256(endpoint_path.read_bytes()).hexdigest(),
            "endpoint_historical_key_not_trial_number": historical_stage,
            "image_source": image_source, "rotation_clockwise_deg": 90 if raw_bundle else 0,
            "interpretation": "Final aperture alone cannot distinguish B from C; registered RGB-D establishes the door endpoint.",
        })
    for case in cases:
        source = ROOT / case["image_source"]
        case["image_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        thumbnail = ASSETS / f"door_grasp_pattern_{case['trial']}_head.jpg"
        with Image.open(source) as image:
            image = image.convert("RGB")
            if case["rotation_clockwise_deg"] == 90:
                image = image.transpose(Image.Transpose.ROTATE_270)
            image.thumbnail((768, 576))
            image.save(thumbnail, quality=92)
        case["display_image"] = str(thumbnail.relative_to(ROOT))
        case["display_image_sha256"] = hashlib.sha256(thumbnail.read_bytes()).hexdigest()
    return {"cases": cases, "limitations": [
        "Representative observed patterns, not schematic or complete continuous pull traces.",
        "Images are later observations, not necessarily simultaneous with the aperture samples.",
        "Case C confirms an open endpoint and absent grasp, not the precise temporal order of opening and slip.",
        "Display images are full-frame downscaled only; T7 is rotated clockwise as in the endpoint loader.",
    ]}


def save(fig, stem):
    for ext in ("png", "svg"):
        path = ASSETS / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, metadata={"Date": None} if ext == "svg" else {})
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def plot_summary(report):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.8), sharey=True,
                             layout="constrained")
    colors = ["#2878B5", "#E38B2C", "#9145A3"]
    for ax, trial in zip(axes.flat, report["trials"]):
        vals = [trial["closed_aperture"], trial["post_proof_median"],
                trial["post_pull_aperture"]]
        for x, val in enumerate(vals):
            if val is not None:
                ax.scatter(x, val, color=colors[x], s=100, zorder=3)
                ax.annotate(f"{val:.3f}", (x, val), xytext=(0, 11),
                            textcoords="offset points", ha="center", fontsize=13)
            else:
                ax.text(x, .07, "not saved", ha="center", fontsize=12, color="#666666")
        ax.axhline(report["empty_aperture_reference"], color="#999999", ls=":", lw=1)
        ax.set(xticks=[0, 1, 2], xticklabels=["Closed", "After\n5 mm pull", "Post-pull"],
               xlim=(-.45, 2.45), ylim=(-.035, .53))
        ax.set_title(f'{trial["trial"]}: {trial["label"]}', fontsize=15, loc="left")
        ax.text(.02, .96, trial["outcome_note"], transform=ax.transAxes,
                va="top", fontsize=11, color="#444444")
        ax.tick_params(labelsize=12)
        sns.despine(ax=ax)
    axes[1, 3].axis("off")
    axes[1, 3].text(0, .93,
        "Measured opening, NOT pressure\n\n"
        "Closed: saved settled median\n"
        "After 5 mm: monitoring median\n"
        "Post-pull: one observation\n\n"
        "0 = fully closed; 1 = fully open\n"
        "Dotted: empty bound (0.02)\n\n"
        "No continuous full-pull trace.\n"
        "Phases are not a time axis.",
        transform=axes[1, 3].transAxes, va="top", fontsize=13, linespacing=1.5)
    fig.supylabel("Measured gripper opening (0–1)", fontsize=18)
    fig.suptitle("Door opening: grasp state across seven physical trials", fontsize=22)
    return fig


def plot_samples(report):
    fig, axes = plt.subplots(len(report["trials"]), 2, figsize=(12, 16),
                             sharex=True, sharey=True, layout="constrained")
    for row, trial in enumerate(report["trials"]):
        for col, key in enumerate(("close_samples", "post_proof_samples")):
            ax = axes[row, col]
            values = checked_samples(trial[key])
            ax.plot(np.arange(len(values)), values, color=["#2878B5", "#E38B2C"][col],
                    marker=".", markersize=4, linewidth=1.3)
            ax.axhline(report["empty_aperture_reference"], color="#999999", ls=":", lw=1)
            ax.set(ylim=(-.02, 1.02), xlim=(-1, 71))
            ax.text(.97, .9, f"n = {len(values)}", transform=ax.transAxes,
                    ha="right", fontsize=12)
            ax.tick_params(labelsize=12)
            sns.despine(ax=ax)
        axes[row, 0].set_ylabel(f'{trial["trial"]}\nOpening (0–1)', fontsize=14)
    axes[0, 0].set_title("Monitoring after close command", fontsize=18)
    axes[0, 1].set_title("Monitoring AFTER 5 mm proof pull", fontsize=18)
    for ax in axes[-1]:
        ax.set_xlabel("Sample index within this block (not seconds)", fontsize=14)
    fig.suptitle("Measured aperture samples — not pressure\n"
                 "Separate observation blocks; gaps and full-pull motion are not reconstructed",
                 fontsize=19)
    return fig


def plot_demo_comparison(report):
    comparison = report["demo_comparison"]
    rows = comparison["trials"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    x = np.arange(len(rows))
    colors = ["#999999" if row["trial"].startswith("E") else
              "#21875D" if row["trial"] in ("T6", "T7") else "#2878B5" for row in rows]
    for ax, key, title, ylabel, ymax in [
        (axes[0], "position_difference_mm", "Contact-position difference", "Distance to demo EE origin (mm)", 165),
        (axes[1], "orientation_difference_deg", "Contact-orientation difference", "Full rotation difference (degrees)", 12),
    ]:
        y = [row[key] for row in rows]
        ax.scatter(x, y, c=colors, s=90, zorder=3)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=12)
        ax.axhline(0, ls=":", color="#999999", lw=1)
        ax.set(xticks=x, xticklabels=[row["trial"] for row in rows],
               ylim=(-.03*ymax, ymax), xlim=(-.6, len(rows)-.4),
               xlabel="Chronological contact attempt", ylabel=ylabel, title=title)
        ax.tick_params(labelsize=13)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(18)
        sns.despine(ax=ax)
    fig.suptitle("Approaching the successful demo — not monotonically", fontsize=22)
    fig.text(.5, .035,
             "Gray: early contact-only attempts   |   Blue: T1–T5   |   Green: opening observed/verified (T6/T7)\n"
             "Absolute robot-frame comparison to the fixed medoid; door displacement is NOT compensated.",
             ha="center", fontsize=12)
    fig.tight_layout(rect=(0, .11, 1, .92))
    return fig


def plot_three_patterns(report):
    cases = report["three_patterns"]["cases"]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11),
                             gridspec_kw={"width_ratios": [1.15, 1, 1.2]})
    colors = ["#C44E52", "#D78B22", "#21875D"]
    for index, (case, color) in enumerate(zip(cases, colors)):
        close_ax, pull_ax, image_ax = axes[index]
        samples = checked_samples(case["samples"])
        close_ax.plot(np.arange(len(samples)), samples, color=color, lw=2,
                      marker=".", markersize=3)
        close_ax.axhline(report["empty_aperture_reference"], color="#999999", ls=":", lw=1)
        close_ax.set(xlim=(-1, 71), ylim=(-.03, .9), xlabel="Sample index after close command",
                     ylabel="Measured opening (0–1)")
        close_ax.set_title(f'{case["title"]} ({case["trial"]})', loc="left", fontsize=15, color=color)
        close_ax.annotate(f'Settled: {case["closed"]:.3f}',
                          (len(samples)-1, samples[-1]), xytext=(-8, 15),
                          textcoords="offset points", ha="right", fontsize=13)

        if case["proof"] is None:
            pull_ax.axis("off")
            pull_ax.text(.5, .62, "Empty grasp\nNo proof / full pull", transform=pull_ax.transAxes,
                         ha="center", va="center", color=color, fontsize=18)
            pull_ax.text(.5, .24, "Stopped at contact verification", transform=pull_ax.transAxes,
                         ha="center", fontsize=12, color="#666666")
        else:
            for x, val in enumerate([case["proof"], case["post_pull"]]):
                pull_ax.scatter(x, val, color=color, s=110, zorder=3)
                pull_ax.annotate(f"{val:.4f}", (x, val), xytext=(0, 13),
                                 textcoords="offset points", ha="center", fontsize=14)
            pull_ax.axvspan(.28, .72, color="#EEEEEE", zorder=0)
            pull_ax.text(.5, .24, "Full-pull\ntrace\nunavailable", ha="center", fontsize=12, color="#666666")
            pull_ax.axhline(report["empty_aperture_reference"], color="#999999", ls=":", lw=1)
            pull_ax.set(xticks=[0, 1], xticklabels=["After 5 mm\nproof pull", "Post-pull\nobservation"],
                        xlim=(-.4, 1.4), ylim=(-.03, .5), ylabel="Measured opening (0–1)")
            pull_ax.set_title("Retained after proof → later near zero", fontsize=14)
        image_ax.imshow(plt.imread(ROOT / case["display_image"]))
        image_ax.axis("off")
        state = case["door_state"]
        caption = "After close attempt (no endpoint test)" if index == 0 else f"RGB-D endpoint: {state.upper()}"
        image_ax.set_title(caption, fontsize=16, color=color)
        image_ax.text(.5, -.06, "Full-frame head image; later observation", transform=image_ax.transAxes,
                      ha="center", fontsize=11, color="#555555")
        for ax in (close_ax, pull_ax):
            ax.tick_params(labelsize=12)
            sns.despine(ax=ax)
    fig.suptitle("Three observed grasp patterns — measured aperture, not pressure", fontsize=22)
    fig.text(.5, .025,
             "B and C both end at 0.0034: aperture alone cannot determine whether the door opened.\n"
             "Separate observation blocks, not a continuous time axis; the exact opening/slip timing is unresolved.",
             ha="center", fontsize=14)
    fig.tight_layout(rect=(0, .085, 1, .94), h_pad=3, w_pad=2)
    return fig


def plot_successful_opening(report):
    data = report["successful_opening"]
    trial = next(t for t in report["trials"] if t["trial"] == data["trial"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={"height_ratios": [1, 1.15]})
    color = "#21875D"
    ax = axes[0, 0]
    values = trial["close_samples"]
    ax.plot(np.arange(len(values)), values, color=color, lw=2, marker=".", markersize=4)
    ax.axhline(.02, color="#999999", ls=":", lw=1)
    ax.set(xlim=(-1, 71), ylim=(-.03, .9), xlabel="Sample index after close command",
           ylabel="Measured opening (0–1)", title="Close: settles with opening remaining")
    ax.annotate(f'{trial["closed_aperture"]:.4f}', (69, values[-1]),
                xytext=(-10, 15), textcoords="offset points", ha="right", fontsize=15)
    ax = axes[0, 1]
    for x, value in enumerate([trial["post_proof_median"], trial["post_pull_aperture"]]):
        ax.scatter(x, value, color=color, s=120, zorder=3)
        ax.annotate(f"{value:.4f}", (x, value), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=15)
    ax.axvspan(.28, .72, color="#EEEEEE", zorder=0)
    ax.text(.5, .22, "Full-pull trace\nunavailable", ha="center", fontsize=13, color="#666666")
    ax.axhline(.02, color="#999999", ls=":", lw=1)
    ax.set(xlim=(-.4, 1.4), ylim=(-.03, .5), xticks=[0, 1],
           xticklabels=["After 5 mm proof", "Post-pull"], ylabel="Measured opening (0–1)",
           title="Proof retained; grasp later lost")
    for ax in axes[0]:
        ax.tick_params(labelsize=13)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.title.set_size(16)
        sns.despine(ax=ax)
    for ax, key, title in [
        (axes[1, 0], "initial_display_image", "Before: CLOSED (RGB-D verified)"),
        (axes[1, 1], "final_display_image", "After: OPEN (RGB-D verified)"),
    ]:
        ax.imshow(plt.imread(ROOT / data[key]))
        ax.axis("off")
        ax.set_title(title, fontsize=18, color=color)
    fig.suptitle("Successful autonomous door opening — T7", fontsize=23, color=color)
    fig.text(.5, .025,
             "Task success: door changed from closed to open. Continuous grasp was not maintained.\n"
             "Measured aperture, not pressure; separate observation blocks, not a continuous time trace.",
             ha="center", fontsize=13)
    fig.tight_layout(rect=(0, .075, 1, .94), h_pad=2)
    return fig


def plot_outcome_overlay(report):
    cases = report["outcome_overlay"]["cases"]
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    colors = ["#C44E52", "#DE8F05", "#21966F", "#246CB4"]
    markers = ["o", "s", "^", "D"]
    ax.axvspan(2.4, 3.6, color="#EEEEEE", zorder=0)
    ax.text(3, .51, "Full pull\n(no saved\ncontinuous trace)", ha="center", fontsize=13, color="#666666")
    for index, (case, color, marker) in enumerate(zip(cases, colors, markers)):
        values = case["aperture"]
        outcome = "success: open" if case["door_state"] == "open" else "failure: closed"
        ax.plot([0, 1, 2], values[:3], ls="--", color=color, lw=1.5, alpha=.8)
        ax.scatter([0, 1, 2, 4], values, marker=marker, s=110, linewidths=1.8,
                   facecolors="none", edgecolors=color, zorder=3+index,
                   label=f'{case["trial"]} — {outcome}')
        photo_ax = fig.add_subplot(gs[index//2, 1+index%2])
        photo_ax.imshow(plt.imread(ROOT / case["display_image"]))
        photo_ax.axis("off")
        photo_ax.set_title(f'{case["trial"]}: {case["door_state"].upper()}', fontsize=19, color=color)
    ax.axhline(report["empty_aperture_reference"], color="#888888", ls=":", lw=1)
    ax.set(xticks=[0, 1, 2, 4], xticklabels=["Open\nstart", "After\nclose", "After 5 mm\nproof", "Post-pull"],
           xlim=(-.25, 4.3), ylim=(-.06, 1.12), ylabel="Measured gripper opening (0–1)",
           xlabel="Observation phase (not elapsed time)")
    ax.annotate("All four start at measured 1.0", (0, 1), xytext=(.25, 1.045), fontsize=13)
    ax.annotate("All four end at 0.0034\n(points overlap)", (4, cases[0]["aperture"][-1]),
                xytext=(2.3, .17), fontsize=13,
                arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.legend(loc="upper right", bbox_to_anchor=(1, .91), fontsize=13, frameon=True)
    ax.tick_params(labelsize=14)
    ax.yaxis.label.set_size(17)
    ax.xaxis.label.set_size(15)
    sns.despine(ax=ax)
    fig.suptitle("Gripper opening: two failed and two successful door-opening trials", fontsize=22)
    fig.text(.72, .88, "Later head images / RGB-D endpoint labels", ha="center", fontsize=14)
    fig.text(.5, .025,
             "Dashed lines only connect saved phase summaries; no full-pull curve is inferred.\n"
             "Opening remaining after proof does not by itself distinguish success from failure. This is not pressure.",
             ha="center", fontsize=13)
    fig.tight_layout(rect=(0, .085, 1, .86), w_pad=2, h_pad=3.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-report", action="store_true")
    args = parser.parse_args()
    if args.rebuild_report:
        REPORT.write_text(json.dumps(build_report(), indent=2) + "\n")
    report = json.loads(REPORT.read_text())
    with sns.axes_style("whitegrid"), plt.rc_context({
        "font.family": "DejaVu Sans", "svg.fonttype": "none",
        "svg.hashsalt": "door_grasp_signals_v1",
    }):
        for plot, stem in [(plot_summary, "door_grasp_aperture_trials"),
                           (plot_samples, "door_grasp_aperture_samples"),
                           (plot_demo_comparison, "door_contact_demo_comparison"),
                           (plot_three_patterns, "door_grasp_three_patterns"),
                           (plot_successful_opening, "door_grasp_success_T7"),
                           (plot_outcome_overlay, "door_grasp_outcome_overlay")]:
            fig = plot(report)
            save(fig, stem)
            plt.close(fig)
            print(ASSETS / f"{stem}.png")


if __name__ == "__main__":
    main()
