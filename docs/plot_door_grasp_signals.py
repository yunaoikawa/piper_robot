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
    return {
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
    }


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
                           (plot_samples, "door_grasp_aperture_samples")]:
            fig = plot(report)
            save(fig, stem)
            plt.close(fig)
            print(ASSETS / f"{stem}.png")


if __name__ == "__main__":
    main()
