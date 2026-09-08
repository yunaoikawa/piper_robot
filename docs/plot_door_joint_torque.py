#!/usr/bin/env python3
"""Offline Door torque comparison: event-selected samples, never pressure.

Default plots tracked measurements only. --rebuild-report reads preserved run
JSON and records source hashes; it never accesses the robot.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/code_as_learning_machine"
REPORT = ASSETS / "door_joint_torque_report.json"


def validate_torque(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Expected six finite nonnegative torque magnitudes")
    return values


def build_report():
    grasp_path = ASSETS / "door_grasp_signal_report.json"
    grasp = json.loads(grasp_path.read_text())
    outcomes = {case["trial"]: case for case in grasp["outcome_overlay"]["cases"]}
    sources = {str(grasp_path.relative_to(ROOT)): hashlib.sha256(grasp_path.read_bytes()).hexdigest()}
    trials = []
    for row in grasp["trials"]:
        path = ROOT / "data/runs/pasteur" / row["proof_source"].replace("proof_state.json", "proof_pull.json")
        raw = path.read_bytes()
        motion = json.loads(raw)["motion"]
        warning = motion["last_torque_warning"]
        if warning is None or warning["invalid"] or warning["stage"] != "while settling after trajectory":
            raise ValueError(f"Missing or different-stage warning: {path}")
        values = validate_torque(warning["sample"])
        source = str(path.relative_to(ROOT))
        sources[source] = hashlib.sha256(raw).hexdigest()
        trials.append({
            "trial": row["trial"], "label": row["label"], "source": source,
            "selection": "last_torque_warning", "stage": warning["stage"],
            "torque_magnitude_nm": values.tolist(), "warning_limit_nm": warning["limit"],
            "torque_warning_count": motion["torque_warning_count"],
            "torque_stop_enforced": motion["torque_stop_enforced"],
            "saved_vectors": 1,
            "door_state": outcomes.get(row["trial"], {}).get("door_state"),
        })
    return {"schema": "door_joint_torque_audit/v1",
            "measurement": "Controller-reported physical-right arm joint torque magnitude",
            "units": "N m", "is_pressure": False,
            "phase": "Settling after the 5 mm proof pull",
            "trials": trials,
            "plotted_trials": ["T1", "T2", "T6", "T7"],
            "endpoint_images": grasp["outcome_overlay"]["cases"],
            "limitations": [
                "One last warning-triggered vector per proof-pull stage, not a mean, peak, or uniformly sampled trace.",
                "Values were stored after abs(); torque signs cannot be recovered from these warning records.",
                "Trigger selection is biased toward threshold-exceeding states; no unbiased success/failure test is possible.",
                "Joint configuration, gravity, motion, friction and contact can all affect reported torque.",
                "No gravity/dynamics subtraction or Jacobian-based contact-force estimation is applied.",
                "6 joint categories on x-axis, NOT time or motion progress. No values normalized to 1.",
                "Door endpoints come from the separate RGB-D evaluation, not a torque threshold.",
                "No complete full-pull torque trace was identified in these run summaries.",
            ],
            "source_sha256": sources}


def plot_comparison(report):
    rows = {r["trial"]: r for r in report["trials"]}
    images = {r["trial"]: r for r in report["endpoint_images"]}
    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, 3, width_ratios=[2.1, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    colors = ["#C44E52", "#DE8F05", "#21966F", "#246CB4"]
    markers = ["o", "s", "^", "D"]
    for index, (name, color, marker) in enumerate(zip(report["plotted_trials"], colors, markers)):
        row = rows[name]
        outcome = "success: open" if row["door_state"] == "open" else "failure: closed"
        ax.scatter(np.arange(1, 7), validate_torque(row["torque_magnitude_nm"]),
                   marker=marker, s=115, facecolors="none", edgecolors=color, linewidths=2,
                   label=f"{name} — {outcome}", zorder=3)
        photo_ax = fig.add_subplot(grid[index//2, 1+index%2])
        photo_ax.imshow(plt.imread(ROOT / images[name]["display_image"]))
        photo_ax.axis("off")
        photo_ax.set_title(f'{name}: {row["door_state"].upper()}', color=color, fontsize=19)
    ax.set(xticks=np.arange(1, 7), xticklabels=[f"J{i}" for i in range(1, 7)],
           xlim=(.6, 6.4), ylim=(-.07, 2.0),
           xlabel="Right-arm joint (not time)", ylabel="Reported joint torque magnitude |τ| (N m)")
    ax.tick_params(labelsize=14)
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.legend(loc="upper right", fontsize=13, frameon=True)
    ax.text(.03, .97, "After 5 mm proof pull\nLast warning sample only\nn = 1 vector per trial",
            transform=ax.transAxes, va="top", fontsize=14)
    sns.despine(ax=ax)
    fig.suptitle("Joint torque: two failed and two successful door-opening trials", fontsize=22)
    fig.text(.73, .88, "Later head images / RGB-D endpoint labels", ha="center", fontsize=14)
    fig.text(.5, .025,
             "Event-selected samples, NOT averages, peaks, or pressure; posture/gravity effects are not removed.\n"
             "The differences are descriptive only: these samples do not establish a torque-based success criterion.",
             ha="center", fontsize=13)
    fig.tight_layout(rect=(0, .085, 1, .86), w_pad=2, h_pad=3.5)
    return fig


def plot_all(report):
    rows = report["trials"]
    values = np.asarray([validate_torque(r["torque_magnitude_nm"]) for r in rows])
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    sns.heatmap(values, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1.7,
                xticklabels=[f"J{i}" for i in range(1, 7)],
                yticklabels=[r["trial"] for r in rows],
                cbar_kws={"label": "Torque magnitude (N m)"}, annot_kws={"size": 13}, ax=ax)
    ax.set(xlabel="Right-arm joint", ylabel="Physical trial",
           title="After proof pull: one last-warning vector per trial\nNot a time trace or unbiased torque statistic")
    ax.tick_params(labelsize=13)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-report", action="store_true")
    args = parser.parse_args()
    if args.rebuild_report:
        REPORT.write_text(json.dumps(build_report(), indent=2) + "\n")
    report = json.loads(REPORT.read_text())
    with sns.axes_style("whitegrid"), plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "door_torque_v1"}):
        for plot, stem in [(plot_comparison, "door_joint_torque_comparison"), (plot_all, "door_joint_torque_all_trials")]:
            fig = plot(report)
            for ext in ("png", "svg"):
                path = ASSETS / f"{stem}.{ext}"
                fig.savefig(path, dpi=200, metadata={"Date": None} if ext == "svg" else {})
                if ext == "svg":
                    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
            plt.close(fig)
            print(ASSETS / f"{stem}.png")


if __name__ == "__main__":
    main()
