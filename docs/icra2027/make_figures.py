#!/usr/bin/env python3
"""Paper-size figures from archived measurements; no robot or model calls."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import seaborn as sns

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets/code_as_learning_machine"
OUT = HERE / "figures"


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def box(ax, x, y, w, h, label, color="#EAF1F8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                              fc=color, ec="#47617A", lw=.8))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=9)


def arrow(ax, start, end):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "lw": 1, "color": "#47617A"})


def main():
    OUT.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font="DejaVu Sans")
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "pdf.fonttype": 42})
    paths = [ASSETS / "door_contact_t7_comparison_orientation.json",
             ASSETS / "door_grasp_signal_report.json",
             ASSETS / "door_grasp_overlay_T7_head.jpg",
             ASSETS / "cap_verified_transfer_rgb.jpg"]
    comparison = json.loads(paths[0].read_text())
    report = json.loads(paths[1].read_text())
    rows = comparison["trials"]
    colors = ["#8C8C8C" if r["trial"].startswith("E") else
              "#16845D" if r["trial"] in ("T6", "T7") else "#2878B5" for r in rows]
    fig, ax = plt.subplots(figsize=(4.5, 2.45))
    x = np.arange(len(rows))
    y = [r["orientation_difference_deg"] for r in rows]
    ax.scatter(x, y, c=colors, s=30, zorder=3)
    for xx, yy in zip(x, y):
        ax.annotate(f"{yy:.1f}", (xx, yy), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=[r["trial"] for r in rows], ylim=(-.4, 11.5),
           ylabel="Orientation difference (deg)", xlabel="Chronological contact attempt")
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "orientation")

    fig, ax = plt.subplots(figsize=(4.5, 2.55))
    all_rows = {r["trial"]: r for r in report["trials"]}
    metrics = {}
    for ident, color, marker in [("T1", "#C44E52", "o"), ("T2", "#D58B16", "s"),
                                  ("T6", "#16845D", "^"), ("T7", "#2878B5", "D")]:
        r = all_rows[ident]
        values = [r["preclose_measured_aperture"], r["closed_aperture"],
                  r["post_proof_median"], r["post_pull_aperture"]]
        metrics[ident] = values
        label = ident + (": closed" if ident in ("T1", "T2") else ": open")
        ax.scatter([0, 1, 2, 3], values, edgecolor=color, facecolor="none", marker=marker,
                   s=40, lw=1.3, label=label, zorder=3)
    ax.set(xticks=[0, 1, 2, 3], xticklabels=["Pre-close", "Closed", "Proof", "Later"],
           ylim=(-.05, 1.08), ylabel="Measured opening (0–1)")
    ax.legend(fontsize=8, loc="upper right", frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "aperture")

    # Whole archived cap endpoint, exported for builds without raw captures.
    cap_export = OUT / "cap_endpoint.png"
    if not cap_export.exists():
        from PIL import Image
        raw = HERE.parents[1] / "data/captures/pasteur/2026-08-06/20260806T090638.703002Z_head_culture_media_cap_transport_home_hold_c01dae62/derived/head_rgb_landscape.png"
        with Image.open(raw) as im:
            im.convert("RGB").save(cap_export)
    paths.append(cap_export)
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.25))
    for ax, path, title in zip(axes, [paths[2], cap_export], ["Door: open endpoint", "Cap: held transport"]):
        ax.imshow(plt.imread(path))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout(pad=.2)
    save(fig, "task_overview")

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    for ax, ident in zip(axes.flat, ["T1", "T2", "T6", "T7"]):
        path = ASSETS / f"door_grasp_overlay_{ident}_head.jpg"
        paths.append(path)
        ax.imshow(plt.imread(path))
        ax.set_title(ident + (": closed" if ident in ("T1", "T2") else ": open"), fontsize=11)
        ax.axis("off")
    fig.tight_layout(pad=.4, h_pad=1.6)
    save(fig, "door_endpoints")

    fig, ax = plt.subplots(figsize=(4.5, 2.6))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, .02, .72, .28, .21, "Observe\nimages + state")
    box(ax, .37, .72, .27, .21, "Diagnose\nphysical failure")
    box(ax, .72, .72, .26, .21, "Revise\ncode + tests")
    arrow(ax, (.30, .825), (.37, .825)); arrow(ax, (.64, .825), (.72, .825))
    box(ax, .36, .39, .29, .20, "Execute bounded\nrobot stage", "#E5F2EB")
    arrow(ax, (.85, .71), (.64, .50)); arrow(ax, (.36, .50), (.16, .71))
    ax.text(.82, .49, "Human\ncorrections", ha="center", va="center", fontsize=9)
    arrow(ax, (.84, .60), (.84, .71))
    box(ax, .02, .04, .96, .19, "Accepted runtime: align → engage → move → verify\nNo LLM in the servo loop", "#F4F0E4")
    arrow(ax, (.50, .38), (.50, .24))
    fig.tight_layout(pad=.2); save(fig, "learning_loop")

    fig, ax = plt.subplots(figsize=(4.5, 3.1))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    box(ax, .02, .77, .96, .18, "Coding agent + human feedback\ninspect → revise → test", "#F4F0E4")
    box(ax, .02, .47, .43, .20, "Task policy P\nrecognition + motion")
    box(ax, .55, .47, .43, .20, "Task harness H\nobserve / verify / recover")
    arrow(ax, (.24, .76), (.24, .68)); arrow(ax, (.77, .76), (.77, .68))
    box(ax, .02, .19, .96, .16, "Inherited services B: cameras, kinematics, robot RPC", "#E5F2EB")
    arrow(ax, (.24, .46), (.24, .36)); arrow(ax, (.77, .46), (.77, .36))
    ax.text(.5, .06, "Saved physical evidence → next development iteration", ha="center", fontsize=9)
    fig.tight_layout(pad=.2); save(fig, "harness_layers")
    manifest = {"description": "Read-only input provenance for ICRA figures; no robot calls",
                "sources": {str(p.relative_to(HERE.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in paths},
                "orientation_deg": dict(zip([r["trial"] for r in rows], y)),
                "aperture_phases": metrics}
    (HERE / "figure_provenance.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
