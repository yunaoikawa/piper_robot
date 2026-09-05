#!/usr/bin/env python3
"""Generate separate, evidence-grounded Door and Cap evolution figures.

The upper panels are recomputed from preserved experimental JSON rather than
being assigned ordinal capability levels.  The lower panels summarize the
changing executable tool graph; they are historical context, not extra data
points on the quantitative axes.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parent
OUTPUT = ROOT / "assets" / "code_as_learning_machine"

NAVY = "#17324D"
BLUE = "#2B6CB0"
TEAL = "#168AAD"
GREEN = "#2F855A"
ORANGE = "#C05621"
RED = "#B83232"
GRAY = "#657786"
LIGHT = "#F5F8FB"
GRID = "#D7E1EA"


def node(
    ax,
    xy,
    width,
    height,
    text,
    *,
    edge=BLUE,
    face="white",
    fontsize=7.4,
    linestyle="-",
    linewidth=1.35,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        linewidth=linewidth, edgecolor=edge, facecolor=face,
        linestyle=linestyle, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize, color=NAVY,
        linespacing=1.05, zorder=4,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=GRAY,
    linestyle="-",
    width=1.15,
    connectionstyle="arc3",
):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8,
        linewidth=width, color=color, linestyle=linestyle,
        connectionstyle=connectionstyle, shrinkA=2, shrinkB=2, zorder=2,
    )
    ax.add_patch(patch)
    return patch


def stage_panel(ax, title, subtitle):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.02), 0.98, 0.94,
            boxstyle="round,pad=0.018,rounding_size=0.04",
            linewidth=1.15, edgecolor=GRID, facecolor=LIGHT,
        )
    )
    ax.text(0.05, 0.90, title, ha="left", va="top", fontsize=9.4,
            color=NAVY, weight="bold")
    ax.text(0.05, 0.75, subtitle, ha="left", va="top", fontsize=7.1,
            color=GRAY, linespacing=1.05)


def three_node_chain(ax, labels, *, colors=(GRAY, ORANGE, RED), y=0.42):
    xs = (0.06, 0.37, 0.68)
    for x, label, color in zip(xs, labels, colors):
        node(ax, (x, y), 0.25, 0.16, label, edge=color)
    arrow(ax, (0.31, y + 0.08), (0.37, y + 0.08))
    arrow(ax, (0.62, y + 0.08), (0.68, y + 0.08))


# Door graph slices ---------------------------------------------------------


def door_d0(ax):
    stage_panel(ax, "D0  Generic replay", "Before the door-specific system")
    three_node_chain(ax, ("camera", "absolute\nreplay", "robot"))
    ax.text(0.5, 0.23, "command ≠ endpoint", ha="center", fontsize=8,
            color=RED, weight="bold")


def door_d1(ax):
    stage_panel(ax, "D1  Relative demo", "Contact frame replaces world replay")
    three_node_chain(
        ax, ("Quest +\nHDF5/MP4", "demo\ncompiler", "SE(3) + IK"),
        colors=(TEAL, BLUE, GREEN), y=0.49,
    )
    node(ax, (0.35, 0.22), 0.30, 0.14, "contact-relative\ntrajectory",
         edge=ORANGE)
    arrow(ax, (0.50, 0.49), (0.50, 0.36), color=ORANGE)


def door_d2(ax):
    stage_panel(ax, "D2  Checkpointed pull", "Short proof no longer\nlicenses a blind pull")
    node(ax, (0.08, 0.51), 0.24, 0.14, "close once", edge=ORANGE)
    node(ax, (0.38, 0.51), 0.24, 0.14, "aperture\ngate", edge=GREEN)
    node(ax, (0.68, 0.51), 0.24, 0.14, "5 mm proof", edge=GREEN)
    arrow(ax, (0.32, 0.58), (0.38, 0.58))
    arrow(ax, (0.62, 0.58), (0.68, 0.58))
    node(ax, (0.18, 0.24), 0.29, 0.14, "checkpointed\npull", edge=BLUE)
    node(ax, (0.57, 0.24), 0.25, 0.14, "stop on slip", edge=RED)
    arrow(ax, (0.80, 0.51), (0.39, 0.38), color=BLUE,
          connectionstyle="arc3,rad=0.18")
    arrow(ax, (0.47, 0.31), (0.57, 0.31), color=RED)


def door_d3(ax):
    stage_panel(ax, "D3  Metric alignment", "Shadow loses authority;\ndepth owns geometry")
    node(ax, (0.04, 0.55), 0.22, 0.13, "Record3D\nRGB-D", edge=TEAL)
    node(ax, (0.32, 0.55), 0.24, 0.13, "Tag/PnP", edge=BLUE)
    node(ax, (0.63, 0.55), 0.31, 0.13, "RANSAC\ndoor plane", edge=GREEN)
    arrow(ax, (0.26, 0.615), (0.32, 0.615))
    arrow(ax, (0.56, 0.615), (0.63, 0.615))
    node(ax, (0.08, 0.29), 0.27, 0.13, "bounded\nwrist feature", edge=TEAL)
    node(ax, (0.44, 0.29), 0.28, 0.13, "retargeted\npull", edge=ORANGE)
    node(ax, (0.78, 0.29), 0.16, 0.13, "robot", edge=RED)
    arrow(ax, (0.35, 0.355), (0.44, 0.355))
    arrow(ax, (0.72, 0.355), (0.78, 0.355))
    arrow(ax, (0.79, 0.55), (0.61, 0.42), color=GREEN,
          connectionstyle="arc3,rad=0.12")


def door_d4(ax):
    stage_panel(ax, "D4  Autonomous endpoints", "Open and close use\ndifferent branches")
    node(ax, (0.06, 0.57), 0.25, 0.13, "fresh RGB-D\nstate", edge=TEAL)
    node(ax, (0.38, 0.57), 0.24, 0.13, "open/closed\nclassifier", edge=GREEN)
    node(ax, (0.69, 0.57), 0.24, 0.13, "state\nmachine", edge=BLUE)
    arrow(ax, (0.31, 0.635), (0.38, 0.635))
    arrow(ax, (0.62, 0.635), (0.69, 0.635))
    node(ax, (0.14, 0.28), 0.28, 0.14, "relative pull\n(open)", edge=ORANGE)
    node(ax, (0.58, 0.28), 0.28, 0.14, "dedicated push\n(close)", edge=ORANGE)
    arrow(ax, (0.81, 0.57), (0.31, 0.42), color=ORANGE,
          connectionstyle="arc3,rad=0.18")
    arrow(ax, (0.81, 0.57), (0.72, 0.42), color=ORANGE,
          connectionstyle="arc3,rad=-0.18")
    ax.text(0.5, 0.13, "subprocess + JSON journal", ha="center", fontsize=7.8,
            color=GRAY, weight="bold")


def door_d5(ax):
    stage_panel(ax, "D5  Evidence hardening", "Same capability; stricter classifier")
    node(ax, (0.06, 0.54), 0.26, 0.14, "closed-plane\nmask", edge=BLUE)
    node(ax, (0.37, 0.54), 0.26, 0.14, "endpoint\ndifference", edge=TEAL)
    node(ax, (0.68, 0.54), 0.26, 0.14, "intersection", edge=GREEN)
    arrow(ax, (0.32, 0.61), (0.37, 0.61))
    arrow(ax, (0.63, 0.61), (0.68, 0.61))
    node(ax, (0.19, 0.27), 0.28, 0.14, "exclude arm\ncontamination", edge=RED)
    node(ax, (0.55, 0.27), 0.28, 0.14, "fail closed\n+ tests + Git", edge=GRAY)
    arrow(ax, (0.81, 0.54), (0.39, 0.41), color=RED,
          connectionstyle="arc3,rad=0.16")
    arrow(ax, (0.47, 0.34), (0.55, 0.34))


# Cap graph slices ----------------------------------------------------------


def cap_c0(ax):
    stage_panel(ax, "C0  Ambiguous target", "Largest-white and tabletop\nassumptions fail")
    three_node_chain(ax, ("head RGB", "largest white\ncomponent", "wrong object"))
    ax.text(0.5, 0.23, "identity loss", ha="center", fontsize=8,
            color=RED, weight="bold")


def cap_c1(ax):
    stage_panel(ax, "C1  Bound identity", "One click anchors a\nrelational detector")
    node(ax, (0.05, 0.53), 0.22, 0.14, "one click", edge=GRAY)
    node(ax, (0.33, 0.53), 0.28, 0.14, "white cap\nabove neck", edge=BLUE)
    node(ax, (0.68, 0.53), 0.26, 0.14, "instance\nanchor", edge=GREEN)
    arrow(ax, (0.27, 0.60), (0.33, 0.60))
    arrow(ax, (0.61, 0.60), (0.68, 0.60))
    node(ax, (0.22, 0.25), 0.25, 0.14, "cyan jaws", edge=TEAL)
    node(ax, (0.56, 0.25), 0.26, 0.14, "midpoint +\nspan", edge=ORANGE)
    arrow(ax, (0.47, 0.32), (0.56, 0.32))
    arrow(ax, (0.81, 0.53), (0.70, 0.39), color=GREEN,
          connectionstyle="arc3,rad=0.18")


def cap_c2(ax):
    stage_panel(ax, "C2  Completed 3-D scene", "Depth and catalog volume\nhave separate owners")
    node(ax, (0.04, 0.55), 0.22, 0.13, "Record3D\ndepth", edge=TEAL)
    node(ax, (0.32, 0.55), 0.22, 0.13, "fixed tag", edge=BLUE)
    node(ax, (0.61, 0.55), 0.33, 0.13, "3-D surface", edge=GREEN)
    arrow(ax, (0.26, 0.615), (0.32, 0.615))
    arrow(ax, (0.54, 0.615), (0.61, 0.615))
    node(ax, (0.08, 0.28), 0.27, 0.13, "catalog size", edge=GRAY)
    node(ax, (0.42, 0.28), 0.22, 0.13, "MJCF", edge=ORANGE)
    node(ax, (0.71, 0.28), 0.23, 0.13, "MuJoCo\ncritic", edge=GREEN)
    arrow(ax, (0.35, 0.345), (0.42, 0.345))
    arrow(ax, (0.64, 0.345), (0.71, 0.345))
    arrow(ax, (0.77, 0.55), (0.55, 0.41), color=GREEN,
          connectionstyle="arc3,rad=0.14")


def cap_c3(ax):
    stage_panel(ax, "C3  Learned local control", "Seven probes replace global calibration")
    node(ax, (0.05, 0.55), 0.22, 0.14, "7 open-jaw\nprobes", edge=ORANGE)
    node(ax, (0.34, 0.55), 0.28, 0.14, "NumPy\npseudoinverse", edge=BLUE)
    node(ax, (0.69, 0.55), 0.25, 0.14, "local image\nJacobian", edge=GREEN)
    arrow(ax, (0.27, 0.62), (0.34, 0.62))
    arrow(ax, (0.62, 0.62), (0.69, 0.62))
    node(ax, (0.08, 0.27), 0.24, 0.14, "lateral\nfirst", edge=TEAL)
    node(ax, (0.39, 0.27), 0.24, 0.14, "vertical\ndescent", edge=ORANGE)
    node(ax, (0.70, 0.27), 0.22, 0.14, "close once", edge=RED)
    arrow(ax, (0.32, 0.34), (0.39, 0.34))
    arrow(ax, (0.63, 0.34), (0.70, 0.34))
    arrow(ax, (0.82, 0.55), (0.20, 0.41), color=GREEN,
          connectionstyle="arc3,rad=0.19")


def cap_c4(ax):
    stage_panel(ax, "C4  Verified held transfer", "Independent modalities\npromote a prefix")
    node(ax, (0.04, 0.55), 0.21, 0.13, "aperture", edge=ORANGE)
    node(ax, (0.30, 0.55), 0.20, 0.13, "FK", edge=BLUE)
    node(ax, (0.55, 0.55), 0.21, 0.13, "RGB-D", edge=TEAL)
    node(ax, (0.81, 0.55), 0.15, 0.13, "support", edge=GREEN)
    node(ax, (0.12, 0.28), 0.28, 0.14, "removal\nevaluator", edge=GREEN)
    node(ax, (0.47, 0.28), 0.28, 0.14, "transport\nevaluator", edge=GREEN)
    node(ax, (0.80, 0.28), 0.16, 0.14, "route\nhash", edge=GRAY)
    for x in (0.145, 0.40, 0.655, 0.885):
        arrow(ax, (x, 0.55), (0.31 if x < 0.5 else 0.61, 0.42), color=GREEN)
    arrow(ax, (0.75, 0.35), (0.80, 0.35))


def cap_c5(ax):
    stage_panel(ax, "C5  Release branch", "Placement evidence did not pass")
    node(ax, (0.07, 0.52), 0.24, 0.15, "held cap", edge=GREEN)
    node(ax, (0.38, 0.52), 0.24, 0.15, "release\nmotion", edge=ORANGE)
    node(ax, (0.69, 0.52), 0.24, 0.15, "placement\nevaluator", edge=RED)
    arrow(ax, (0.31, 0.595), (0.38, 0.595))
    arrow(ax, (0.62, 0.595), (0.69, 0.595))
    node(
        ax, (0.27, 0.24), 0.46, 0.15, "insufficient\nsupport evidence",
        edge=RED, face="#FFF5F5", linestyle="--",
    )
    arrow(ax, (0.81, 0.52), (0.66, 0.39), color=RED, linestyle="--",
          connectionstyle="arc3,rad=0.16")
    ax.text(0.5, 0.11, "placement not promoted", ha="center", fontsize=8,
            color=RED, weight="bold")


def load_json(relative_path):
    with (REPOSITORY / relative_path).open() as stream:
        return json.load(stream)


def validate_available_source_hashes(value):
    """Verify local raw inputs while allowing generation from the tracked snapshot."""
    if isinstance(value, dict):
        if "source_path" in value and "source_sha256" in value:
            path = REPOSITORY / value["source_path"]
            if path.exists():
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != value["source_sha256"]:
                    raise ValueError(f"source hash changed: {value['source_path']}")
        for item in value.values():
            validate_available_source_hashes(item)
    elif isinstance(value, list):
        for item in value:
            validate_available_source_hashes(item)


def goal_conditioned_endpoint_score(state, goal):
    """Map registered endpoint errors to [0, 1], with 1 at the requested goal."""
    d_open = float(state["relative_open_error"])
    d_closed = float(state["relative_closed_error"])
    denominator = d_open + d_closed
    if denominator <= 0:
        raise ValueError("endpoint errors must have a positive sum")
    if goal == "open":
        return d_closed / denominator
    if goal == "closed":
        return d_open / denominator
    raise ValueError(f"unsupported endpoint goal: {goal}")


def figure_shell(title, subtitle, count=6):
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    grid = fig.add_gridspec(
        2, count, height_ratios=[1.02, 1.0], left=0.075, right=0.988,
        top=0.875, bottom=0.06, hspace=0.30, wspace=0.14,
    )
    fig.suptitle(title, x=0.075, y=0.957, ha="left", fontsize=17,
                 color=NAVY, weight="bold")
    fig.text(0.077, 0.912, subtitle, ha="left", va="center", fontsize=10.3,
             color=GRAY)
    return fig, grid


def finish_figure(fig, *, stem, footnote=None):
    fig.text(
        0.075, 0.018,
        footnote or "Upper panels are computed from preserved observations; lower panels show the changing executable graph and are not additional samples.",
        ha="left", fontsize=8.2, color=GRAY,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / f"{stem}.png"
    svg = OUTPUT / f"{stem}.svg"
    metadata = {"Creator": "piper_robot manuscript figure generator", "Date": None}
    fig.savefig(png, dpi=180, facecolor="white", metadata=metadata)
    fig.savefig(svg, facecolor="white", metadata=metadata)
    plt.close(fig)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    return png, svg


def make_door_figure():
    report = load_json(
        "docs/assets/code_as_learning_machine/door_configuration_curve_report.json"
    )
    configurations = report["configurations"]
    labels = [f"{item['stage']}\n{item['label']}" for item in configurations]
    measured_x = []
    errors_mm = []
    for index, item in enumerate(configurations):
        if item["status"] != "measured":
            continue
        error_mm = item["relative_open_error"] * report["evaluator"]["endpoint_separation_m"] * 1000.0
        if not np.isfinite(error_mm) or error_mm < 0:
            raise ValueError(f"invalid Door depth error for {item['stage']}")
        if not np.isclose(error_mm, item["open_reference_median_absolute_depth_error_mm"], atol=1e-9):
            raise ValueError(f"stored Door depth error does not match inputs for {item['stage']}")
        measured_x.append(index)
        errors_mm.append(error_mm)
    fig, grid = figure_shell(
        "Door task: depth error by executable agent configuration",
        "Distance from the open reference in registered depth (mm); lower is better. Selected outcomes, same development session.",
    )
    ax = fig.add_subplot(grid[0, :])
    x = np.arange(len(configurations), dtype=float)
    ax.set_xlim(-0.5, len(configurations) - 0.5)
    ax.set_ylim(0, max(errors_mm) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(NAVY)
    ax.set_ylabel("Median absolute depth error to open reference (mm)", fontsize=11, color=NAVY)
    ax.set_xlabel("Executable agent configuration", fontsize=10.5, color=NAVY)
    ax.set_xticks(x, labels, fontsize=9.5)
    ax.grid(color=GRID, linewidth=0.8, linestyle=":")
    ax.plot(measured_x, errors_mm, color=BLUE, linewidth=3, marker="o",
            markersize=8, markerfacecolor="white", markeredgewidth=2.2)
    for index, error_mm in zip(measured_x, errors_mm):
        ax.annotate(f"{error_mm:.1f} mm", (index, error_mm), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=NAVY, weight="bold")
    for index, item in enumerate(configurations):
        if item["status"] == "measured":
            continue
        missing_y = max(errors_mm) * 0.60
        ax.annotate("N/A", (index, missing_y), ha="center", va="center", fontsize=9.5,
                    color=GRAY, weight="bold")
        note = "baseline outcome\nnot identified" if index == 0 else "no new execution\nafter hardening"
        ax.annotate(note, (index, missing_y),
                    xytext=(0, -17), textcoords="offset points", ha="center",
                    va="top", fontsize=7.8, color=GRAY)
    ax.text(
        0.02, 0.93,
        r"$E_{open}=1000\,\mathrm{median}_{p\in M}|Z(p)-Z_{open}(p)|$; "
        "fixed registration and plane mask",
        transform=ax.transAxes, fontsize=9.5, color=GRAY, va="top",
    )
    for column, draw in enumerate((door_d0, door_d1, door_d2, door_d3, door_d4, door_d5)):
        draw(fig.add_subplot(grid[1, column]))
    return finish_figure(fig, stem="door_tool_graph_evolution")


def make_cap_figure():
    evidence = load_json(
        "docs/assets/code_as_learning_machine/quantitative_figure_evidence.json"
    )
    alignment_sources = evidence["cap_alignment"]
    alignment = [item["error_norm_jaw_spans"] for item in alignment_sources]
    transfer_sources = evidence["cap_transfer"]
    before, lifted, transported = (
        np.asarray(item["right_ee_translation_xyz_m"], dtype=float)
        for item in transfer_sources
    )
    lift_mm = float((lifted[2] - before[2]) * 1000.0)
    transport_mm = float(np.linalg.norm(transported - lifted) * 1000.0)
    held_path_mm = (0.0, lift_mm, lift_mm + transport_mm)

    fig, grid = figure_shell(
        "Cap task: quantitative alignment and held-motion evidence above its tool graph",
        "The two axes retain their physical meanings; they are not collapsed into an arbitrarily weighted score.",
    )
    align_ax = fig.add_subplot(grid[0, :4])
    x = np.arange(len(alignment), dtype=float)
    align_ax.spines[["top", "right"]].set_visible(False)
    align_ax.spines[["left", "bottom"]].set_color(NAVY)
    align_ax.set_ylabel(r"Jaw-normalized image error  $\Vert e\Vert$", fontsize=10.5,
                        color=NAVY)
    align_ax.set_xticks(
        x, [item["label"].replace(" ", "\n") for item in alignment_sources],
        fontsize=8.5,
    )
    align_ax.set_ylim(0, max(alignment) * 1.15)
    align_ax.grid(color=GRID, linewidth=0.8, linestyle=":")
    align_ax.plot(x, alignment, color=BLUE, linewidth=2.8, marker="o",
                  markersize=7.5, markerfacecolor="white", markeredgewidth=2)
    align_ax.annotate("exploration worsened error", (1, alignment[1]),
                      xytext=(20, -28), textcoords="offset points", fontsize=8.4,
                      color=RED, arrowprops={"arrowstyle": "->", "color": RED})
    for index in (0, 2, 4, 5):
        align_ax.annotate(f"{alignment[index]:.3f}", (x[index], alignment[index]),
                          xytext=(0, 10), textcoords="offset points", ha="center",
                          fontsize=8.5, color=NAVY, weight="bold")
    align_ax.text(0.01, 0.96, "A  Approach alignment (lower is better)",
                  transform=align_ax.transAxes, va="top", fontsize=10,
                  color=NAVY, weight="bold")

    held_ax = fig.add_subplot(grid[0, 4:])
    hx = np.arange(3, dtype=float)
    held_ax.spines[["top", "right"]].set_visible(False)
    held_ax.spines[["left", "bottom"]].set_color(NAVY)
    held_ax.set_ylabel("Cumulative verified held motion (mm)", fontsize=10.5,
                       color=NAVY)
    held_ax.set_xticks(hx, ("before\nlift", "lift\nprobe", "held after\ntransport"),
                       fontsize=8.5)
    held_ax.set_ylim(0, held_path_mm[-1] * 1.22)
    held_ax.grid(color=GRID, linewidth=0.8, linestyle=":")
    held_ax.plot(hx, held_path_mm, color=GREEN, linewidth=2.8, marker="o",
                 markersize=7.5, markerfacecolor="white", markeredgewidth=2)
    for index, value in enumerate(held_path_mm):
        held_ax.annotate(f"{value:.3f}", (hx[index], value), xytext=(0, 10),
                         textcoords="offset points", ha="center", fontsize=8.5,
                         color=NAVY, weight="bold")
    held_ax.text(0.02, 0.96, "B  Retained transfer (higher is farther)",
                 transform=held_ax.transAxes, va="top", fontsize=10,
                 color=NAVY, weight="bold")
    held_ax.text(0.98, 0.08, f"+{lift_mm:.3f} mm vertical\n+{transport_mm:.3f} mm 3-D",
                 transform=held_ax.transAxes, ha="right", fontsize=8.4, color=GRAY)

    for column, draw in enumerate((cap_c0, cap_c1, cap_c2, cap_c3, cap_c4, cap_c5)):
        draw(fig.add_subplot(grid[1, column]))
    return finish_figure(fig, stem="cap_tool_graph_evolution")


def door_approach_rows(report):
    """Compact display IDs only; preserve historical IDs in raw evidence."""
    measured = [r for r in report["configurations"] if r["status"] == "measured"]
    return [{**r, "source_stage": r["stage"], "stage": f"D{i}"}
            for i, r in enumerate(measured, 1)]


def make_door_approach_figure(*, distance=False):
    report = load_json("docs/assets/code_as_learning_machine/door_first_approach_report.json")
    validate_available_source_hashes(report)
    rows = door_approach_rows(report)
    distance_rows = None
    if distance:
        distance_report = load_json("docs/assets/code_as_learning_machine/door_approach_distance_report.json")
        validate_available_source_hashes(distance_report)
        distance_rows = distance_report["configurations"]
        if [r["historical_stage"] for r in distance_rows] != [r["source_stage"] for r in rows]:
            raise ValueError("Distance and image configurations do not match")
    fig = plt.figure(figsize=(12, 11), facecolor="white")
    grid = fig.add_gridspec(2, len(rows), height_ratios=[1.25, 1],
                           left=0.13, right=0.97, top=0.85, bottom=0.11,
                           hspace=0.30, wspace=0.12)
    fig.suptitle("Door: estimated approach distance" if distance else "Door: first-approach alignment", x=0.13, y=0.97,
                 ha="left", fontsize=25, color=NAVY, weight="bold")
    subtitle = ("EE-frame distance to recorded successful contact\nFixed-door assumption; NOT fully RGB-D registered." if distance else
                "Red-label position vs. successful-demo mean\nLower is closer; not a direct grasp-pose measurement.")
    fig.text(0.13, 0.914, subtitle,
             fontsize=15, color=GRAY, linespacing=1.5, va="center")
    labels = [r["stage"] for r in rows]
    ax = fig.add_subplot(grid[0, :])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.5, len(rows)-0.5)
    ax.set_xticks(range(len(rows)), labels, fontsize=18)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_xlabel("Agent configuration", color=NAVY, fontsize=17, labelpad=10)
    ax.set_ylabel("Estimated EE-position distance (mm)" if distance else "Normalized image-position error", color=NAVY, fontsize=17, labelpad=12)
    ax.grid(color=GRID, linestyle=":")
    values = [r["distance_mm"] for r in distance_rows] if distance else [r["uv_error"] for r in rows]
    ax.set_ylim(0, np.nanmax(values) * 1.35)
    # Markers only: missing configurations and single observations must not
    # become an interpolated or apparently smooth learning curve.
    ax.plot(range(len(rows)), values, linestyle="none", marker="o", color=BLUE,
            markersize=12, markerfacecolor="white", markeredgewidth=3,
            label="Earlier successful contact reference")
    if distance:
        ax.plot(range(len(rows)), [r["alternate_contact_distance_mm"] for r in distance_rows],
                linestyle="none", marker="x", color=GRAY, markersize=10, markeredgewidth=2,
                label="Alternate contact (sensitivity only)")
        ax.legend(loc="upper right", fontsize=11, frameon=False)
    for index, value in enumerate(values):
        offset = (-44, -7) if distance and distance_rows[index]["alternate_contact_distance_mm"] > value else (0, 16)
        ax.annotate(f"{value:.1f}" if distance else f"{value:.3f}", (index, value), xytext=offset,
                    textcoords="offset points", ha="center", color=NAVY, fontsize=19, weight="bold")
    graphs = {
        "D1": ("Relative demo", ("Teleoperation data", "Demo compiler", "Contact-relative path")),
        "D2": ("Checkpointed pull", ("Close + aperture gate", "5 mm proof pull", "Checkpoints + slip stop")),
        "D4": ("Autonomous pipeline", ("RGB-D state + plane yaw", "Bounded wrist alignment", "Pull + endpoint check")),
    }
    for column, row in enumerate(rows):
        panel = fig.add_subplot(grid[1, column])
        panel.set(xlim=(0, 1), ylim=(0, 1))
        panel.axis("off")
        title, steps = graphs[row["source_stage"]]
        panel.text(0.5, 0.98, f"{row['stage']}\n{title}", ha="center", va="top",
                   fontsize=16, weight="bold", color=NAVY, linespacing=1.5)
        for i, step in enumerate(steps):
            y = 0.59 - i*0.24
            node(panel, (0.04, y), 0.92, 0.16, step, fontsize=13,
                 edge=(BLUE, TEAL, GREEN)[i])
            if i < 2:
                arrow(panel, (0.5, y-0.015), (0.5, y-0.065), width=1.8)
    fig.text(0.13, 0.064, "Two contact references, not confidence intervals. Includes the intended preclose gap." if distance else
             "One selected run per configuration. D3 reuses a successful same-scene anchor.",
             color=GRAY, fontsize=12)
    return finish_figure(fig, stem="door_first_approach_distance" if distance else "door_first_approach", footnote=
                         "Display D1 / D2 / D3 = historical D1 / D2 / D4. Unmeasured configurations omitted.")


def make_door_approach_audit():
    report = load_json("docs/assets/code_as_learning_machine/door_first_approach_report.json")
    rows = door_approach_rows(report)
    corrected = report["within_run_correction"]["after_one_correction"]
    rows.append({**corrected, "stage": "D3 after one correction (not initial)"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="white")
    goal = report["evaluator"]["goal_feature_uv_log_area"]
    for ax, row in zip(axes.flat, rows):
        image_source = next(s for s in row["sources"] if s["source_path"].endswith("right.png"))
        path = REPOSITORY / image_source["source_path"]
        if not path.exists():
            raise FileNotFoundError(f"Audit montage requires the raw frame: {path}")
        ax.imshow(plt.imread(path))
        height, width = row["image_shape_hw"]
        x, y, w, h = row["detection"]["box_xywh"]
        feature = row["detection"]["feature_uv_log_area"]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#00ffbb", linewidth=2))
        ax.plot(feature[0]*width, feature[1]*height, "o", color="#00ffbb", markersize=5)
        ax.plot(goal[0]*width, goal[1]*height, "+", color="#ffdd33", markersize=15, markeredgewidth=2)
        ax.plot([feature[0]*width, goal[0]*width], [feature[1]*height, goal[1]*height],
                color="#ffdd33", linewidth=1.5, linestyle="--")
        ax.set_title(f"{row['stage']}\nposition mismatch={row['uv_error']:.3f}",
                     fontsize=11, color=NAVY)
        ax.axis("off")
    fig.suptitle("Detection audit: green = red-label detection; yellow + = successful-demo feature mean",
                 fontsize=12, color=NAVY)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.07, hspace=0.2)
    return finish_figure(fig, stem="door_first_approach_detection_audit", footnote=
                         "Same detector and reference on all frames. Label alignment is not direct handle-pose measurement; the fourth panel is a within-run diagnostic.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-source-audit", action="store_true",
                        help="Also render the detection montage (requires local raw right-camera frames)")
    args = parser.parse_args()
    evidence = load_json(
        "docs/assets/code_as_learning_machine/quantitative_figure_evidence.json"
    )
    validate_available_source_hashes(evidence)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "svg.hashsalt": "piper_robot_task_tool_graphs_v3",
        }
    )
    outputs = []
    outputs.extend(make_door_figure())
    outputs.extend(make_cap_figure())
    outputs.extend(make_door_approach_figure())
    outputs.extend(make_door_approach_figure(distance=True))
    if args.include_source_audit:
        outputs.extend(make_door_approach_audit())
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
