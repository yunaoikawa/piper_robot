#!/usr/bin/env python3
"""Generate separate, evidence-grounded Door and Cap evolution figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
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
    stage_panel(ax, "D2  Mechanical proof", "Short proof no longer\nlicenses a blind pull")
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


def make_figure(
    *, title, subtitle, stem, stage_labels, levels, level_labels,
    annotations, drawers, plateau_from=None,
):
    count = len(drawers)
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    grid = fig.add_gridspec(
        2, count, height_ratios=[1.02, 1.0], left=0.16, right=0.988,
        top=0.89, bottom=0.055, hspace=0.27, wspace=0.09,
    )
    ax = fig.add_subplot(grid[0, :])
    x = np.arange(count, dtype=float)
    y = np.asarray(levels, dtype=float)
    ax.set_xlim(-0.15, count - 0.55)
    ax.set_ylim(-0.25, max(y) + 0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(NAVY)
    ax.spines[["left", "bottom"]].set_linewidth(1.35)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=9)
    ticks = sorted(set(levels))
    ax.set_yticks(ticks)
    ax.set_yticklabels([level_labels[item] for item in ticks], fontsize=8.5)
    ax.set_ylabel("Highest evidence-supported capability prefix", fontsize=11,
                  color=NAVY, labelpad=16)
    ax.grid(axis="both", color=GRID, linewidth=0.8, linestyle=":")
    ax.plot(x, y, color=BLUE, linewidth=3.1, zorder=3)
    ax.scatter(x, y, s=82, color="white", edgecolor=BLUE, linewidth=2.3,
               zorder=4)
    if plateau_from is not None:
        ax.plot(x[plateau_from:], y[plateau_from:], color=ORANGE, linewidth=4,
                zorder=5)
    for index, label in enumerate(annotations):
        below = index >= count - 2 and y[index] == y[-1]
        ax.annotate(
            label, (x[index], y[index]), xytext=(0, -38 if below else 23),
            textcoords="offset points", ha="center",
            va="top" if below else "bottom", fontsize=8.6,
            color=RED if index == count - 1 and plateau_from is not None else NAVY,
            weight="bold" if index >= count - 2 else "normal",
        )
    for column, draw in enumerate(drawers):
        draw(fig.add_subplot(grid[1, column]))
    fig.suptitle(title, x=0.16, y=0.955, ha="left", fontsize=17,
                 color=NAVY, weight="bold")
    fig.text(0.162, 0.915, subtitle, ha="left", va="center", fontsize=10.3,
             color=GRAY)
    fig.text(
        0.16, 0.018,
        "Lower panels show the decisive graph slice added or changed at each stage; later stages retain applicable earlier components.",
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


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
            "svg.hashsalt": "piper_robot_task_tool_graphs_v2",
        }
    )
    outputs = []
    outputs.extend(
        make_figure(
            title="Door task: the executable system changed as different failures became measurable",
            subtitle=(
                "Pasteur experiments on 8 August 2026; task states are categorical, "
                "not fabricated success-rate samples."
            ),
            stem="door_tool_graph_evolution",
            stage_labels=(
                "D0\ngeneric replay", "D1\nrelative demo", "D2\nmechanical proof",
                "D3\nmetric alignment", "D4\nautonomous endpoints",
                "D5\nevidence hardening",
            ),
            levels=(0, 1, 2, 3, 4, 4),
            level_labels={
                0: "unverified motion", 1: "contact-frame approach",
                2: "verified 5 mm proof", 3: "verified open endpoint",
                4: "verified open + close",
            },
            annotations=(
                "No endpoint contract", "Absolute replay rejected",
                "Short proof passes; slips remain", "Door opens before detected slip",
                "Fresh RGB-D verifies both endpoints", "Same result; stronger mask",
            ),
            drawers=(door_d0, door_d1, door_d2, door_d3, door_d4, door_d5),
            plateau_from=4,
        )
    )
    outputs.extend(
        make_figure(
            title="Cap task: a separate tool graph was learned without a cap action demonstration",
            subtitle=(
                "Pasteur experiments on 6 August 2026, two days before the reported Door run; "
                "task states are categorical."
            ),
            stem="cap_tool_graph_evolution",
            stage_labels=(
                "C0\nambiguous target", "C1\nbound identity",
                "C2\ncompleted 3-D scene", "C3\nlocal control",
                "C4\nheld transfer", "C5\nrelease branch",
            ),
            levels=(0, 1, 2, 3, 4, 4),
            level_labels={
                0: "wrong/ambiguous target", 1: "identity bound",
                2: "3-D approach model", 3: "aligned side pinch",
                4: "verified removal + held transport",
            },
            annotations=(
                "Global whiteness fails", "Click + support relation",
                "Depth + catalog complete volume",
                "Seven probes identify local Jacobian",
                "9.175 mm lift; 107.415 mm transport",
                "Placement remains unverified",
            ),
            drawers=(cap_c0, cap_c1, cap_c2, cap_c3, cap_c4, cap_c5),
            plateau_from=4,
        )
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
