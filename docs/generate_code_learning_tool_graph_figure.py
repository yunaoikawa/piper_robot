#!/usr/bin/env python3
"""Generate the manuscript's qualitative tool-graph learning figure."""

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


def node(ax, xy, width, height, text, *, edge=BLUE, face="white", fontsize=8.2,
         linestyle="-", linewidth=1.4):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=fontsize, color=NAVY,
        linespacing=1.08, zorder=4,
    )
    return patch


def arrow(ax, start, end, *, color=GRAY, linestyle="-", width=1.2,
          mutation_scale=9, connectionstyle="arc3"):
    patch = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=width,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=2,
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
            linewidth=1.2,
            edgecolor=GRID,
            facecolor=LIGHT,
        )
    )
    ax.text(0.05, 0.90, title, ha="left", va="top", fontsize=10.2,
            color=NAVY, weight="bold")
    ax.text(0.05, 0.74, subtitle, ha="left", va="top", fontsize=7.7,
            color=GRAY)


def draw_baseline(ax):
    stage_panel(ax, "S0  Fixed replay", "No endpoint evidence")
    node(ax, (0.08, 0.39), 0.23, 0.18, "camera", edge=GRAY)
    node(ax, (0.39, 0.39), 0.23, 0.18, "absolute\nreplay", edge=ORANGE)
    node(ax, (0.70, 0.39), 0.22, 0.18, "robot", edge=RED)
    arrow(ax, (0.31, 0.48), (0.39, 0.48))
    arrow(ax, (0.62, 0.48), (0.70, 0.48))
    ax.text(0.50, 0.22, "command sent ≠ physical success", ha="center",
            va="center", fontsize=8.2, color=RED, weight="bold")


def draw_contact(ax):
    stage_panel(ax, "S1  Door contact prefix", "Relative replay + contact proof")
    node(ax, (0.05, 0.49), 0.24, 0.15, "HDF5 + MP4", edge=TEAL)
    node(ax, (0.37, 0.49), 0.25, 0.15, "demo compiler", edge=BLUE)
    node(ax, (0.70, 0.49), 0.24, 0.15, "SE(3) + IK", edge=GREEN)
    arrow(ax, (0.29, 0.565), (0.37, 0.565))
    arrow(ax, (0.62, 0.565), (0.70, 0.565))
    node(ax, (0.20, 0.20), 0.26, 0.15, "aperture", edge=ORANGE)
    node(ax, (0.56, 0.20), 0.27, 0.15, "5 mm proof", edge=ORANGE)
    arrow(ax, (0.82, 0.49), (0.70, 0.35), color=ORANGE,
          connectionstyle="arc3,rad=0.18")
    arrow(ax, (0.46, 0.275), (0.56, 0.275), color=ORANGE)
    arrow(ax, (0.56, 0.24), (0.42, 0.24), color=ORANGE,
          connectionstyle="arc3,rad=-0.28")


def draw_door(ax):
    stage_panel(ax, "S2  Door endpoint", "Verified open + close")
    node(ax, (0.04, 0.50), 0.22, 0.13, "Record3D\nRGB-D", edge=TEAL)
    node(ax, (0.32, 0.50), 0.25, 0.13, "Tag/PnP +\nRANSAC plane", edge=BLUE)
    node(ax, (0.66, 0.50), 0.28, 0.13, "registered\nendpoint", edge=GREEN)
    arrow(ax, (0.26, 0.565), (0.32, 0.565))
    arrow(ax, (0.57, 0.565), (0.66, 0.565))
    node(ax, (0.05, 0.26), 0.21, 0.13, "wrist RGB\nfeature", edge=TEAL)
    node(ax, (0.33, 0.26), 0.23, 0.13, "controller +\ncheckpoints", edge=ORANGE)
    node(ax, (0.64, 0.26), 0.21, 0.13, "RPC/CAN\nrobot", edge=RED)
    arrow(ax, (0.26, 0.325), (0.33, 0.325))
    arrow(ax, (0.56, 0.325), (0.64, 0.325))
    arrow(ax, (0.445, 0.50), (0.445, 0.39), color=BLUE)
    arrow(ax, (0.745, 0.39), (0.79, 0.50), color=GREEN,
          connectionstyle="arc3,rad=-0.16")
    node(ax, (0.31, 0.07), 0.38, 0.11, "JSON + tests + Git", edge=GRAY,
         fontsize=7.8)
    arrow(ax, (0.50, 0.26), (0.50, 0.18), color=GRAY)


def draw_cap(ax):
    stage_panel(ax, "S3  Cap transfer", "Verified removal + held transport")
    node(ax, (0.03, 0.55), 0.19, 0.12, "one click", edge=GRAY)
    node(ax, (0.27, 0.55), 0.22, 0.12, "OpenCV\nrelation", edge=BLUE)
    node(ax, (0.55, 0.55), 0.19, 0.12, "RGB-D +\ntag", edge=TEAL)
    node(ax, (0.79, 0.55), 0.17, 0.12, "3-D target", edge=GREEN)
    arrow(ax, (0.22, 0.61), (0.27, 0.61))
    arrow(ax, (0.49, 0.61), (0.55, 0.61))
    arrow(ax, (0.74, 0.61), (0.79, 0.61))
    node(ax, (0.04, 0.34), 0.22, 0.12, "7 probes", edge=ORANGE)
    node(ax, (0.30, 0.34), 0.28, 0.12, "local\nJacobian", edge=ORANGE,
         fontsize=7.4)
    node(ax, (0.62, 0.34), 0.28, 0.12, "MJCF +\nMuJoCo", edge=GREEN,
         fontsize=7.4)
    arrow(ax, (0.26, 0.40), (0.30, 0.40))
    arrow(ax, (0.58, 0.40), (0.62, 0.40))
    arrow(ax, (0.875, 0.55), (0.76, 0.46), color=GREEN,
          connectionstyle="arc3,rad=0.15")
    node(ax, (0.11, 0.13), 0.25, 0.12, "RPC/CAN", edge=RED)
    node(ax, (0.43, 0.13), 0.45, 0.12, "aperture + FK\n+ RGB-D audit", edge=GREEN,
         fontsize=7.2)
    arrow(ax, (0.69, 0.34), (0.29, 0.25), color=RED,
          connectionstyle="arc3,rad=0.14")
    arrow(ax, (0.36, 0.19), (0.43, 0.19), color=GREEN)


def draw_plateau(ax):
    stage_panel(ax, "S4  Release branch", "No promoted placement gain")
    node(ax, (0.07, 0.49), 0.24, 0.15, "held cap", edge=GREEN)
    node(ax, (0.38, 0.49), 0.24, 0.15, "release\nmotion", edge=ORANGE)
    node(ax, (0.69, 0.49), 0.24, 0.15, "placement\nevaluator", edge=RED)
    arrow(ax, (0.31, 0.565), (0.38, 0.565))
    arrow(ax, (0.62, 0.565), (0.69, 0.565))
    node(ax, (0.27, 0.22), 0.46, 0.15, "insufficient support evidence", edge=RED,
         face="#FFF5F5", linestyle="--", fontsize=8.1)
    arrow(ax, (0.81, 0.49), (0.66, 0.37), color=RED, linestyle="--",
          connectionstyle="arc3,rad=0.16")
    ax.text(0.50, 0.10, "placement not promoted", ha="center", va="center",
            fontsize=8.8, color=RED, weight="bold")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "svg.hashsalt": "piper_robot_tool_graph_v1",
    })
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(
        2, 5,
        height_ratios=[1.03, 1.0],
        left=0.065,
        right=0.985,
        top=0.90,
        bottom=0.055,
        hspace=0.26,
        wspace=0.09,
    )

    ax = fig.add_subplot(grid[0, :])
    x = np.arange(5, dtype=float)
    y = np.asarray([0.09, 0.42, 0.72, 0.94, 0.94])
    ax.set_xlim(-0.25, 4.25)
    ax.set_ylim(0.0, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(NAVY)
    ax.spines[["left", "bottom"]].set_linewidth(1.4)
    ax.set_xticks(x)
    ax.set_xticklabels([
        "S0\nfixed replay",
        "S1\ndoor contact",
        "S2\ndoor endpoint",
        "S3\ncap transfer",
        "S4\nrelease attempt",
    ], fontsize=9.5)
    ax.set_yticks([])
    ax.set_ylabel(
        "Cumulative independently verified capability\n(qualitative; not a success-rate axis)",
        fontsize=11.5,
        color=NAVY,
        labelpad=17,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.9, linestyle=":")

    # Draw straight segments so the figure does not imply unobserved samples.
    ax.plot(x, y, color=BLUE, linewidth=3.3, zorder=3)
    ax.scatter(x, y, s=90, color="white", edgecolor=BLUE, linewidth=2.5, zorder=4)
    ax.plot(x[3:], y[3:], color=ORANGE, linewidth=4.2, zorder=5)

    annotations = [
        (0, y[0], "No independent\ntask endpoint", (0.02, 0.12), "left"),
        (1, y[1], "Non-empty closure\n+ 5 mm proof", (0.00, 0.12), "center"),
        (2, y[2], "Door open + close\nverified by fresh RGB-D", (0.00, 0.12), "center"),
        (3, y[3], "Cap removed 9.175 mm\nand held over 107.415 mm", (-0.02, -0.18), "center"),
        (4, y[4], "Placement remained\nunverified", (-0.02, -0.18), "center"),
    ]
    for xi, yi, label, offset, align in annotations:
        ax.annotate(
            label,
            (xi, yi),
            xytext=(xi + offset[0], yi + offset[1]),
            ha=align,
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=9.6,
            color=RED if xi == 4 else NAVY,
            weight="bold" if xi in (2, 3, 4) else "normal",
        )

    ax.text(
        3.50, 1.035,
        "additional branch complexity did not weaken the evidence gate",
        ha="center", va="bottom", fontsize=9.4, color=ORANGE,
    )
    ax.annotate(
        "",
        xy=(4.02, 0.975),
        xytext=(3.02, 0.975),
        arrowprops=dict(arrowstyle="->", color=ORANGE, linewidth=1.5),
    )

    drawers = [draw_baseline, draw_contact, draw_door, draw_cap, draw_plateau]
    for column, draw in enumerate(drawers):
        draw(fig.add_subplot(grid[1, column]))

    fig.suptitle(
        "Program learning expanded the robot's tool graph until evidence-backed capability plateaued",
        x=0.065, y=0.965, ha="left", fontsize=17, color=NAVY, weight="bold",
    )
    fig.text(
        0.067, 0.925,
        "Observed milestones from the incubator-door and culture-cap case studies; graph snapshots are simplified from the audited code paths.",
        ha="left", va="center", fontsize=10.5, color=GRAY,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    png = OUTPUT / "tool_graph_learning_curve.png"
    svg = OUTPUT / "tool_graph_learning_curve.svg"
    metadata = {"Creator": "piper_robot manuscript figure generator", "Date": None}
    fig.savefig(png, dpi=180, facecolor="white", metadata=metadata)
    fig.savefig(svg, facecolor="white", metadata=metadata)
    plt.close(fig)
    # Matplotlib emits spaces before newlines in multiline SVG path data.
    # Normalize them so repository whitespace checks remain useful.
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    print(png)
    print(svg)


if __name__ == "__main__":
    main()
