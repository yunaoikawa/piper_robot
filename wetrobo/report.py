"""Build the paper figure/table from a daily-CAD ablation log — real rollouts only.

    python -m wetrobo.report --log runs/ablation.jsonl --out runs/figure.png

Reads the JSONL episode records and reports, per condition (with-CAD vs vision),
the task success rate and the mean number of attempts to success. No numbers are
synthesised; everything is aggregated from logged MuJoCo rollouts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wetrobo.episode_log import EpisodeLog


def aggregate(records):
    eps = [r for r in records if r.get("kind") == "episode"]
    out = {}
    preferred = ("cad", "vision", "oracle_cad", "measured_daily_cad",
                 "calibrated_vision_only", "tag_assisted_deployment")
    conditions = list(dict.fromkeys([*preferred, *(r.get("condition") for r in eps)]))
    for cond in conditions:
        if cond is None:
            continue
        rows = [r for r in eps if r["condition"] == cond]
        if not rows:
            continue
        succ = np.array([r["success"] for r in rows], float)
        att = np.array([r["attempts"] for r in rows if r["success"]], float)
        stages = ("perceived", "reached", "grasped", "placed")
        rank = {"started": 0, "perceived": 1, "reached": 2, "grasped": 3, "placed": 4}
        # Backward-compatible inference keeps old JSONL logs reportable.
        def reached_stage(row):
            if "stage_reached" in row:
                return row["stage_reached"]
            if row.get("success"):
                return "placed"
            return "grasped" if row.get("outcome") == "place_miss" else (
                "perceived" if row.get("outcome") == "grasp_miss" else "started")
        progress = {stage: float(np.mean([
            rank[reached_stage(r)] >= rank[stage] for r in rows
        ])) for stage in stages}
        out[cond] = {
            "n": len(rows),
            "success_rate": float(succ.mean()),
            "mean_attempts": float(att.mean()) if len(att) else float("nan"),
            "attempts_list": att.tolist(),
            "progress_rate": progress,
        }
    return out


def print_table(agg):
    print(f"\n{'condition':10s} {'n':>4} {'success%':>9} {'mean_attempts':>14} "
          f"{'perceived%':>11} {'reached%':>9} {'grasped%':>10}")
    print("-" * 76)
    for cond, s in agg.items():
        p = s["progress_rate"]
        print(f"{cond:10s} {s['n']:>4} {s['success_rate']*100:>8.1f}% "
              f"{s['mean_attempts']:>14.2f} {p['perceived']*100:>10.1f}% "
              f"{p['reached']*100:>8.1f}% {p['grasped']*100:>9.1f}%")


def make_figure(agg, path):
    order = ("cad", "vision", "oracle_cad", "measured_daily_cad",
             "calibrated_vision_only", "tag_assisted_deployment")
    conds = [c for c in order if c in agg] + [c for c in agg if c not in order]
    labels = {"cad": "oracle CAD", "vision": "vision only",
              "oracle_cad": "oracle CAD", "measured_daily_cad": "measured daily CAD",
              "calibrated_vision_only": "calibrated vision only",
              "tag_assisted_deployment": "tag-assisted deployment"}
    palette = plt.get_cmap("tab10")
    colors = {c: palette(i) for i, c in enumerate(conds)}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(conds))
    a1.bar(x, [agg[c]["success_rate"] * 100 for c in conds],
           color=[colors[c] for c in conds])
    a1.set_xticks(x); a1.set_xticklabels([labels.get(c, c) for c in conds], rotation=15)
    a1.set_ylabel("task success rate (%)"); a1.set_ylim(0, 105)
    a1.set_title("Success rate")
    for i, c in enumerate(conds):
        a1.text(i, agg[c]["success_rate"] * 100 + 2, f"{agg[c]['success_rate']*100:.0f}%",
                ha="center")
    a2.bar(x, [agg[c]["mean_attempts"] for c in conds], color=[colors[c] for c in conds])
    a2.set_xticks(x); a2.set_xticklabels([labels.get(c, c) for c in conds], rotation=15)
    a2.set_ylabel("mean attempts to success"); a2.set_title("Convergence speed")
    for i, c in enumerate(conds):
        v = agg[c]["mean_attempts"]
        a2.text(i, v + 0.05, f"{v:.2f}", ha="center")
    stages = ("perceived", "reached", "grasped", "placed")
    sx = np.arange(len(stages))
    width = 0.8 / max(1, len(conds))
    for i, c in enumerate(conds):
        vals = [agg[c]["progress_rate"][s] * 100 for s in stages]
        a3.bar(sx + (i - (len(conds) - 1) / 2) * width, vals, width,
               label=labels.get(c, c), color=colors[c])
    a3.set_xticks(sx); a3.set_xticklabels(stages, rotation=20)
    a3.set_ylim(0, 105); a3.set_ylabel("episodes reaching stage (%)")
    a3.set_title("Subgoal progress"); a3.legend(fontsize=7)
    fig.suptitle("Authoring the day's CAD improves WetRobo success and speed\n"
                 "(flask → incubator, MuJoCo rollouts)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/ablation.jsonl")
    ap.add_argument("--out", default="runs/figure.png")
    args = ap.parse_args()
    records = EpisodeLog.read(args.log)
    agg = aggregate(records)
    if not agg:
        print(f"no episode records in {args.log}")
        return
    print_table(agg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make_figure(agg, args.out)


if __name__ == "__main__":
    main()
