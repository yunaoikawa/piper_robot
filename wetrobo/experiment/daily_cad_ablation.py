"""Daily-CAD ablation — the paper's experiment.

For each simulated day (a flask position) and each seed, WetRobo runs the
flask -> incubator task twice: once with the day's CAD (exact poses) and once with
vision only (rendered-depth perception, poor on the transparent flask). Every attempt
is a real MuJoCo rollout, logged to JSONL. report.py turns the log into the figure.

    python -m wetrobo.experiment.daily_cad_ablation --days 6 --seeds 4 --out runs/ablation.jsonl
"""
from __future__ import annotations

import argparse

import numpy as np
import mujoco

from wetrobo.sim.lab_env import LabEnv
from wetrobo.perception.cad import CADObserver
from wetrobo.perception.vision import VisionObserver
from wetrobo.skills.library import SkillLibrary
from wetrobo.agent.planner import DeterministicPlanner
from wetrobo.tasks.flask_to_incubator import incubator_goal
from wetrobo.episode_log import EpisodeLog
from wetrobo.experiment.layouts import daily_flask_positions


def _set_flask(env, xy):
    adr = env.model.jnt_qposadr[env.model.joint("flask_joint").id]
    env.data.qpos[adr:adr + 7] = [xy[0], xy[1], 0.0, 1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)


def run_episode(condition, flask_xy, seed, max_attempts, log, day):
    env = LabEnv()
    _set_flask(env, flask_xy)
    goal = incubator_goal(env)
    skills = SkillLibrary()
    planner = DeterministicPlanner(skills, arm="right", max_attempts=max_attempts)
    if condition == "cad":
        cad = CADObserver()
        observer = lambda e: cad.observe(e)
    else:
        rng = np.random.default_rng(seed)
        observer = lambda e: VisionObserver(rng=rng).observe(e)
    meta = {"condition": condition, "day": day, "seed": seed,
            "flask_xy": [round(flask_xy[0], 4), round(flask_xy[1], 4)]}
    res = planner.run_episode(env, observer, goal, log=log, meta={**meta, "kind": "attempt"})
    log.add(kind="episode", **meta, success=res["success"],
            attempts=res["attempts"], outcome=res["outcome"],
            stage_reached=res["stage_reached"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--layout-seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ablation.jsonl")
    args = ap.parse_args()

    positions = daily_flask_positions(args.days, seed=args.layout_seed)
    log = EpisodeLog(args.out)
    summary = {"cad": [], "vision": []}
    for day, xy in enumerate(positions):
        for seed in range(args.seeds):
            for cond in ("cad", "vision"):
                res = run_episode(cond, xy, seed, args.max_attempts, log, day)
                summary[cond].append((res["success"], res["attempts"]))
                print(f"day{day} seed{seed} {cond:6s} "
                      f"success={res['success']} attempts={res['attempts']}")

    print("\n=== summary (real rollouts) ===")
    for cond, rows in summary.items():
        succ = np.mean([s for s, _ in rows])
        att = np.mean([a for s, a in rows if s]) if any(s for s, _ in rows) else float("nan")
        print(f"{cond:6s} success_rate={succ*100:5.1f}%  mean_attempts_to_success={att:.2f}  n={len(rows)}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
