"""Run one WetRobo episode of the flask -> incubator task in MuJoCo.

    python -m wetrobo.run_wetrobo --observer cad
    python -m wetrobo.run_wetrobo --observer vision --seed 3 --max-attempts 5

The observer choice is the paper's independent variable: `cad` uses the day's CAD
(exact poses), `vision` uses rendered-depth perception (poor on the transparent flask).
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


def set_flask(env, xy):
    adr = env.model.jnt_qposadr[env.model.joint("flask_joint").id]
    env.data.qpos[adr:adr + 7] = [xy[0], xy[1], 0.0, 1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)


def make_observer(kind, seed):
    if kind == "cad":
        obs = CADObserver()
        return lambda env: obs.observe(env)
    rng = np.random.default_rng(seed)
    return lambda env: VisionObserver(rng=rng).observe(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observer", choices=["cad", "vision"], default="cad")
    ap.add_argument("--flask-x", type=float, default=0.20)
    ap.add_argument("--flask-y", type=float, default=-0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--params", default=None, help="SkillLibrary JSON param store")
    ap.add_argument("--log", default=None, help="append JSONL episode log here")
    args = ap.parse_args()

    env = LabEnv()
    set_flask(env, (args.flask_x, args.flask_y))
    goal = incubator_goal(env)
    skills = SkillLibrary(args.params)
    planner = DeterministicPlanner(skills, arm="right", max_attempts=args.max_attempts)
    log = EpisodeLog(args.log) if args.log else None

    observer = make_observer(args.observer, args.seed)
    res = planner.run_episode(env, observer, goal, log=log,
                              meta={"observer": args.observer, "seed": args.seed,
                                    "flask_xy": [args.flask_x, args.flask_y]})
    print(f"observer={args.observer} success={res['success']} "
          f"attempts={res['attempts']} outcome={res['outcome']}")
    return 0 if res["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
