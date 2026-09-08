"""Watch a WetRobo episode live in MuJoCo's interactive viewer.

macOS needs the ``mjpython`` launcher for the passive viewer:

    mjpython -m wetrobo.view_wetrobo --observer cad
    mjpython -m wetrobo.view_wetrobo --observer vision --seed 3 --max-attempts 6

Same rollout as ``run_wetrobo`` (real physics, real SceneVerifier end-state) — this only
adds a window: ``env.step`` is wrapped to sync the viewer and pace to ~real time so you
can watch the perceive -> pick -> place -> verify -> retry loop actually happen.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer

from wetrobo.sim.lab_env import LabEnv
from wetrobo.perception.cad import CADObserver
from wetrobo.perception.vision import VisionObserver
from wetrobo.skills.library import SkillLibrary
from wetrobo.agent.planner import DeterministicPlanner
from wetrobo.tasks.flask_to_incubator import incubator_goal
from wetrobo.run_wetrobo import set_flask, make_observer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observer", choices=["cad", "vision"], default="cad")
    ap.add_argument("--flask-x", type=float, default=0.20)
    ap.add_argument("--flask-y", type=float, default=-0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--camera", default="lab_overview")
    ap.add_argument("--speed", type=float, default=1.0, help="playback rate (x real time)")
    args = ap.parse_args()

    env = LabEnv()
    set_flask(env, (args.flask_x, args.flask_y))
    goal = incubator_goal(env)
    skills = SkillLibrary(None)
    planner = DeterministicPlanner(skills, arm="right", max_attempts=args.max_attempts)
    observer = make_observer(args.observer, args.seed)

    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if cam_id >= 0:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

    dt = env.model.opt.timestep / max(args.speed, 1e-6)
    orig_step = env.step

    def viewed_step(n: int = 1):
        for _ in range(n):
            orig_step(1)
            if not viewer.is_running():
                raise KeyboardInterrupt
            viewer.sync()
            time.sleep(dt)

    env.step = viewed_step

    print(f"Viewer open — observer={args.observer}, seed={args.seed}. Close the window to stop.")
    try:
        res = planner.run_episode(env, observer, goal,
                                  meta={"observer": args.observer, "seed": args.seed})
        print(f"episode: success={res['success']} attempts={res['attempts']} "
              f"outcome={res['outcome']}")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
