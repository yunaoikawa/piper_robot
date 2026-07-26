"""DeterministicPlanner — the reproducible attempt/verify/reflect/refine loop.

This is the controller used for the daily-CAD ablation: given a task, an env, an
observer (CAD or vision), and a SkillLibrary, it perceives, executes pick+place,
verifies the real sim end-state, and on failure reflects on the cause and either
re-perceives (perception miss) or nudges a skill parameter (placement miss), retrying
up to a budget. "Learning by performing" = these measured parameter refinements, which
persist in the SkillLibrary across episodes.

The LLM skill-author (agent.llm_agent) is an optional escalation on top of this and is
never required for the ablation to run.
"""
from __future__ import annotations

import numpy as np

from wetrobo.tasks.flask_to_incubator import grasp_point_from_obs, success
from wetrobo.items import FLASK


class DeterministicPlanner:
    def __init__(self, skills, arm: str = "right", max_attempts: int = 5):
        self.skills = skills
        self.arm = arm
        self.max_attempts = max_attempts

    def run_episode(self, env, observer, goal, *, log=None, meta=None) -> dict:
        meta = meta or {}
        attempts = 0
        outcome = "no_attempt"
        best_stage = "started"
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            if attempt > 1:
                env.home_arm(self.arm)             # clean pose before re-attempting
            obs = observer(env)
            gp = grasp_point_from_obs(obs)
            perceived = None if gp is None else gp[:2].tolist()

            if gp is None:
                outcome = "not_perceived"          # perception dropout -> re-perceive
                self._record(log, meta, attempt, outcome, perceived, None, None,
                             stage_reached="started")
                continue

            best_stage = self._later_stage(best_stage, "perceived")
            pick = self.skills.pick(env, self.arm, FLASK.body, gp, FLASK.grasp_local)
            reach = pick.get("reach", {})
            reached = bool(reach.get("reached", False))
            if reached:
                best_stage = self._later_stage(best_stage, "reached")
            if not pick["grasped"]:
                outcome = "grasp_miss"             # gripper missed -> re-perceive
                self._record(log, meta, attempt, outcome, perceived, pick["dist_m"], None,
                             stage_reached="reached" if reached else "perceived",
                             reach_ee_err_m=reach.get("ee_err_m"),
                             grasp_along_m=pick.get("along_m"),
                             grasp_perp_m=pick.get("perp_m"))
                continue

            best_stage = self._later_stage(best_stage, "grasped")
            self.skills.place(env, self.arm, goal.place_xy)
            env.step(800)
            s = success(env, goal)
            if s["success"]:
                outcome = "success"
                best_stage = "placed"
                self._record(log, meta, attempt, outcome, perceived, pick["dist_m"], s["flask_pos"],
                             stage_reached="placed", reach_ee_err_m=reach.get("ee_err_m"),
                             grasp_along_m=pick.get("along_m"),
                             grasp_perp_m=pick.get("perp_m"))
                return {"success": True, "attempts": attempts, "outcome": outcome,
                        "stage_reached": best_stage}

            # grasped but not placed inside -> reflect on placement and refine params
            outcome = "place_miss"
            self._reflect_place(env, goal, s["flask_pos"])
            self._record(log, meta, attempt, outcome, perceived, pick["dist_m"], s["flask_pos"],
                         stage_reached="grasped", reach_ee_err_m=reach.get("ee_err_m"),
                         grasp_along_m=pick.get("along_m"),
                         grasp_perp_m=pick.get("perp_m"))

        return {"success": False, "attempts": attempts, "outcome": outcome,
                "stage_reached": best_stage}

    @staticmethod
    def _later_stage(a: str, b: str) -> str:
        order = {"started": 0, "perceived": 1, "reached": 2, "grasped": 3, "placed": 4}
        return a if order[a] >= order[b] else b

    def _reflect_place(self, env, goal, flask_pos) -> None:
        """Nudge placement parameters toward the goal from the observed miss (the
        parameter-refinement form of learning-by-performing)."""
        p = self.skills.params
        fp = np.asarray(flask_pos)
        if fp[0] < goal.lo[0]:                 # fell short of the interior front
            goal.place_xy[0] = min(goal.place_xy[0] + 0.02, goal.hi[0] - 0.02)
        if fp[2] > goal.shelf_z + 0.08:        # released too high (toppled/bounced)
            p["place_z"] = max(p["place_z"] - 0.02, goal.shelf_z + 0.05)
        self.skills.save()

    @staticmethod
    def _record(log, meta, attempt, outcome, perceived_xy, grasp_dist_m, flask_pos,
                **evidence):
        if log is None:
            return
        log.add(attempt=attempt, outcome=outcome, perceived_xy=perceived_xy,
                grasp_dist_m=grasp_dist_m, flask_pos=flask_pos, **evidence, **meta)
