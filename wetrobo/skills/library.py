"""SkillLibrary — parametric motor skills WetRobo composes and refines.

Skills are plain code (pick / place) built on LabEnv primitives, parameterised by a
small set of numbers (approach heights, grasp depth, place height, grasp tolerance).
The parameters live in a JSON store so they persist and improve across attempts — this
is how WetRobo "learns by performing": not weights, but refined skill parameters. The
DeterministicPlanner reads a failure's cause and nudges these numbers; the LLM hook can
rewrite the skill code itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_PARAMS = {
    "approach_z": 0.22,     # height above the grasp point to pre-position [m]
    "lift_z": 0.30,         # height to lift the object to after grasping [m]
    "transit_z": 0.38,      # height to carry over the destination [m]
    "place_z": 0.34,        # grasp-site height at release over the destination [m]
    "grasp_tol_m": 0.035,   # kinematic-grasp gate: max grasp-site-to-object distance
    "grasp_settle": 60,
}


class SkillLibrary:
    def __init__(self, params_path: str | Path | None = None):
        self.params_path = Path(params_path) if params_path else None
        self.params = dict(DEFAULT_PARAMS)
        if self.params_path and self.params_path.exists():
            self.params.update(json.loads(self.params_path.read_text()))

    def save(self) -> None:
        if self.params_path:
            self.params_path.parent.mkdir(parents=True, exist_ok=True)
            self.params_path.write_text(json.dumps(self.params, indent=2))

    # --- skills ------------------------------------------------------------ #
    def pick(self, env, arm: str, obj_body: str, grasp_point_world, grasp_local) -> dict:
        """Grasp ``obj_body`` at the world ``grasp_point_world`` (from perception).

        The object is frozen during the reach (a bench object stays put until grasped),
        then the position-gated grasp attaches it only if the gripper actually reached
        the object's true grasp point — so an accurate (CAD) target grasps and an
        inaccurate (vision) one misses. Returns the grasp result dict."""
        p = self.params
        gp = np.asarray(grasp_point_world, float)
        env.set_gripper(arm, 1.0, settle_steps=p["grasp_settle"])
        env.freeze(obj_body)
        approach = env.move_ee(arm, [gp[0], gp[1], p["approach_z"]])
        reach = env.move_ee(arm, [gp[0], gp[1], gp[2]])
        res = env.grasp(arm, obj_body, grasp_local=grasp_local, tol=p["grasp_tol_m"])
        env.unfreeze(obj_body)
        lift = None
        if res["grasped"]:
            lift = env.move_ee(arm, [gp[0], gp[1], p["lift_z"]])
        # Keep the motion evidence: downstream evaluation should distinguish a bad
        # Cartesian reach from a well-executed reach aimed at a bad perceived pose.
        return {**res, "approach": approach, "reach": reach, "lift": lift}

    def place(self, env, arm: str, dest_xy) -> dict:
        """Carry a held object over ``dest_xy`` and release it there. Carries high to
        clear obstacles, lowers to the reachable release height, then releases."""
        p = self.params
        dx, dy = float(dest_xy[0]), float(dest_xy[1])
        env.move_ee(arm, [dx, dy, p["transit_z"]])
        r = env.move_ee(arm, [dx, dy, p["place_z"]])
        env.release(arm)
        return {"place_ee_err_m": r["ee_err_m"], "released_at": r["ee_pos"]}
