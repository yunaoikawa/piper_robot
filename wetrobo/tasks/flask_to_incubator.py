"""Task: put the flask in the incubator.

Everything geometric is derived from the loaded MJCF at runtime (Principle 5: no
hardcoded coordinates): the goal region is the incubator's interior footprint computed
from its shelf/floor geoms, and success is measured on the real sim end-state (the
flask body's pose), not asserted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

from wetrobo.items import FLASK, INCUBATOR


def _geom_world_aabb(model, data, name):
    gid = model.geom(name).id
    pos = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
    sz = model.geom_size[gid]
    corners = np.array([[sx * sz[0], sy * sz[1], sz[2]]
                        for sx in (-1, 1) for sy in (-1, 1)])
    w = (R @ corners.T).T + pos
    return w.min(0), w.max(0)


@dataclass
class Goal:
    lo: np.ndarray          # interior footprint min (world)
    hi: np.ndarray          # interior footprint max (world)
    shelf_z: float          # top of the shelf the flask should rest on
    place_xy: np.ndarray    # a reachable point inside the interior to release over

    def contains(self, pos, xy_margin=0.02, z_margin=0.10) -> bool:
        return (self.lo[0] - xy_margin < pos[0] < self.hi[0] + xy_margin and
                self.lo[1] - xy_margin < pos[1] < self.hi[1] + xy_margin and
                self.shelf_z - 0.06 < pos[2] < self.shelf_z + z_margin)


def incubator_goal(env, reach_x: float = 0.44) -> Goal:
    """Interior footprint from the incubator shelf + floor geoms; the release point is
    clamped to the arm's forward reach so it stays physically achievable."""
    lo1, hi1 = _geom_world_aabb(env.model, env.data, "fridge_shelf")
    lo2, hi2 = _geom_world_aabb(env.model, env.data, "fridge_int_bottom")
    lo, hi = np.minimum(lo1, lo2), np.maximum(hi1, hi2)
    shelf_z = float(max(hi1[2], hi2[2]))
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    place_x = min(cx, reach_x)                       # keep within reach
    place_x = max(place_x, lo[0] + 0.05)             # but inside the front edge
    return Goal(lo, hi, shelf_z, np.array([place_x, cy]))


def grasp_point_from_obs(observed, spec=FLASK):
    """World grasp point for the flask given a perceived BenchState: the perceived XY
    (this is the CAD/vision-sensitive part) at the neck height above the bench (flasks
    stand on the table, so the grasp height is known from the object's geometry)."""
    item = {it.label: it for it in observed.items}.get(spec.label)
    if item is None:
        return None
    return np.array([item.t[0], item.t[1], spec.grasp_local[2]])


def success(env, goal: Goal, body: str = FLASK.body) -> dict:
    pos, _ = env.get_object_pose(body)
    ok = goal.contains(pos)
    return {"success": bool(ok), "flask_pos": pos.tolist()}
