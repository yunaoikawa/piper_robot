"""Self-critique + repair of an authored CAD scene.

The premise: if the authored layout is physically implausible we can *detect* it (objects
interpenetrating, floating above / sunk into their support, out of the arm's reach, or
resting on a provisional-size guess) and *repair* the cheap-to-fix parts by projecting
the layout onto those constraints. This is the perceive->verify->reflect->refine loop of
the planner, applied to CAD authoring: physical plausibility is the verification signal.

What repair CAN fix from geometry alone: seating objects on a support, separating
overlaps, flagging unreachable/te provisional items. What it CANNOT invent: correct
absolute metric scale (that still needs the anchor — fiducial / measurement / LiDAR).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from bench_verify.scene_graph import Item, BenchState
from wetrobo.perception.catalog import LabwareCatalog


def half_extents(entry) -> np.ndarray:
    """Axis-aligned half-extents [m] of an entry's proxy geom (upright assumption)."""
    if entry.shape == "cyl":
        r, h = entry.dims
        return np.array([r, r, h / 2])
    if entry.shape == "box":
        return np.array(entry.dims) / 2
    return np.array([0.05, 0.05, 0.05])


def critique_scene(state: BenchState, catalog: LabwareCatalog, *,
                   supports=(0.0, 0.15), reach: float = 0.6,
                   base_xy=(0.0, 0.0), overlap_tol: float = 2e-3) -> list[dict]:
    """Return a list of plausibility issues, each a dict {type, who, amount}."""
    rec = [(it, catalog.get(it.container), half_extents(catalog.get(it.container)))
           for it in state.items]
    issues: list[dict] = []

    # 1. interpenetration (AABB overlap on all 3 axes)
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            (a, _, ha), (b, _, hb) = rec[i], rec[j]
            ov = (ha + hb) - np.abs(np.asarray(a.t) - np.asarray(b.t))
            if np.all(ov > overlap_tol):
                issues.append({"type": "interpenetration",
                               "who": (a.label, b.label), "amount": float(ov.min())})

    # 2. floating above / sunk below the nearest support surface
    for it, e, h in rec:
        base_z = it.t[2] - h[2]
        nearest = min(supports, key=lambda s: abs(base_z - s))
        gap = base_z - nearest
        if abs(gap) > 0.01:
            issues.append({"type": "floating" if gap > 0 else "sunk_into_support",
                           "who": it.label, "amount": float(gap)})

    # 3. graspable item outside the arm's planar reach
    for it, e, h in rec:
        if e.graspable:
            d = float(np.linalg.norm(np.asarray(it.t[:2]) - np.asarray(base_xy)))
            if d > reach:
                issues.append({"type": "out_of_reach", "who": it.label, "amount": d})

    # 4. resting on an unverified (provisional) size prior
    for it, e, h in rec:
        if e.provisional_dims:
            issues.append({"type": "provisional_size", "who": it.label, "amount": None})
    return issues


def repair_scene(state: BenchState, catalog: LabwareCatalog, *,
                 supports=(0.0, 0.15), iters: int = 40) -> BenchState:
    """Project the layout onto the plausibility constraints it can satisfy from geometry:
    (a) seat every object base on its nearest support, (b) iteratively push apart any
    interpenetrating pair along their axis of least overlap (x or y only — vertical is
    fixed by the support). Absolute scale is untouched (needs the metric anchor)."""
    ent = {it.item_id: catalog.get(it.container) for it in state.items}
    half = {it.item_id: half_extents(ent[it.item_id]) for it in state.items}
    t = {it.item_id: np.asarray(it.t, float).copy() for it in state.items}

    # (a) seat on nearest support
    for it in state.items:
        h = half[it.item_id]
        base_z = t[it.item_id][2] - h[2]
        nearest = min(supports, key=lambda s: abs(base_z - s))
        t[it.item_id][2] = nearest + h[2]

    # (b) resolve horizontal interpenetration (skip fixed anchors: non-graspable & heavy)
    ids = [it.item_id for it in state.items]
    movable = {it.item_id: ent[it.item_id].graspable for it in state.items}
    for _ in range(iters):
        moved = False
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                d = t[a] - t[b]
                ov = (half[a] + half[b]) - np.abs(d)
                if ov[0] > 2e-3 and ov[1] > 2e-3 and ov[2] > 2e-3:  # overlapping
                    ax = 0 if ov[0] < ov[1] else 1                  # least-overlap horiz axis
                    push = (ov[ax] + 5e-3) * (1.0 if d[ax] >= 0 else -1.0)
                    # move the graspable/lighter one; if both fixed, split
                    if movable[a] and not movable[b]:
                        t[a][ax] += push
                    elif movable[b] and not movable[a]:
                        t[b][ax] -= push
                    else:
                        t[a][ax] += push / 2; t[b][ax] -= push / 2
                    moved = True
        if not moved:
            break

    items = [replace(it, t=t[it.item_id]) for it in state.items]
    return BenchState(state.bench_id + "_repaired", items,
                      frame=state.frame, captured_by="critique.repair_scene")
