"""Photo-grounded pose fitting: place each object so it REPROJECTS onto where it is seen
in the calibrated photo, while staying physically plausible.

Given a metric `Camera` (from `calibrate.py`), the per-object observed keypoints (at
minimum its base-contact pixel; optionally a top pixel for a height check) and its
support-plane height, we solve each object's (x,y) by least-squares to minimise
reprojection error, with a soft non-overlap penalty so the fit stays plausible. This is
the correction `repair_scene` couldn't do: the objective is the photo, not just internal
consistency.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import least_squares

from wetrobo.perception.camera import Camera
from wetrobo.perception.catalog import LabwareCatalog
from wetrobo.perception.critique import half_extents
from bench_verify.scene_graph import Item, BenchState


class ObjObs:
    """One object's photo observation: container id, base-contact pixel, support-plane
    height [m], and optional top-center pixel (for a reprojected-height residual)."""
    def __init__(self, container, det_id, base_uv, support_z, top_uv=None):
        self.container, self.det_id = container, det_id
        self.base_uv = np.asarray(base_uv, float)
        self.support_z = float(support_z)
        self.top_uv = None if top_uv is None else np.asarray(top_uv, float)


def fit_poses(cam: Camera, obs: list[ObjObs], catalog: LabwareCatalog, *,
              overlap_w: float = 300.0) -> tuple[BenchState, dict]:
    """Solve object (x,y) on their support planes to fit the photo + stay non-overlapping.

    Init from direct base back-projection (exact for the base pixel), then least-squares
    refine so top-pixel and non-overlap residuals are traded off. Returns (state, info)."""
    n = len(obs)
    hz = [half_extents(catalog.get(o.container))[2] for o in obs]
    hxy = [half_extents(catalog.get(o.container))[:2] for o in obs]
    z = [o.support_z + hz[i] for i, o in enumerate(obs)]

    # initial (x,y) = back-project each base pixel onto its support plane
    xy0 = np.array([cam.backproject_to_plane(o.base_uv, o.support_z)[:2] for o in obs])

    def residuals(p):
        xy = p.reshape(n, 2)
        res = []
        for i, o in enumerate(obs):
            base_w = np.array([xy[i, 0], xy[i, 1], o.support_z])
            res += list(cam.project(base_w)[0] - o.base_uv)          # base reprojection
            if o.top_uv is not None:
                top_w = np.array([xy[i, 0], xy[i, 1], z[i] + hz[i]])
                res += list(cam.project(top_w)[0] - o.top_uv)        # top reprojection
        # soft non-overlap penalty (x,y AABB)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.abs(xy[i] - xy[j])
                ov = (np.array(hxy[i]) + np.array(hxy[j])) - d
                pen = np.clip(ov, 0, None)
                res += list(overlap_w * pen)
        return np.array(res)

    sol = least_squares(residuals, xy0.reshape(-1), method="lm" if overlap_w == 0 else "trf")
    xy = sol.x.reshape(n, 2)
    items = [catalog.to_item(o.container, o.det_id, [xy[i, 0], xy[i, 1], z[i]],
                             np.eye(3), 1.0) for i, o in enumerate(obs)]
    reproj_px = float(np.sqrt(np.mean([
        np.sum((cam.project([xy[i, 0], xy[i, 1], o.support_z])[0] - o.base_uv) ** 2)
        for i, o in enumerate(obs)])))
    state = BenchState("photo_fitted", items, captured_by="fit.fit_poses")
    return state, {"init_xy": xy0, "reproj_rms_px": reproj_px, "cost": float(sol.cost)}
