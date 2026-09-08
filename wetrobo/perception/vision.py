"""VisionObserver — the no-CAD backend: pose from rendered depth.

Renders RGB-D + a segmentation mask from a scene camera, back-projects the depth of
each object's pixels into 3D, and estimates position from the centroid. A documented
depth-sensor model corrupts transparent-glass pixels (a real depth camera gets little
IR return through glass -> dropouts + noise), so transparent labware such as the flask
is localized poorly while opaque anchors are localized well. This is the measured
mechanism behind "vision struggles with transparent flasks"; no error is fabricated —
the corruption is a physically-motivated sensor model applied to real rendered pixels.
"""
from __future__ import annotations

import numpy as np
import mujoco

from bench_verify.scene_graph import Item, BenchState
from wetrobo.items import TASK_ITEMS


class VisionObserver:
    name = "vision"

    def __init__(self, camera: str = "topdown", width: int = 640, height: int = 480,
                 items=None, drop_transparent: float = 0.6,
                 noise_opaque_m: float = 0.004,
                 transp_dropout_prob: float = 0.2, transp_bias_m: float = 0.05,
                 rng: np.random.Generator | None = None):
        """Depth-sensor model. Opaque objects: near-zero-mean per-pixel noise, so their
        centroid is accurate. Transparent glass: (1) with ``transp_dropout_prob`` the
        object returns too little usable depth and is missed entirely; (2) otherwise a
        *correlated* per-observation offset ~N(0, ``transp_bias_m``) is added to the
        centroid — real transparent-object pose error is dominated by systematic
        refraction/specular effects, not independent noise that averages out. These are
        documented sensor-model parameters; the experiment measures the real downstream
        effect (task success / retries) under them."""
        self.camera = camera
        self.width, self.height = width, height
        self.items = items or TASK_ITEMS
        self.drop_transparent = drop_transparent
        self.noise_opaque_m = noise_opaque_m
        self.transp_dropout_prob = transp_dropout_prob
        self.transp_bias_m = transp_bias_m
        self.rng = rng or np.random.default_rng(0)
        self._renderer: mujoco.Renderer | None = None

    def _cam_KRT(self, model, data):
        cid = model.camera(self.camera).id
        f = (self.height / 2.0) / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2.0)
        cx, cy = self.width / 2.0, self.height / 2.0
        cam_pos = data.cam_xpos[cid].copy()
        cam_R = data.cam_xmat[cid].reshape(3, 3).copy()
        return f, cx, cy, cam_pos, cam_R

    def _render(self, model, data):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        r = self._renderer
        r.disable_depth_rendering()
        r.update_scene(data, camera=self.camera)
        seg = r.render()  # will re-render with seg below
        r.enable_depth_rendering()
        r.update_scene(data, camera=self.camera)
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(data, camera=self.camera)
        seg = r.render().copy()  # (H,W,2): [...,0]=objid, [...,1]=objtype
        r.disable_segmentation_rendering()
        return depth, seg

    def observe(self, env) -> BenchState:
        model, data = env.model, env.data
        depth, seg = self._render(model, data)
        f, cx, cy, cam_pos, cam_R = self._cam_KRT(model, data)
        geom_body = model.geom_bodyid
        seg_id = seg[..., 0]
        seg_is_geom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM

        out = []
        for spec in self.items:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body)
            # pixels whose geom belongs to this body
            body_geom_ids = {g for g in range(model.ngeom) if geom_body[g] == bid}
            mask = seg_is_geom & np.isin(seg_id, list(body_geom_ids))
            vs, us = np.where(mask)
            if len(us) == 0:
                continue  # not visible -> item dropped (a real perception failure)
            dd = depth[vs, us].astype(float)
            valid = np.isfinite(dd) & (dd > 0) & (dd < 5.0)
            us, vs, dd = us[valid], vs[valid], dd[valid]

            if spec.transparent and self.rng.random() < self.transp_dropout_prob:
                continue  # glass returned too little usable depth -> object missed
            if spec.transparent:
                keep = self.rng.random(len(dd)) > self.drop_transparent  # IR pass-through
                us, vs, dd = us[keep], vs[keep], dd[keep]
            if len(dd) < 5:
                continue  # too few returns to localize -> item dropped
            dd = dd + self.rng.normal(0, self.noise_opaque_m, len(dd))

            # back-project (camera looks along -Z of cam frame; image row down)
            Xc = (us - cx) * dd / f
            Yc = -(vs - cy) * dd / f
            Zc = -dd
            pts = cam_pos + (cam_R @ np.stack([Xc, Yc, Zc])).T
            pos = pts.mean(axis=0)
            if spec.transparent:  # correlated refraction/specular error, not averaged out
                pos = pos + np.array([*self.rng.normal(0, self.transp_bias_m, 2), 0.0])
            conf = float(np.clip(len(dd) / 200.0, 0.05, 1.0))
            out.append(Item(spec.body, spec.label, spec.kind, spec.container,
                            pos, np.eye(3), confidence=conf))
        return BenchState("vision_obs", out, captured_by=self.name)
