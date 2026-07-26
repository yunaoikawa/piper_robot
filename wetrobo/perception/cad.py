"""CADObserver — the "daily CAD" backend.

Reads exact object poses straight from the live sim (which is the MJCF twin the robot
authored as its CAD-of-the-day), and returns them as a bench_verify.BenchState. This
is the with-CAD condition: material-independent, near-zero pose error.
"""
from __future__ import annotations

import numpy as np

from bench_verify.scene_graph import Item, BenchState
from wetrobo.items import TASK_ITEMS


class CADObserver:
    name = "cad"

    def __init__(self, items=None):
        self.items = items or TASK_ITEMS

    def observe(self, env) -> BenchState:
        out = []
        for spec in self.items:
            pos, quat = env.get_object_pose(spec.body)  # true pose from the model
            R = np.zeros(9)
            import mujoco
            mujoco.mju_quat2Mat(R, quat)
            out.append(Item(spec.body, spec.label, spec.kind, spec.container,
                            pos.copy(), R.reshape(3, 3), confidence=1.0))
        return BenchState("cad_obs", out, captured_by=self.name)
