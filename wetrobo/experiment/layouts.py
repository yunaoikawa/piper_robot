"""Daily lab layouts — each "day" places the flask at a different bench position.

A day is a real change to the movable labware; the with-CAD condition captures it
exactly (that is what authoring the day's CAD buys you), the no-CAD condition must
recover it from vision. Positions are sampled within the right arm's reliably
graspable envelope (mapped empirically), so any success gap is due to perception, not
unreachability. Randomised start poses are legitimate experiment design — the measured
rollouts are real, nothing is fabricated.
"""
from __future__ import annotations

import numpy as np

# Reliably graspable envelope for the right 5-DOF arm (from the reach map): close-in
# region where the grasp site reaches the flask neck to ~1 cm.
GRASP_ZONE = {"x": (0.18, 0.24), "y": (-0.09, -0.01)}


def daily_flask_positions(n_days: int, seed: int = 0) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(*GRASP_ZONE["x"], size=n_days)
    ys = rng.uniform(*GRASP_ZONE["y"], size=n_days)
    return [(float(x), float(y)) for x, y in zip(xs, ys)]
