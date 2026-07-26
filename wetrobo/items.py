"""Bench item specifications for the MuJoCo lab scene.

Maps MuJoCo bodies to bench_verify identity + the info WetRobo needs: whether the
object is transparent (drives the depth-sensor model), its grasp point in the body
frame, and its labware container id. Kept data-driven so tasks/observers share one
source of truth (no per-module hardcoding).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ItemSpec:
    body: str
    label: str
    kind: str                 # "Reagent" | "Labware"
    container: str
    transparent: bool = False
    grasp_local: tuple = (0.0, 0.0, 0.0)   # grasp point in body frame [m]
    graspable: bool = False
    geom_prefixes: tuple = field(default_factory=tuple)  # seg-mask hint for vision


# The lab-scene.xml movable/anchor items. The flask is the transparent target the
# whole CAD-vs-vision comparison is built around; the incubator is a stable anchor.
FLASK = ItemSpec(
    body="flask", label="flask", kind="Labware", container="erlenmeyer_flask",
    transparent=True, grasp_local=(0.0, 0.0, 0.105), graspable=True,
    geom_prefixes=("flask_",),
)
INCUBATOR = ItemSpec(
    body="fridge", label="incubator", kind="Labware", container="incubator",
    transparent=False, graspable=False, geom_prefixes=("fridge_",),
)

TASK_ITEMS = [FLASK, INCUBATOR]
