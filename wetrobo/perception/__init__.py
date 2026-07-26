"""Perception backends: turn the sim into an observed scene graph (BenchState).

Two backends realize the paper's independent variable:

* CADObserver   — reads exact object poses from the MJCF/sim (the "daily CAD").
* VisionObserver — back-projects the camera's rendered depth into 3D and estimates
  poses, applying a documented depth-sensor model that fails on transparent glass
  (IR pass-through) — so labware like the flask is localized poorly, exactly the
  failure the repo README notes ("SigLIP struggles with transparent flasks").

Both return a bench_verify.BenchState, so downstream code is backend-agnostic.
"""
from wetrobo.perception.cad import CADObserver  # noqa: F401
from wetrobo.perception.vision import VisionObserver  # noqa: F401
