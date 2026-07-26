"""Perception backends and the real SAM calibration adapter.

The two legacy simulation backends realize the original paper variable:

* CADObserver   — reads exact object poses from the MJCF/sim (the "daily CAD").
* VisionObserver — back-projects the camera's rendered depth into 3D and estimates
  poses, applying a documented depth-sensor model that fails on transparent glass
  (IR pass-through) — so labware like the flask is localized poorly, exactly the
  failure the repo README notes ("SigLIP struggles with transparent flasks").

For deployment, SamCalibrationArtifact quality-gates synchronized SAM + RGB-D and
publishes a BenchState only after an explicit camera-to-robot transform is accepted.
The nominal MJCF supplies geometry priors, never observed poses.
"""
from wetrobo.perception.cad import CADObserver  # noqa: F401
from wetrobo.perception.sam import (  # noqa: F401
    CalibrationRejected,
    SamCalibrationArtifact,
    SamLabelBinding,
    SamQualityGates,
)
from wetrobo.perception.vision import VisionObserver  # noqa: F401
