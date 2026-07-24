#!/usr/bin/env python3
"""Checks for successful-grip run extraction."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_manipulation_template import longest_closed_run


assert longest_closed_run([1, 0, 0, 1, 0, 0, 0, 1]) == {
    "frame": 4,
    "length": 3,
}

try:
    longest_closed_run(np.ones(5))
except ValueError:
    pass
else:
    raise AssertionError("open-only demo should not produce a grasp goal")

print("manipulation template checks passed")
