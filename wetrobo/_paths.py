"""Path resolution + make the existing `bench_verify` package importable.

`bench_verify` lives at robot/piper-mujoco/bench_verify and is normally run as
`python -m bench_verify.*` from that directory. We add that directory to sys.path so
WetRobo can reuse it (scene_graph, verify, mujoco_oracle) without duplicating code.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPER_MUJOCO = REPO_ROOT / "robot" / "piper-mujoco"
LAB_SCENE_XML = PIPER_MUJOCO / "xml" / "lab-scene.xml"

# bench_verify uses relative imports; expose its parent dir as an import root.
_bench_parent = str(PIPER_MUJOCO)
if _bench_parent not in sys.path:
    sys.path.insert(0, _bench_parent)
