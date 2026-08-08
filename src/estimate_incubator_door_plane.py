#!/usr/bin/env python3
"""Estimate the current incubator front normal from a head RGB-D bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.incubator_door_plane import estimate_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path("src/configs/pasteur_incubator_door_demo.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    report = estimate_bundle(
        args.capture,
        profile["door_plane"]["tag_config"],
        profile["door_plane"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
