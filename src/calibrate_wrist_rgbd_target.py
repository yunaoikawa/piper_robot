#!/usr/bin/env python3
"""Calibrate a wrist RGB-D target tracker from stopped capture bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.wrist_rgbd_target import calibrate_from_config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scene-model",
        help="override the configured positioned scene model",
    )
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    result = calibrate_from_config(
        config,
        args.output_dir,
        scene_model=args.scene_model,
    )
    print(json.dumps(result["report"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
