#!/usr/bin/env python3
"""Audit immutable cap-transfer captures without camera or robot I/O."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollout.cylindrical_cap_transfer import audit_cap_transfer_captures


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-profile", required=True)
    parser.add_argument("--before", required=True)
    parser.add_argument("--lift", required=True)
    parser.add_argument("--transported", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    task = json.loads(Path(args.task_profile).read_text())
    identity = json.loads(Path(task["tap_identity"]).read_text())
    result = audit_cap_transfer_captures(
        args.before,
        args.lift,
        args.transported,
        target_anchor_uv=identity["tap"]["uv"],
        settings=task["verified_transfer"],
    )
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(encoded)
    print(encoded, end="")
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
