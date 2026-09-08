"""Structured JSONL logging — every attempt WetRobo makes is recorded as real data.

One JSON object per line per attempt (condition, day, seed, perceived pose, grasp
distance, outcome, final flask pose). The experiment's figures are built only from
these logs (Principle 1: no fabricated numbers).
"""
from __future__ import annotations

import json
from pathlib import Path


class EpisodeLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def add(self, **record) -> None:
        self.records.append(record)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    @staticmethod
    def read(path: str | Path) -> list[dict]:
        p = Path(path)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _json_default(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
