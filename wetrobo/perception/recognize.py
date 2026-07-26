"""Confidence-gated identity resolution for day-0 CAD authoring.

A perception front-end (Grounded-SAM-2 masks -> CLIP/DINO embedding match, a VLM, or an
ArUco reader) produces one `Detection` per physical object: a pose plus *ranked
candidate identities from the catalog* (closed set). This module turns those into a
confirmed `BenchState` through a three-stage gate:

    1. tag       -> if a marker id was read, identity is unambiguous (accept).
    2. confident -> top candidate clears an absolute threshold AND beats the runner-up
                    by a margin (unambiguous enough) -> accept automatically.
    3. otherwise -> hand the ranked candidates + image crop to `confirm_fn` (a human at
                    authoring time). The sim never silently commits a shaky guess.

Every decision is logged so the authored CAD is auditable ("why is this a petri dish?").
The recognizer is model-agnostic: it consumes candidate *scores*, so any matcher plugs
in. It does NOT invent scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from wetrobo.perception.catalog import LabwareCatalog


@dataclass
class Detection:
    det_id: str
    t: np.ndarray                                   # position [m], robot-base frame
    R: np.ndarray                                   # rotation (3x3)
    candidates: list[tuple[str, float]]             # ranked [(container, score)]
    tag_id: int | None = None                       # manifest AprilTag id, if read
    crop_path: str | None = None                    # image crop for the human to inspect


@dataclass
class Resolved:
    det_id: str
    container: str | None                           # None = rejected / unknown
    confidence: float
    source: str                                     # "tag" | "auto" | "confirmed" | "rejected"
    candidates: list[tuple[str, float]] = field(default_factory=list)


def interactive_confirm(det: Detection, ranked: list[tuple[str, float]],
                        catalog: LabwareCatalog) -> str | None:
    """Default confirm_fn: show the ranked candidates + crop and read a choice from the
    operator. Returns a container id, or None to reject/skip. Non-interactive callers
    should pass their own confirm_fn (e.g. a GUI, or `scripted_confirm`)."""
    print(f"\n[confirm] detection {det.det_id}  pos={np.round(det.t, 3)}"
          + (f"  crop={det.crop_path}" if det.crop_path else ""))
    for i, (c, s) in enumerate(ranked[:5]):
        print(f"   {i}: {c:32s} ({catalog.get(c).label})   score={s:.2f}")
    print("   r: reject / unknown")
    ans = input("   choose index (or 'r'): ").strip()
    if ans.lower() == "r" or ans == "":
        return None
    try:
        return ranked[int(ans)][0]
    except (ValueError, IndexError):
        return None


def scripted_confirm(answers: dict[str, str | None]):
    """A confirm_fn for tests/replays: det_id -> chosen container (or None)."""
    def _fn(det: Detection, ranked, catalog):
        return answers.get(det.det_id)
    return _fn


def resolve_identities(
    detections: list[Detection],
    catalog: LabwareCatalog,
    confirm_fn=interactive_confirm,
    accept_thresh: float = 0.75,
    margin: float = 0.12,
    log_path: str | Path | None = None,
):
    """Returns (authored BenchState of confirmed items, list[Resolved] decisions)."""
    from bench_verify.scene_graph import BenchState
    resolved: list[Resolved] = []
    items = []
    decisions = []
    for det in detections:
        ranked = sorted(det.candidates, key=lambda kv: kv[1], reverse=True)
        tag_entry = catalog.by_tag(det.tag_id)
        if tag_entry is not None:                          # (1) tag: unambiguous
            r = Resolved(det.det_id, tag_entry.container, 1.0, "tag", ranked)
        elif ranked and ranked[0][1] >= accept_thresh and \
                (len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= margin):
            r = Resolved(det.det_id, ranked[0][0], ranked[0][1], "auto", ranked)  # (2)
        else:                                              # (3) gated confirmation
            chosen = confirm_fn(det, ranked, catalog)
            if chosen is None:
                r = Resolved(det.det_id, None, 0.0, "rejected", ranked)
            else:
                r = Resolved(det.det_id, chosen, 1.0, "confirmed", ranked)
        resolved.append(r)
        if r.container is not None:
            items.append(catalog.to_item(r.container, det.det_id, det.t, det.R, r.confidence))
        decisions.append({"det_id": det.det_id, "source": r.source,
                          "chosen": r.container, "confidence": r.confidence,
                          "tag_id": det.tag_id, "candidates": ranked})
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            for d in decisions:
                f.write(json.dumps(d, default=float) + "\n")
    state = BenchState("authored_day0", items, captured_by="recognize.resolve_identities")
    return state, resolved
