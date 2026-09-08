"""verify.py

SceneVerifier: a metric, CAD-grounded replacement for ENPIRE's EN-module
success signal. Given a canonical bench (the hand-authored MJCF) and an
observed scene graph (from perception), it returns:

  * success  : bool, whether the observed scene matches canonical
  * category : "ok" | "frame_offset" | "moved" | "missing_extra"
  * the offending items, and the reset target to drive the next episode.

Mechanism (from scene_graph.transfer_diff):
  single SE(3) Kabsch fit over matched objects
    -> global frame offset   (harmless; absorbed by re-grounding)
    -> per-object residual    (object actually moved -> verification fail)
    -> identity set-diff      (missing / extra item   -> verification fail)

Replaces the noisy camera/heuristic success signal that pure-vision verify
relies on; especially relevant where vision is unreliable (transparent /
specular labware).
"""
from __future__ import annotations

from dataclasses import dataclass

from .scene_graph import BenchState, TransferDiff, transfer_diff, reground


@dataclass
class VerifyResult:
    success: bool
    category: str            # "ok" | "frame_offset" | "moved" | "missing_extra"
    moved: list[str]
    missing: list[str]
    extra: list[str]
    frame_offset_m: float
    frame_offset_deg: float
    max_residual_m: float
    diff: TransferDiff

    def reason(self) -> str:
        if self.category == "missing_extra":
            return f"identity diff: missing={self.missing} extra={self.extra}"
        if self.category == "moved":
            return f"objects moved > tol: {self.moved} (max {self.max_residual_m*100:.1f} cm)"
        if self.category == "frame_offset":
            return (f"global frame offset {self.frame_offset_deg:.1f} deg / "
                    f"{self.frame_offset_m*100:.1f} cm")
        return "scene matches canonical"


class SceneVerifier:
    """Verify an observed bench against a fixed canonical bench.

    Parameters
    ----------
    canonical : the ground-truth scene (e.g. from_mjcf("bench_canonical.xml")).
    pos_tol_m / rot_tol_deg : per-object tolerance after the rigid fit.
    frame_offset : "absorb" treats a whole-bench shift as success (cross-lab
        transfer use-case); "fail" rejects a shift beyond frame_tol (single-
        bolted-bench reset use-case).
    frame_tol_m / frame_tol_deg : bound separating "small" from "big" offset.
    """

    def __init__(self, canonical: BenchState, pos_tol_m: float = 0.02,
                 rot_tol_deg: float = 20.0, frame_offset: str = "absorb",
                 frame_tol_m: float = 0.30, frame_tol_deg: float = 25.0):
        assert frame_offset in ("absorb", "fail")
        self.canonical = canonical
        self.pos_tol_m = pos_tol_m
        self.rot_tol_deg = rot_tol_deg
        self.frame_offset = frame_offset
        self.frame_tol_m = frame_tol_m
        self.frame_tol_deg = frame_tol_deg

    def verify(self, observed: BenchState) -> VerifyResult:
        d = transfer_diff(self.canonical, observed, move_thresh_m=self.pos_tol_m)
        max_res = max(d.residual_m.values()) if d.residual_m else 0.0
        moved = [k for k in d.residual_m
                 if d.residual_m[k] > self.pos_tol_m
                 or d.rot_resid_deg[k] > self.rot_tol_deg]
        big_frame = (d.frame_offset_m > self.frame_tol_m
                     or d.frame_offset_deg > self.frame_tol_deg)

        def res(success, category, moved_=(), missing=(), extra=()):
            return VerifyResult(success, category, list(moved_), list(missing),
                                list(extra), d.frame_offset_m, d.frame_offset_deg,
                                max_res, d)

        if d.only_in_a or d.only_in_b:
            return res(False, "missing_extra", missing=d.only_in_a, extra=d.only_in_b)
        if moved:
            return res(False, "moved", moved_=moved)
        if self.frame_offset == "fail" and big_frame:
            return res(False, "frame_offset")
        return res(True, "frame_offset" if big_frame else "ok")

    # --- reset support ----------------------------------------------------- #
    def reset_target(self) -> BenchState:
        """Canonical scene = the state the reset policy must restore."""
        return self.canonical

    def reset_target_in_observed_frame(self, observed: BenchState) -> BenchState:
        """Canonical mapped into the current world frame, so per-object goals
        are commandable directly even after a global frame offset."""
        d = transfer_diff(self.canonical, observed)
        return reground(self.canonical, d)
