"""validate.py  --  Phase 0: does CAD/geometric ground truth actually help?

This isolates the hypothesis *before* any GPU perception is built. It does not
need a robot, FoundationPose, or even MuJoCo to run; it answers one question:

  Given the same SceneVerifier and the same tolerances, does a CAD/model-based
  pose estimate (low, material-independent error) verify reset/success more
  reliably than a vision heuristic that degrades on transparent labware?

How it works
------------
1. A canonical bench is authored manually (synthetic here, or load your own
   hand-made MJCF with --from-mjcf bench_canonical.xml).
2. Each trial samples a ground-truth scenario:
     ok            (reset succeeded; truly a success)
     frame_offset  (whole bench shifted; harmless -> truly a success)
     moved         (one object displaced 5 cm; truly a FAILURE)
     missing       (one object absent;          truly a FAILURE)
3. Two estimators perceive that scene:
     vision : small error on opaque items, large error + dropout on transparent
              items  (models "SigLIP struggles with transparent flasks")
     cad    : small, material-independent error (FoundationPose with a mesh)
   Each perceived scene is judged by the SAME SceneVerifier.
4. We tally, per estimator and per transparent-fraction:
     false_success : predicted OK but truly a failure  (-> corrupted episode)
     false_reset   : predicted FAIL but truly OK        (-> wasted robot time)
     accuracy.

The two estimator callables are the only stubs; in Phase 2 replace them with
real vision and real FoundationPose to regenerate this exact table on hardware.

Run:  python -m bench_verify.validate            (synthetic canonical)
      python -m bench_verify.validate --from-mjcf bench_canonical.xml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .scene_graph import Item, BenchState, from_mjcf
from .verify import SceneVerifier


# --------------------------------------------------------------------------- #
# Canonical bench (manual "CAD"): two flasks are transparency-capable.
# --------------------------------------------------------------------------- #
def synthetic_canonical() -> BenchState:
    spec = [  # label, kind, container, position [m]
        ("flask A",  "Labware", "bottle_500ml", [0.30, -0.12, 0.09]),
        ("flask B",  "Labware", "bottle_500ml", [0.30,  0.12, 0.09]),
        ("LB broth", "Reagent", "falcon_50ml",  [0.18, -0.05, 0.06]),
        ("tip rack", "Labware", "tip_rack_p200", [0.16, 0.22, 0.03]),
        ("plate",    "Labware", "plate_96well", [0.05, 0.05, 0.01]),
        ("ethanol",  "Reagent", "bottle_500ml", [-0.18, 0.15, 0.09]),
    ]
    items = [Item(f"i{i}", lab, kind, cont, np.array(p, float), np.eye(3), 1.0)
             for i, (lab, kind, cont, p) in enumerate(spec)]
    return BenchState("canonical", items, captured_by="manual")


# Labels that can be transparent (glass/clear plastic) when sampled.
TRANSPARENT_CANDIDATES = {"flask A", "flask B", "ethanol"}


# --------------------------------------------------------------------------- #
# Scenario sampler -> (true_state, true_success, tag)
# --------------------------------------------------------------------------- #
def sample_scenario(canon: BenchState, rng) -> tuple[BenchState, bool, str]:
    base = [Item(it.item_id, it.label, it.kind, it.container,
                 it.t + rng.normal(0, 0.002, 3), it.R, 1.0)  # 2 mm placement noise
            for it in canon.items]
    u = rng.random()
    if u < 0.40:
        return BenchState("ok", base), True, "ok"
    if u < 0.60:  # harmless global frame offset
        R = Rot.from_euler("z", rng.uniform(4, 10), degrees=True).as_matrix()
        t = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), 0.0])
        items = [Item(it.item_id, it.label, it.kind, it.container,
                      R @ it.t + t, R @ it.R, 1.0) for it in base]
        return BenchState("frame", items), True, "frame_offset"
    if u < 0.80:  # one object genuinely moved 5 cm
        k = int(rng.integers(len(base)))
        d = rng.normal(0, 1, 3); d = 0.05 * d / max(np.linalg.norm(d), 1e-9)
        base[k] = Item(base[k].item_id, base[k].label, base[k].kind,
                       base[k].container, base[k].t + d, base[k].R, 1.0)
        return BenchState("moved", base), False, "moved"
    k = int(rng.integers(len(base)))  # one object missing
    del base[k]
    return BenchState("missing", base), False, "missing"


# --------------------------------------------------------------------------- #
# Estimators: true scene -> perceived scene. Only these are stubs.
# --------------------------------------------------------------------------- #
def _perceive(state, transp, rng, sig_op, sig_tr, sig_rot_deg, p_drop_tr):
    items = []
    for it in state.items:
        is_tr = transp.get(it.label, False)
        if is_tr and rng.random() < p_drop_tr:
            continue  # transparent dropout -> shows up as missing
        s = sig_tr if is_tr else sig_op
        t = it.t + rng.normal(0, s, 3)
        dR = Rot.from_rotvec(np.deg2rad(rng.normal(0, sig_rot_deg, 3))).as_matrix()
        items.append(Item(it.item_id, it.label, it.kind, it.container,
                          t, dR @ it.R, 1.0))
    return BenchState(state.bench_id + "_obs", items)


def vision_estimator(state, transp, rng):
    # opaque ~8 mm; transparent ~60 mm + 15% dropout
    return _perceive(state, transp, rng, 0.008, 0.060, 3.0, p_drop_tr=0.15)


def cad_estimator(state, transp, rng):
    # FoundationPose-with-mesh proxy: ~4 mm, material-independent, no dropout
    return _perceive(state, transp, rng, 0.004, 0.005, 1.0, p_drop_tr=0.0)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run(canon: BenchState, n_trials: int = 4000, seed: int = 0,
        transp_fracs=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    verifier = SceneVerifier(canon, pos_tol_m=0.02, rot_tol_deg=20,
                             frame_offset="absorb",
                             frame_tol_m=0.30, frame_tol_deg=25)
    estimators = {"vision": vision_estimator, "cad": cad_estimator}
    rows = []
    for frac in transp_fracs:
        rng = np.random.default_rng(seed)
        tally = {n: dict(false_success=0, false_reset=0, correct=0, n=0)
                 for n in estimators}
        for _ in range(n_trials):
            true_state, true_ok, _ = sample_scenario(canon, rng)
            transp = {it.label: (it.label in TRANSPARENT_CANDIDATES
                                 and rng.random() < frac)
                      for it in canon.items}
            for name, est in estimators.items():
                obs = est(true_state, transp, rng)
                pred_ok = verifier.verify(obs).success
                t = tally[name]
                t["n"] += 1
                t["correct"] += int(pred_ok == true_ok)
                t["false_success"] += int(pred_ok and not true_ok)
                t["false_reset"] += int((not pred_ok) and true_ok)
        for name, t in tally.items():
            rows.append(dict(
                transp_frac=frac, estimator=name,
                false_success_pct=100 * t["false_success"] / t["n"],
                false_reset_pct=100 * t["false_reset"] / t["n"],
                accuracy_pct=100 * t["correct"] / t["n"]))
    return {"canonical_items": [it.label for it in canon.items],
            "n_trials": n_trials, "rows": rows}


def print_table(result: dict) -> None:
    print(f"\nTable 1  CAD-grounded vs vision verify  (n={result['n_trials']}/cell)")
    print("transparent items can be: " + ", ".join(sorted(TRANSPARENT_CANDIDATES)))
    print("-" * 72)
    print(f"{'transp%':>8} {'estimator':>10} "
          f"{'false_success%':>15} {'false_reset%':>13} {'accuracy%':>10}")
    print("-" * 72)
    for r in result["rows"]:
        print(f"{r['transp_frac']*100:>7.0f}% {r['estimator']:>10} "
              f"{r['false_success_pct']:>15.2f} {r['false_reset_pct']:>13.2f} "
              f"{r['accuracy_pct']:>10.2f}")
    print("-" * 72)
    print("false_success = corrupted episode kept (worst);  "
          "false_reset = wasted robot time.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-mjcf", default=None,
                    help="canonical bench MJCF (else synthetic)")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="validate_cad_vs_vision.json")
    args = ap.parse_args()

    canon = from_mjcf(args.from_mjcf) if args.from_mjcf else synthetic_canonical()
    result = run(canon, n_trials=args.n, seed=args.seed)
    print_table(result)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
