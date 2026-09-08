"""bench_scene_graph.py

Minimal, transfer-aware bench scene graph for wet-lab robotics.

Spine of the system (model-agnostic, runnable as-is):
  1. Data model: Item + BenchState (identity + metric SE(3) pose).
  2. CAD export: assemble each item as real geometry at its 6-DoF pose into a
     single OBJ(+MTL) scene. Mesh source is CAD library / image-to-3D when
     available, else a parametric primitive keyed by labware ontology.
  3. Transfer diff: single SE(3) Kabsch fit over matched objects -> frame
     offset (auto-correctable) + per-object residual (real transfer risk)
     + identity set-diff (unmappable).
  4. Comparison harness: pluggable Tier-A/B/C perception backends and
     ADD / ADD-S pose metrics.

Heavy models (FoundationPose, Hunyuan3D/TRELLIS/SAM-3D, Gemini Robotics-ER)
attach only at the marked plug-in points; nothing here depends on them.
Swapping OBJ -> MJCF/USD is a writer change: primitives map to native
MuJoCo geoms / USD prims.

Deps: numpy, scipy. No network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Item:
    """A single bench item with a metric 6-DoF pose in the robot base frame."""
    item_id: str
    label: str                       # open-vocab name, e.g. "LB broth"
    kind: str                        # "Reagent" | "Labware"
    container: str                   # labware ontology id, e.g. "falcon_50ml"
    t: np.ndarray                    # (3,) translation [m]
    R: np.ndarray                    # (3, 3) rotation, base<-object
    confidence: float = 1.0

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T


@dataclass
class BenchState:
    """A versioned snapshot of one bench at one time, in one reference frame."""
    bench_id: str
    items: list[Item]
    frame: str = "robot_base"
    captured_by: str = "unknown"
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def by_label(self) -> dict[str, Item]:
        return {it.label: it for it in self.items}


# --------------------------------------------------------------------------- #
# CAD export: per-item geometry placed at pose -> OBJ (+ MTL)
# --------------------------------------------------------------------------- #
# Parametric proxies (meters). Real CAD/image-to-3D meshes override via
# MESH_LIBRARY. Plate/rack use the SBS footprint (127.76 x 85.48 mm).
PRIMITIVES: dict[str, tuple[str, tuple]] = {
    "falcon_50ml":   ("cyl", (0.0145, 0.1145)),
    "falcon_15ml":   ("cyl", (0.0085, 0.1200)),
    "bottle_500ml":  ("cyl", (0.0380, 0.1800)),
    "vial_2ml":      ("cyl", (0.0060, 0.0430)),
    "tip_rack_p200": ("box", (0.12776, 0.08548, 0.0600)),
    "plate_96well":  ("box", (0.12776, 0.08548, 0.0143)),
}
DEFAULT_PRIM = ("box", (0.03, 0.03, 0.03))

# Plug: load STEP/STL (CAD library) or GLB (image-to-3D) -> (V[n,3], F[m,3]).
MESH_LIBRARY: dict[str, tuple[np.ndarray, np.ndarray]] = {}

MATERIALS = {  # (r, g, b) in 0..1
    "Reagent": (0.85, 0.55, 0.15),
    "Labware": (0.55, 0.60, 0.70),
}


def _box(dx: float, dy: float, dz: float) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = dx / 2, dy / 2, dz / 2
    V = np.array([[sx * x, sy * y, sz * z]
                  for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return V, F


def _cyl(r: float, h: float, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.c_[r * np.cos(a), r * np.sin(a)]
    bot = np.c_[ring, np.full(n, -h / 2)]
    top = np.c_[ring, np.full(n, h / 2)]
    V = np.vstack([bot, top, [0, 0, -h / 2], [0, 0, h / 2]])
    cb, ct = 2 * n, 2 * n + 1
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]      # side
        F += [[cb, j, i], [ct, n + i, n + j]]        # caps
    return V, np.array(F)


def _geom_for(it: Item) -> tuple[np.ndarray, np.ndarray]:
    if it.container in MESH_LIBRARY:
        return MESH_LIBRARY[it.container]
    shape, dims = PRIMITIVES.get(it.container, DEFAULT_PRIM)
    return (_cyl(*dims) if shape == "cyl" else _box(*dims))


def to_cad(state: BenchState, obj_path: str | Path) -> Path:
    """Write the bench as an assembled CAD scene (OBJ + sibling MTL).
    Each item is one named object placed at its pose; returns the OBJ path."""
    obj_path = Path(obj_path)
    mtl_path = obj_path.with_suffix(".mtl")

    mtl = ["# bench scene materials"]
    for name, (r, g, b) in MATERIALS.items():
        mtl += [f"newmtl {name}", f"Kd {r:.3f} {g:.3f} {b:.3f}", ""]
    mtl_path.write_text("\n".join(mtl))

    out = [f"# bench: {state.bench_id}  frame: {state.frame}",
           f"# generated: {state.captured_at}  by: {state.captured_by}",
           f"mtllib {mtl_path.name}", ""]
    voff = 0
    for it in state.items:
        V, F = _geom_for(it)
        Vw = (it.R @ V.T).T + it.t                   # local -> world (pose)
        name = f"{it.label}_{it.item_id}".replace(" ", "_")
        mat = it.kind if it.kind in MATERIALS else "Labware"
        out.append(f"o {name}")
        out.append(f"usemtl {mat}")
        out += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in Vw]
        out += [f"f {a+voff+1} {b+voff+1} {c+voff+1}" for a, b, c in F]
        out.append("")
        voff += len(Vw)
    obj_path.write_text("\n".join(out))
    return obj_path


# --------------------------------------------------------------------------- #
# Transfer diff: SE(3) Kabsch fit + residuals + identity set-diff
# --------------------------------------------------------------------------- #
@dataclass
class TransferDiff:
    R_ba: np.ndarray                 # rigid frame correction A -> B
    t_ba: np.ndarray
    frame_offset_deg: float          # global rotation magnitude
    frame_offset_m: float            # global translation magnitude
    residual_m: dict[str, float]     # per-object position residual after fit
    rot_resid_deg: dict[str, float]  # per-object orientation residual
    moved: list[str]                 # residual above threshold -> transfer risk
    only_in_a: list[str]             # present at train, missing at eval
    only_in_b: list[str]             # extra at eval


def _kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best rigid R, t mapping A -> B (no scale). A, B: (n, 3)."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cB - R @ cA


def transfer_diff(
    train: BenchState, eval_: BenchState, move_thresh_m: float = 0.02
) -> TransferDiff:
    """Decompose a train->eval scene change into (1) a global frame offset
    that re-grounding can absorb, and (2) per-object residuals + identity
    differences that are the genuine transfer risks."""
    a, b = train.by_label(), eval_.by_label()
    shared = sorted(set(a) & set(b))
    A = np.array([a[k].t for k in shared])
    B = np.array([b[k].t for k in shared])

    R, t = _kabsch(A, B) if len(shared) >= 3 else (np.eye(3), np.zeros(3))
    resid = {k: float(np.linalg.norm(R @ a[k].t + t - b[k].t)) for k in shared}
    rot_resid = {
        k: float(Rot.from_matrix(b[k].R @ (R @ a[k].R).T).magnitude() * 180 / np.pi)
        for k in shared
    }
    return TransferDiff(
        R_ba=R, t_ba=t,
        frame_offset_deg=float(Rot.from_matrix(R).magnitude() * 180 / np.pi),
        frame_offset_m=float(np.linalg.norm(t)),
        residual_m=resid, rot_resid_deg=rot_resid,
        moved=[k for k, v in resid.items() if v > move_thresh_m],
        only_in_a=sorted(set(a) - set(b)),
        only_in_b=sorted(set(b) - set(a)),
    )


def reground(train: BenchState, diff: TransferDiff) -> BenchState:
    """Apply the recovered frame offset to remap a training-frame state into
    the eval frame (auto-correction of the harmless component)."""
    out = [Item(it.item_id, it.label, it.kind, it.container,
                diff.R_ba @ it.t + diff.t_ba, diff.R_ba @ it.R, it.confidence)
           for it in train.items]
    return BenchState(train.bench_id + "_regrounded", out, train.frame)


# --------------------------------------------------------------------------- #
# Comparison harness: pluggable Tier-A/B/C backends + pose metrics
# --------------------------------------------------------------------------- #
class PerceptionBackend:
    """Override estimate(). Returns a BenchState from sensor inputs."""
    name = "base"

    def estimate(self, rgb, depth=None, intrinsics=None, **kw) -> BenchState:
        raise NotImplementedError


class TierA_VLMOnly(PerceptionBackend):
    """Cheapest: VLM emits labels + coarse region centers, no orientation.
    PLUG: Gemini Robotics-ER -> parse JSON -> Item(R=I)."""
    name = "A_vlm_only"


class TierB_DetectDepth(PerceptionBackend):
    """Mid: open-vocab seg + metric depth -> 3D centroid, no orientation.
    PLUG: Grounded-SAM-2 masks x Depth-Anything/RGBD -> Item(R=I)."""
    name = "B_detect_depth"


class TierC_FullPose(PerceptionBackend):
    """Most accurate: CAD-or-image-to-3D mesh -> FoundationPose -> 6-DoF.
    PLUG: Grounded-SAM-2 -> {CAD lookup | Hunyuan3D/TRELLIS} -> FoundationPose."""
    name = "C_full_pose"


def add_metric(pred: BenchState, gt: BenchState, symmetric: bool = False) -> dict:
    """ADD / ADD-S over matched labels, using each item's actual proxy/CAD
    vertices placed at the two poses."""
    p, g = pred.by_label(), gt.by_label()
    errs = []
    for k in set(p) & set(g):
        V, _ = _geom_for(g[k])
        Pp, Pg = p[k].matrix(), g[k].matrix()
        vp = (Pp[:3, :3] @ V.T).T + Pp[:3, 3]
        vg = (Pg[:3, :3] @ V.T).T + Pg[:3, 3]
        if symmetric:
            d = np.linalg.norm(vp[:, None] - vg[None], axis=-1).min(1).mean()
        else:
            d = np.linalg.norm(vp - vg, axis=-1).mean()
        errs.append(d)
    key = "ADD-S" if symmetric else "ADD"
    return {key: float(np.mean(errs)) if errs else float("nan"),
            "matched": len(errs)}


# --------------------------------------------------------------------------- #
# Demo: two synthetic benches (lab A train vs lab B eval) -> CAD scene
# --------------------------------------------------------------------------- #
def _demo() -> None:
    rng = np.random.default_rng(0)

    def make(frame_R, frame_t, jitter):
        base = {"LB broth": ("Reagent", "falcon_50ml", [.32, -.11, .04]),
                "tip rack": ("Labware", "tip_rack_p200", [.18, .22, .03]),
                "plate":    ("Labware", "plate_96well", [.05, .05, .01]),
                "ethanol":  ("Reagent", "bottle_500ml", [-.20, .15, .09])}
        items = []
        for i, (lab, (kind, cont, pos)) in enumerate(base.items()):
            t = frame_R @ np.array(pos) + frame_t + jitter.get(lab, 0.0)
            R = frame_R @ Rot.from_euler("z", rng.uniform(-.1, .1)).as_matrix()
            items.append(Item(f"i{i}", lab, kind, cont, t, R, 0.95))
        return items

    train = BenchState("bench_A_train", make(np.eye(3), np.zeros(3), {}),
                       captured_by="gemini-robotics-er")
    fR = Rot.from_euler("z", 8, degrees=True).as_matrix()
    eval_ = BenchState(
        "bench_B_eval",
        make(fR, np.array([.12, -.03, 0]), {"ethanol": np.array([.06, 0, 0])}),
        captured_by="gemini-robotics-er")
    eval_.items.append(Item("i9", "BSA powder", "Reagent", "vial_2ml",
                            np.array([.0, -.1, .03]), np.eye(3), 0.9))

    obj = to_cad(eval_, "/home/claude/bench_B_eval.obj")
    n_v = sum(len(_geom_for(it)[0]) for it in eval_.items)
    n_f = sum(len(_geom_for(it)[1]) for it in eval_.items)
    print(f"=== CAD export ===\n{obj}  +  {obj.with_suffix('.mtl').name}")
    print(f"items: {len(eval_.items)}  vertices: {n_v}  faces: {n_f}\n")

    d = transfer_diff(train, eval_)
    print("=== Transfer diff ===")
    print(f"frame offset   : {d.frame_offset_deg:5.2f} deg, "
          f"{d.frame_offset_m*100:4.1f} cm  (auto-correctable)")
    print(f"moved objects  : {d.moved}  (real transfer risk)")
    print(f"only at eval   : {d.only_in_b}  (unmappable)")
    print("residuals [cm] :", {k: round(v*100, 1) for k, v in d.residual_m.items()})
    print(f"\nADD after re-grounding: {add_metric(reground(train, d), eval_)}")


if __name__ == "__main__":
    _demo()


# --------------------------------------------------------------------------- #
# MJCF I/O: the canonical bench *is* a MuJoCo scene.
#   - to_mjcf  : BenchState -> MJCF (each item = a <body> + geom at its pose).
#   - from_mjcf: MJCF -> BenchState (load a hand-authored scene, or your real
#                lab MuJoCo, as the ground-truth scene graph).
# Identity is carried in the body name ("item__id__label__kind__container").
# For a pre-existing scene whose bodies do not follow that convention, pass a
# name_map: {body_name: (item_id, label, kind, container)}.
# Only direct children of <worldbody> are read (canonical scenes are flat).
# --------------------------------------------------------------------------- #
import xml.etree.ElementTree as ET

# container -> mesh file (STL/OBJ) for real CAD geoms; empty => use primitives.
MESH_FILES: dict[str, str] = {}
_SP = "__"


def _enc_name(it: "Item") -> str:
    return _SP.join(["item", it.item_id, it.label.replace(" ", "-"),
                     it.kind, it.container])


def _dec_name(name: str) -> tuple[str, str, str, str]:
    _, item_id, label, kind, container = name.split(_SP)[:5]
    return item_id, label.replace("-", " "), kind, container


def to_mjcf(state: "BenchState", path: str | Path) -> Path:
    mj = ET.Element("mujoco", model=state.bench_id)
    asset = ET.SubElement(mj, "asset")
    wb = ET.SubElement(mj, "worldbody")
    seen: set[str] = set()
    for it in state.items:
        q = Rot.from_matrix(it.R).as_quat(scalar_first=True)  # wxyz
        body = ET.SubElement(
            wb, "body", name=_enc_name(it),
            pos=f"{it.t[0]:.6f} {it.t[1]:.6f} {it.t[2]:.6f}",
            quat=f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        r, g, b = MATERIALS.get(it.kind, MATERIALS["Labware"])
        rgba = f"{r:.3f} {g:.3f} {b:.3f} 1"
        if it.container in MESH_FILES:
            mname = f"mesh_{it.container}"
            if it.container not in seen:
                ET.SubElement(asset, "mesh", name=mname,
                              file=MESH_FILES[it.container])
                seen.add(it.container)
            ET.SubElement(body, "geom", type="mesh", mesh=mname, rgba=rgba)
        else:
            shape, dims = PRIMITIVES.get(it.container, DEFAULT_PRIM)
            if shape == "cyl":
                rad, h = dims
                ET.SubElement(body, "geom", type="cylinder",
                              size=f"{rad:.6f} {h/2:.6f}", rgba=rgba)
            else:
                dx, dy, dz = dims
                ET.SubElement(body, "geom", type="box",
                              size=f"{dx/2:.6f} {dy/2:.6f} {dz/2:.6f}", rgba=rgba)
    tree = ET.ElementTree(mj)
    ET.indent(tree, space="  ")
    Path(path).write_bytes(ET.tostring(mj, encoding="utf-8"))
    return Path(path)


def from_mjcf(path: str | Path, name_map: dict | None = None,
              frame: str = "robot_base", bench_id: str | None = None) -> "BenchState":
    root = ET.parse(str(path)).getroot()
    wb = root.find("worldbody")
    items: list[Item] = []
    for body in wb.findall("body"):
        name = body.get("name", "")
        if name_map and name in name_map:
            item_id, label, kind, container = name_map[name]
        elif name.startswith("item" + _SP):
            item_id, label, kind, container = _dec_name(name)
        else:
            continue
        t = np.array([float(x) for x in body.get("pos", "0 0 0").split()])
        qw = [float(x) for x in body.get("quat", "1 0 0 0").split()]
        R = Rot.from_quat(qw, scalar_first=True).as_matrix()
        items.append(Item(item_id, label, kind, container, t, R, 1.0))
    return BenchState(bench_id or Path(path).stem, items,
                      frame=frame, captured_by="mjcf")
