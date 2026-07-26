"""LabwareCatalog — the lab's real inventory as a closed recognition set.

Identity is *chosen from this catalog*, never open-vocab named. Each entry pins the
container ontology id to its true geometry (so the day's CAD has correct dimensions —
the same dimensions that make a grasp reachable), an optional ArUco ``tag_id`` for
unambiguous identification, and ``aliases``/``ref_images`` used only as coarse matching
cues. This is the "geometry + identity" source of truth that `to_mjcf` and the
confidence-gated recognizer (`recognize.py`) both read.

Deliberately decoupled: geometry acquisition (photos/LiDAR/GenRecon) fills in real dims
and meshes; identity comes from matching a detection against these entries. Reconstruct
geometry, but *look up* identity.
"""
from __future__ import annotations

import json
import difflib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco

import wetrobo._paths  # noqa: F401 - puts bench_verify on sys.path
from bench_verify.scene_graph import Item, BenchState

DEFAULT_CATALOG = Path(__file__).with_name("labware_catalog.json")


@dataclass(frozen=True)
class LabwareEntry:
    container: str                       # ontology id = primary key
    label: str
    kind: str                            # "Reagent" | "Labware"
    shape: str                           # "cyl" | "box" | "mesh"
    dims: tuple                          # cyl:(r,h)  box:(dx,dy,dz)  mesh:()
    mesh_file: str | None = None
    transparent: bool = False
    graspable: bool = False
    grasp_local: tuple = (0.0, 0.0, 0.0)
    mass: float | None = None
    tag_id: int | None = None            # operational AprilTag id, if physically tagged
    aliases: tuple = ()
    ref_images: tuple = ()
    provisional_dims: bool = False       # True until measured (calipers/LiDAR)

    def geom_kwargs(self) -> dict:
        """MuJoCo <geom> attributes for this entry's proxy geometry."""
        if self.shape == "cyl":
            r, h = self.dims
            return {"type": "cylinder", "size": f"{r:.6f} {h/2:.6f}"}
        if self.shape == "box":
            dx, dy, dz = self.dims
            return {"type": "box", "size": f"{dx/2:.6f} {dy/2:.6f} {dz/2:.6f}"}
        if self.shape == "mesh":
            return {"type": "mesh", "mesh": f"mesh_{self.container}"}
        raise ValueError(f"unknown shape {self.shape!r} for {self.container}")


class LabwareCatalog:
    def __init__(self, entries: list[LabwareEntry]):
        self.entries = list(entries)
        self._by_container = {e.container: e for e in self.entries}
        self._by_tag = {e.tag_id: e for e in self.entries if e.tag_id is not None}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LabwareCatalog":
        data = json.loads(Path(path or DEFAULT_CATALOG).read_text())
        entries = []
        for d in data["entries"]:
            d = {k: v for k, v in d.items() if not k.startswith("_")}
            for k in ("dims", "grasp_local", "aliases", "ref_images"):
                if k in d and isinstance(d[k], list):
                    d[k] = tuple(d[k])
            entries.append(LabwareEntry(**d))
        return cls(entries)

    # --- lookups ----------------------------------------------------------- #
    def get(self, container: str) -> LabwareEntry:
        return self._by_container[container]

    def by_tag(self, tag_id: int | None) -> LabwareEntry | None:
        return self._by_tag.get(tag_id)

    def containers(self) -> list[str]:
        return list(self._by_container)

    def alias_scores(self, name: str) -> list[tuple[str, float]]:
        """Coarse closed-set score of a free-text ``name`` against every entry, by
        fuzzy match to its label + aliases. A *cue* for ranking candidates — NOT an
        identity decision (that is what the confidence gate + confirmation are for).
        Returns [(container, score in 0..1)] ranked high-to-low."""
        q = name.strip().lower()
        out = []
        for e in self.entries:
            names = [e.label.lower(), e.container.lower(), *(a.lower() for a in e.aliases)]
            score = max(difflib.SequenceMatcher(None, q, n).ratio() for n in names)
            if any(q == n or q in n or n in q for n in names):
                score = max(score, 0.9)
            out.append((e.container, float(score)))
        return sorted(out, key=lambda kv: kv[1], reverse=True)

    # --- interop ----------------------------------------------------------- #
    def to_item(self, container: str, item_id: str, t, R, confidence: float) -> Item:
        e = self.get(container)
        return Item(item_id, e.label, e.kind, e.container,
                    np.asarray(t, float), np.asarray(R, float), float(confidence))

    def register_into_scene_graph(self) -> None:
        """Feed catalog geometry into bench_verify.scene_graph so its `to_cad`,
        `add_metric`, etc. use the catalog's real dims/meshes for these containers."""
        import bench_verify.scene_graph as sg
        for e in self.entries:
            if e.shape == "mesh" and e.mesh_file:
                sg.MESH_FILES[e.container] = e.mesh_file
            elif e.shape in ("cyl", "box"):
                sg.PRIMITIVES[e.container] = (e.shape, tuple(e.dims))

    def to_mjcf(self, state: BenchState, path: str | Path) -> Path:
        """Write a BenchState to an MJCF using catalog geometry (handles equipment and
        meshes the stock scene_graph.PRIMITIVES table doesn't know, and avoids the
        scipy `scalar_first` dependency by converting rotations via MuJoCo)."""
        import xml.etree.ElementTree as ET
        mj = ET.Element("mujoco", model=state.bench_id)
        asset = ET.SubElement(mj, "asset")
        ET.SubElement(asset, "material", name="Reagent", rgba="0.85 0.55 0.15 1")
        ET.SubElement(asset, "material", name="Labware", rgba="0.55 0.60 0.70 0.6")
        wb = ET.SubElement(mj, "worldbody")
        ET.SubElement(wb, "light", pos="0.4 0 1.4", dir="0 0 -1")
        seen: set[str] = set()
        q = np.zeros(4)
        for it in state.items:
            e = self.get(it.container)
            mujoco.mju_mat2Quat(q, np.asarray(it.R, float).reshape(9))
            body = ET.SubElement(
                wb, "body", name=f"item__{it.item_id}__{it.label.replace(' ', '-')}"
                                 f"__{it.kind}__{it.container}",
                pos=f"{it.t[0]:.6f} {it.t[1]:.6f} {it.t[2]:.6f}",
                quat=f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
            if e.shape == "mesh" and e.mesh_file and it.container not in seen:
                ET.SubElement(asset, "mesh", name=f"mesh_{it.container}", file=e.mesh_file)
                seen.add(it.container)
            g = e.geom_kwargs()
            g["material"] = e.kind if e.kind in ("Reagent", "Labware") else "Labware"
            ET.SubElement(body, "geom", **g)
        tree = ET.ElementTree(mj)
        ET.indent(tree, space="  ")
        Path(path).write_bytes(ET.tostring(mj, encoding="utf-8"))
        return Path(path)
