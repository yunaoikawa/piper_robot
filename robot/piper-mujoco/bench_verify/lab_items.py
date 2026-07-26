"""lab_items.py  --  bridge the real lab MuJoCo to bench_verify.

The lab scene bodies are not item__-named and use euler / a freejoint whose
pose lives in the lab_home keyframe, so use ground_truth_state (MuJoCo reads
world poses after the keyframe reset), not the flat XML from_mjcf.

Stable anchors (incubator, microscope) improve the Kabsch fit; the flask is
the transparent item the whole verifier is built around.

Run on pasteur (mujoco installed):
    python -m bench_verify.lab_items
"""
from .mujoco_oracle import ground_truth_state

# body name -> (item_id, label, kind, container)
LAB_ITEMS = {
    "dish":          ("dish",  "culture dish",  "Labware", "culture_dish_60mm"),
    "fridge":        ("incub", "incubator",     "Labware", "incubator"),
    "scope":         ("scope", "microscope",    "Labware", "microscope_tms"),
    "medium_bottle": ("media", "medium bottle", "Reagent", "media_bottle_250ml"),
}


def lab_canonical(path: str = "xml/lab-scene.xml"):
    return ground_truth_state(path, name_map=LAB_ITEMS, keyframe="lab_home")


if __name__ == "__main__":
    s = lab_canonical()
    print(f"canonical from {s.bench_id}: {len(s.items)} items")
    for it in s.items:
        print(f"  {it.label:12s} t={it.t.round(3)}")
