#!/usr/bin/env python3
"""Extract only numeric evidence and source hashes; never connect to hardware."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    stems = [
        "20260806T090454.207716Z_head_culture_media_cap_hold_before_lift_ce98d39c",
        "20260806T090516.450533Z_head_culture_media_cap_lift_probe10_792a8b3c",
        "20260806T090638.703002Z_head_culture_media_cap_transport_home_hold_c01dae62",
    ]
    frames = []
    for stem in stems:
        path = ROOT / "data/captures/pasteur/2026-08-06" / stem / "manifest.json"
        raw = path.read_bytes()
        source = json.loads(raw)
        state = source["robot_state"]["before"]
        frames.append({"source": str(path.relative_to(ROOT)),
                       "sha256": hashlib.sha256(raw).hexdigest(),
                       "created_at_utc": source["created_at_utc"],
                       "ee_xyz_m": state["right_ee_pose"]["translation_xyz_m"],
                       "opening": state["right_gripper_open_ratio"]})
    xyz = np.array([f["ee_xyz_m"] for f in frames])
    snapshots = {}
    targets = {
        "f2149bb": ["src/run_incubator_door_demo.py", "src/run_incubator_door_autonomy.py", "rollout/teleop_trajectory_stream.py"],
        "fc831c0": ["rollout/media_cap_target.py", "src/run_culture_media_cap_grasp.py"],
        "1f07761": ["rollout/cylindrical_cap_transfer.py"],
    }
    for commit, paths in targets.items():
        for path in paths:
            raw = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
            snapshots[f"{commit}:{path}"] = hashlib.sha256(raw).hexdigest()
    report = {
        "cap": {"frames": frames, "lift_mm": float((xyz[1, 2]-xyz[0, 2])*1000),
                "transport_after_lift_mm": float(np.linalg.norm(xyz[2]-xyz[1])*1000),
                "aperture_drift_after_lift": abs(frames[2]["opening"]-frames[1]["opening"])},
        "historical_source_sha256": snapshots,
        "scope": "Numeric extraction from archived files; no runtime code imported or executed",
    }
    (HERE / "evidence.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["cap"], indent=2))


if __name__ == "__main__":
    main()
