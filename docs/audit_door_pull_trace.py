#!/usr/bin/env python3
"""Read-only historical trace search. Never execute archived commands.

Optional private event-log audit emits only numeric slip-error measurements,
line numbers, timestamps, and hashes; no conversations or credentials exported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data/runs/pasteur"
ASSETS = ROOT / "docs/assets/code_as_learning_machine"
ERROR = re.compile(r"incubator grasp lost before checkpoint: aperture=([0-9.]+) < ([0-9.]+)")


def audit(event_log=None):
    recovered = []
    files = sorted(set(RUNS.glob("incubator*20260808*.stderr")) |
                   set(RUNS.glob("incubator*20260808*/**/stderr.txt")))
    for path in files:
        raw = path.read_bytes()
        for value, bound in ERROR.findall(raw.decode()):
            recovered.append({"source": str(path.relative_to(ROOT)),
                              "sha256": hashlib.sha256(raw).hexdigest(),
                              "aperture_rounded_4dp": float(value), "bound_rounded_4dp": float(bound)})
    private = None
    if event_log:
        count = 0
        matches = []
        for number, line in enumerate(event_log.open(), 1):
            try:
                item = json.loads(line)
            except ValueError:
                continue
            timestamp = item.get("timestamp", "")
            if not "2026-08-08T03:34" <= timestamp <= "2026-08-08T10:10":
                continue
            payload = item.get("payload", {})
            if item.get("type") != "response_item" or payload.get("type") not in (
                    "custom_tool_call_output", "function_call_output"):
                continue
            count += 1
            output = payload.get("output", "")
            text = output if isinstance(output, str) else "\n".join(b.get("text", "") for b in output)
            for value, bound in ERROR.findall(text):
                matches.append({"line": number, "timestamp": timestamp,
                                "aperture_rounded_4dp": float(value), "bound_rounded_4dp": float(bound)})
        private = {"event_log_filename": event_log.name,
                   "window_utc": ["2026-08-08T03:34", "2026-08-08T10:10"],
                   "tool_output_records_scanned": count, "slip_errors": matches,
                   "note": "Repeated error outputs are not independent samples; do not concatenate as a time trace."}
    source = subprocess.check_output(["git", "show", "f2149bb:src/run_incubator_door_demo.py"], cwd=ROOT, text=True)
    return {"schema": "door_pull_trace_audit/v1", "stderr_files_scanned": len(files),
            "recovered_stop_readings": recovered, "private_event_log_audit": private,
            "historical_code": {"commit": "f2149bb", "path": "src/run_incubator_door_demo.py",
                                "sha256": hashlib.sha256(source.encode()).hexdigest()},
            "cause": [
                "Sparse aperture_checks accumulated in RAM inside _stream_retargeted_segment.gate.",
                "gate raises TrajectoryStreamError immediately on aperture loss.",
                "aperture_checkpoints is attached to the returned result only after streamer.execute returns normally.",
                "open_door.json is written later; interruption therefore loses accumulated checkpoint readings.",
                "stderr retains the failing reading rounded to 4 decimal places, without checkpoint time/index.",
                "The first full pull predates sparse checkpoint logging; its result has only before/after aperture.",
            ],
            "recoverability": "Individual stopping values recovered, including T3 and T5 previously absent from selected JSON observations; no complete measured full-pull curve found.",
            "not_valid_replacements": ["Binary demo gripper commands", "Arm joint torques", "Interpolated phase summaries"],
            "video_option": "Sparse still images cannot supply an actual continuous mechanical aperture trace. Calibrated continuous video, if separately found, would be a distinct visual estimate.",
            "runtime_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-log", type=Path)
    args = parser.parse_args()
    report = audit(args.event_log)
    path = ASSETS / "door_pull_trace_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
