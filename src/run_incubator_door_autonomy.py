#!/usr/bin/env python3
"""Run the full incubator open/close workflow without a Codex feedback loop.

The script is an evidence-gated orchestrator around the restartable motion
primitives in :mod:`src.run_incubator_door_demo`.  Every camera and motion
stage runs in a short-lived child process, which avoids retaining fragile
Record3D native state.  The default mode only prints the plan; ``--execute``
is required before any robot command can be sent.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rollout.articulated_appliance import (
    ApplianceState,
    build_endpoint_model,
    classify_endpoint_state,
    load_bundle_endpoint,
    load_endpoint,
    render_endpoint_evidence,
    workflow_stages,
)
from rollout.incubator_door_plane import estimate_bundle, wrap_degrees
from rollout.incubator_door_visual import extract_feature


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


class WorkflowFailure(RuntimeError):
    pass


class Orchestrator:
    def __init__(
        self,
        profile_path: Path,
        run_dir: Path,
        *,
        rpc_host: str,
        rpc_port: int,
    ) -> None:
        self.profile_path = profile_path.resolve()
        self.profile = _load(self.profile_path)
        self.settings = self.profile["autonomy"]
        self.state_settings = self.profile["state_detection"]
        self.run_dir = run_dir.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.rpc_host = rpc_host
        self.rpc_port = int(rpc_port)
        self.sequence = 0
        self.journal: dict[str, Any] = {
            "schema": "piper_robot.incubator_door_autonomy/v1",
            "profile": str(self.profile_path),
            "started_at_utc": _utc_stamp(),
            "rpc": {"host": rpc_host, "port": int(rpc_port)},
            "events": [],
            "commands_sent": False,
            "status": "running",
        }
        self.journal_path = self.run_dir / "journal.json"
        _write_json(self.journal_path, self.journal)
        references = self.state_settings["references"]
        opened = load_endpoint(
            _resolve(references["open"]["image"]),
            _resolve(references["open"]["depth"]),
        )
        closed = load_endpoint(
            _resolve(references["closed"]["image"]),
            _resolve(references["closed"]["depth"]),
        )
        self.endpoint_model = build_endpoint_model(
            opened, closed, self.state_settings
        )

    def record(self, event: str, **payload: Any) -> None:
        self.journal["events"].append(
            {"sequence": len(self.journal["events"]), "event": event, **payload}
        )
        _write_json(self.journal_path, self.journal)

    def finish(self, status: str, **payload: Any) -> dict:
        self.journal.update(
            {"status": status, "completed_at_utc": _utc_stamp(), **payload}
        )
        _write_json(self.journal_path, self.journal)
        return self.journal

    def _next_dir(self, label: str) -> Path:
        self.sequence += 1
        return self.run_dir / f"{self.sequence:02d}_{label}"

    def _run(self, label: str, command: list[str], *, sends_commands: bool) -> dict:
        output_dir = self._next_dir(label)
        output_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = output_dir / "stdout.txt"
        stderr_path = output_dir / "stderr.txt"
        self.record(
            "process-start",
            label=label,
            command=command,
            output_dir=str(output_dir),
            sends_commands=sends_commands,
        )
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)
        if sends_commands:
            self.journal["commands_sent"] = True
        if result.returncode != 0:
            self.record(
                "process-failed",
                label=label,
                returncode=result.returncode,
                stdout=str(stdout_path),
                stderr=str(stderr_path),
            )
            raise WorkflowFailure(
                f"{label} failed with exit {result.returncode}; see {stderr_path}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise WorkflowFailure(
                f"{label} did not return one JSON object; see {stdout_path}"
            ) from error
        self.record(
            "process-complete",
            label=label,
            output_dir=str(output_dir),
        )
        return {"payload": payload, "process_dir": output_dir}

    def run_motion_stage(self, stage: str, **options: Any) -> dict:
        # The motion primitive creates this directory itself, so reserve the
        # numbered parent here and give it a non-existent child.
        process_parent = self._next_dir(stage)
        process_parent.mkdir(parents=True, exist_ok=False)
        motion_output = process_parent / "motion"
        command = [
            sys.executable,
            "src/run_incubator_door_demo.py",
            "--profile",
            str(self.profile_path),
            "--rpc-host",
            self.rpc_host,
            "--rpc-port",
            str(self.rpc_port),
            "--output-dir",
            str(motion_output),
        ]
        for key, value in options.items():
            if value is None:
                continue
            command.extend(["--" + key.replace("_", "-"), str(value)])
        command.append(stage)
        stdout_path = process_parent / "stdout.txt"
        stderr_path = process_parent / "stderr.txt"
        self.record(
            "motion-start",
            stage=stage,
            command=command,
            output_dir=str(motion_output),
        )
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout_path.write_text(result.stdout)
        stderr_path.write_text(result.stderr)
        self.journal["commands_sent"] = True
        if result.returncode != 0:
            self.record(
                "motion-failed",
                stage=stage,
                returncode=result.returncode,
                stderr=str(stderr_path),
            )
            raise WorkflowFailure(
                f"motion stage {stage} failed; see {stderr_path}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise WorkflowFailure(
                f"motion stage {stage} returned invalid JSON; see {stdout_path}"
            ) from error
        self.record("motion-complete", stage=stage, output_dir=str(motion_output))
        return {"payload": payload, "motion_dir": motion_output}

    def capture_and_classify(self, label: str) -> dict:
        capture_root = self.run_dir / "captures"
        command = [
            sys.executable,
            "src/capture_record3d_bundle.py",
            "--camera",
            "head",
            "--frames",
            str(int(self.settings.get("state_capture_frames", 8))),
            "--warmup-frames",
            str(int(self.settings.get("state_capture_warmup_frames", 5))),
            "--timeout-s",
            str(float(self.settings.get("state_capture_timeout_s", 20.0))),
            "--output-root",
            str(capture_root),
            "--condition",
            label,
            "--robot-state",
            "--robot-host",
            self.rpc_host,
            "--robot-port",
            str(self.rpc_port),
        ]
        captured = self._run(
            f"capture_{label}", command, sends_commands=False
        )["payload"]
        capture_dir = Path(captured["session_dir"])
        live = load_bundle_endpoint(capture_dir)
        report = classify_endpoint_state(
            live, self.endpoint_model, self.state_settings
        )
        evidence_dir = self._next_dir(f"state_{label}")
        evidence_dir.mkdir(parents=True, exist_ok=False)
        _write_json(evidence_dir / "state.json", report)
        cv2.imwrite(
            str(evidence_dir / "state_evidence.png"),
            render_endpoint_evidence(live, self.endpoint_model, report),
        )
        report["capture_dir"] = str(capture_dir.resolve())
        report["evidence_image"] = str(
            (evidence_dir / "state_evidence.png").resolve()
        )
        self.record("state-observed", label=label, report=report)
        return report

    def estimate_closed_plane_yaw(self, capture_dir: str | Path) -> dict:
        report = estimate_bundle(
            capture_dir,
            _resolve(self.profile["door_plane"]["tag_config"]),
            self.profile["door_plane"],
        )
        maximum_std = float(self.settings["maximum_plane_yaw_std_deg"])
        if report["normal_yaw_std_deg"] > maximum_std:
            raise WorkflowFailure(
                "incubator plane yaw is unstable: "
                f"{report['normal_yaw_std_deg']:.2f}deg > {maximum_std:.2f}deg"
            )
        path = self.run_dir / "closed_plane.json"
        _write_json(path, report)
        self.record("closed-plane-estimated", report=report, path=str(path))
        return report

    def visual_converged(self, image_path: str | Path) -> dict:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise WorkflowFailure(f"cannot read right-camera image: {image_path}")
        feature, feature_report = extract_feature(
            image, self.profile["visual_feature"]
        )
        compiled = _load(_resolve(self.profile["compiled_demo"]))
        goal = np.asarray(compiled["visual_servo"]["goal_feature_mean"], dtype=float)
        uv_error = float(np.linalg.norm(goal[:2] - feature[:2]))
        log_area_error = float(abs(goal[2] - feature[2]))
        accepted = bool(
            uv_error <= float(self.settings["maximum_visual_uv_error"])
            and log_area_error
            <= float(self.settings["maximum_visual_log_area_error"])
        )
        return {
            "accepted": accepted,
            "uv_error": uv_error,
            "log_area_error": log_area_error,
            "feature": feature_report,
            "goal_feature_uv_log_area": goal.tolist(),
        }

    @staticmethod
    def _nested(result: dict, key: str) -> dict:
        payload = result["payload"]
        if key not in payload:
            raise WorkflowFailure(f"motion result is missing {key}")
        return payload[key]

    def run_open(self, initial: dict) -> dict:
        if initial["state"] == ApplianceState.OPEN.value:
            return self.finish("goal-already-satisfied", final_state=initial)
        if initial["state"] != ApplianceState.CLOSED.value:
            raise WorkflowFailure("opening requires a confidently closed initial state")
        plane = self.estimate_closed_plane_yaw(initial["capture_dir"])
        reference_yaw = float(self.settings["reference_closed_plane_yaw_deg"])
        yaw_delta = wrap_degrees(plane["normal_yaw_deg"] - reference_yaw)
        maximum_attempts = int(self.settings.get("maximum_open_attempts", 2))
        aligned_reference = _resolve(self.settings["reference_aligned_preclose_state"])

        for attempt in range(1, maximum_attempts + 1):
            self.record("open-attempt-start", attempt=attempt, yaw_delta_deg=yaw_delta)
            preclose = self.run_motion_stage(
                "aligned-yaw-preclose",
                aligned_state=aligned_reference,
                world_yaw_deg=yaw_delta,
            )
            preclose_state = self._nested(preclose, "aligned_yaw_preclose")
            visual = self.visual_converged(
                preclose_state["after"]["image_paths"]["right"]
            )
            self.record("visual-check", attempt=attempt, step=0, report=visual)
            for step in range(1, int(self.settings["maximum_visual_steps"]) + 1):
                if visual["accepted"]:
                    break
                aligned = self.run_motion_stage("visual-align-step")
                aligned_state = self._nested(aligned, "visual_alignment")
                visual = self.visual_converged(
                    aligned_state["after"]["image_paths"]["right"]
                )
                self.record("visual-check", attempt=attempt, step=step, report=visual)
            if not visual["accepted"]:
                raise WorkflowFailure(
                    "bounded right-camera alignment did not converge; contact denied"
                )

            contact = self.run_motion_stage("aligned-contact")
            contact_path = Path(contact["motion_dir"]) / "aligned_contact.json"
            closed = self.run_motion_stage(
                "close-verify", contact_state=contact_path
            )
            contact_state = self._nested(closed, "close_verify")["contact_state"]
            if not contact_state["stable_nonempty"]:
                self.record("grasp-rejected", attempt=attempt, reason="empty close")
                self.run_motion_stage("recover-empty-close")
                continue

            contact_state_path = Path(closed["motion_dir"]) / "contact_state.json"
            proof = self.run_motion_stage(
                "proof-pull", contact_state=contact_state_path
            )
            proof_state = self._nested(proof, "proof_pull")["proof_state"]
            if not proof_state["proof_retained"]:
                self.record("grasp-rejected", attempt=attempt, reason="5mm proof lost")
                self.run_motion_stage("recover-empty-close")
                continue
            proof_path = Path(proof["motion_dir"]) / "proof_state.json"
            verified = self.run_motion_stage(
                "reverify-proof", proof_state=proof_path
            )
            verified_state = self._nested(verified, "reverify_proof")["proof_state"]
            if not verified_state["proof_retained"]:
                self.record(
                    "grasp-rejected", attempt=attempt, reason="stationary proof lost"
                )
                self.run_motion_stage("recover-empty-close")
                continue

            verified_path = Path(verified["motion_dir"]) / "proof_state.json"
            try:
                self.run_motion_stage("open-door", proof_state=verified_path)
            except WorkflowFailure as error:
                # A late aperture checkpoint can fire after the door has
                # already opened.  Retreat/release, then classify before any
                # retry; never blindly repeat a pull.
                self.record("open-pull-interrupted", attempt=attempt, error=str(error))
            self.run_motion_stage("recover-empty-close")
            final = self.capture_and_classify(f"open_attempt_{attempt}_result")
            if final["state"] == ApplianceState.OPEN.value:
                return self.finish("success", goal="open", final_state=final)
            if final["state"] == ApplianceState.UNKNOWN.value:
                raise WorkflowFailure(
                    "open pull finished but endpoint evidence is ambiguous; no retry sent"
                )
            self.record("open-attempt-not-open", attempt=attempt, final_state=final)
        raise WorkflowFailure("verified opening attempts exhausted")

    def run_close(self, initial: dict) -> dict:
        if initial["state"] == ApplianceState.CLOSED.value:
            return self.finish("goal-already-satisfied", final_state=initial)
        if initial["state"] != ApplianceState.OPEN.value:
            raise WorkflowFailure("closing requires a confidently open initial state")
        self.run_motion_stage("close-door-demo")
        final = self.capture_and_classify("close_result")
        if final["state"] != ApplianceState.CLOSED.value:
            raise WorkflowFailure(
                "dedicated close demo did not produce closed endpoint evidence; "
                "no extra terminal push was sent"
            )
        return self.finish("success", goal="closed", final_state=final)

    def execute(self, goal: str) -> dict:
        self.journal["goal"] = goal
        _write_json(self.journal_path, self.journal)
        initial = self.capture_and_classify("initial")
        if goal == ApplianceState.OPEN.value:
            return self.run_open(initial)
        return self.run_close(initial)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("goal", choices=("open", "closed"))
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("src/configs/pasteur_incubator_door_demo.json"),
    )
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=8081)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="capture live evidence and permit the staged runner to command the robot",
    )
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "goal": args.goal,
                    "commands_sent": False,
                    "execution_requires": "--execute",
                    "stages": workflow_stages(args.goal),
                },
                indent=2,
            )
        )
        return 0
    output = args.output_dir or Path(
        "data/runs/pasteur"
    ) / f"incubator_autonomy_{_utc_stamp()}_{args.goal}"
    orchestrator: Orchestrator | None = None
    try:
        orchestrator = Orchestrator(
            args.profile,
            output,
            rpc_host=args.rpc_host,
            rpc_port=args.rpc_port,
        )
        result = orchestrator.execute(args.goal)
    except Exception as error:
        if orchestrator is not None:
            result = orchestrator.finish(
                "failed",
                error=f"{type(error).__name__}: {error}",
            )
        else:
            result = {
                "status": "failed",
                "commands_sent": False,
                "error": f"{type(error).__name__}: {error}",
            }
        print(json.dumps(result, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
