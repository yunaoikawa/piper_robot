#!/usr/bin/env python3
"""Run ACT under checkpointed agentic data-collection supervision."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rollout import PolicyController
from rollout.agentic_policy_supervisor import AgenticPolicySupervisor
from rollout.agentic_collection_ui import make_agentic_ui


DEFAULT_PROFILE = ROOT / "src/configs/pasteur_agentic_petri_collection.json"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="ACT-primary petri collection with deterministic checkpoint verification"
    )
    value.add_argument("--profile", default=str(DEFAULT_PROFILE))
    value.add_argument("--mode", choices=("shadow", "supervised", "auto"), default="shadow")
    value.add_argument("--armed", action="store_true", help="One-time authorization for this live session")
    value.add_argument("--condition", choices=("ring", "no_ring", "unknown"), default="unknown")
    value.add_argument("--output-dir", default="data/agentic_collection")
    value.add_argument("--save-dir", default="teleop_demonstrations_agentic")
    value.add_argument("--host", default="192.168.1.50")
    value.add_argument("--obs-port", type=int, default=5555)
    value.add_argument("--action-port", type=int, default=5556)
    value.add_argument("--rate", type=int, default=30)
    value.add_argument("--episode-timeout", type=float, default=90.0)
    value.add_argument("--attach-current", action="store_true")
    value.add_argument("--no-display", action="store_true")
    value.add_argument("--no-head-stream", action="store_true")
    value.add_argument("--head-stream-port", type=int, default=8080)
    value.add_argument("--ui-host", default="0.0.0.0")
    value.add_argument("--ui-port", type=int, default=8098)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.mode != "shadow" and not args.armed:
        raise SystemExit("live supervised/auto collection requires explicit --armed")
    if args.mode == "shadow" and args.armed:
        raise SystemExit("--armed is invalid in shadow mode; shadow never sends robot commands")

    supervisor = AgenticPolicySupervisor(
        args.profile,
        mode=args.mode,
        condition=args.condition,
        armed=args.armed,
        output_dir=args.output_dir,
    )
    ui = None
    controller = None
    try:
        controller = PolicyController(
            hpc_host=args.host,
            obs_port=args.obs_port,
            action_port=args.action_port,
            enable_recording=True,
            save_dir=args.save_dir,
            autonomous_mode=True,
            episode_timeout=args.episode_timeout,
            task=supervisor.instruction,
            display=not args.no_display,
            head_stream=not args.no_head_stream,
            head_stream_port=args.head_stream_port,
            # Shadow must be motion-free, including startup homing.
            home_on_init=(args.mode != "shadow" and not args.attach_current),
            agentic_supervisor=supervisor,
        )
        ui, ui_thread = make_agentic_ui(supervisor, args.ui_host, args.ui_port)
        ui_thread.start()
        print(f"[agentic] mode={args.mode} armed={args.armed}")
        print(f"[agentic] cycle starts with: {supervisor.task.name}")
        print(f"[agentic] audit: {supervisor.run_dir}")
        print(f"[agentic] phone UI: http://<pasteur-ip>:{args.ui_port}/")
        controller.control_loop(control_rate=args.rate)
    except KeyboardInterrupt:
        return 130
    finally:
        if controller is not None:
            controller.stop()
        else:
            supervisor.close()
        if ui is not None:
            ui.shutdown()
            ui.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
