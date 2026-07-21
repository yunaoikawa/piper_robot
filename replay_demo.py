#!/usr/bin/env python3
"""Replay ONE recorded demo through the bias + safety pipeline. No policy server.

The premise: for a fixed lab task where the object barely moves (the petri /
lid / pipette placements vary only ~2-3 cm; see outputs/lab/act/horizon/
EVAL_RESULTS.md), you may not need a policy at all. Record one good
demonstration, then replay its end-effector trajectory. When the object sits a
little off from the demo, shift the WHOLE trajectory with `src/set_bias.py` --
or have Claude/Codex look at the head camera and set the bias.

Why go through PolicyController instead of driving the arm directly: so the
exact same safety and bias layers a policy rollout uses apply here too --
workspace clamp (robot/cone_e.py), keep-out zones + per-step motion cap
(rollout/safety.py), and the per-arm xyz bias. The ONLY thing swapped out is
where actions come from: a recorded HDF5 instead of the ZMQ inference server.

OPEN-LOOP CAVEAT (read before trusting it):
  The trajectory shape is fixed. A bias TRANSLATES the whole path; it does not
  re-time or re-shape it. If the object rotates, moves more than a few cm, or
  the grasp needs a mid-motion correction, replay will not adapt -- that is what
  the policy (ACT / pi0.5) is for. Use replay only for near-fixed placements.

NO COLLISION REACTION:
  The arms expose torque (piperlib JointState has .torque) but nothing here
  reads it yet. Every safety check is preventive/geometric. Keep a hand on the
  stop.

Usage (on pasteur, in the robot-control env):
  python replay_demo.py path/to/episode.hdf5 --safety-config src/configs/safety.json
  # then press 's' to start, 'e' to end, 'q' to quit (keyboard controller)
  # in another shell, adjust live:  python src/set_bias.py --z -0.02
"""

import argparse
import time

import h5py
import numpy as np

from rollout.controller import PolicyController


# HDF5 field names written by rollout/recorder.py.
_POS = "{side}_ee_pos"
_QUAT = "{side}_ee_quat"     # stored wxyz -- matches mink.SE3 / the action wire format
_GRIP = "{side}_gripper"     # binary open/close channel


def load_demo(path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        demo = {k: f[k][()] for k in keys}
    n = len(demo["timestamps"])
    print(f"loaded {n} frames, keys: {keys}")
    for side in ("left", "right"):
        need = [_POS.format(side=side), _QUAT.format(side=side), _GRIP.format(side=side)]
        missing = [k for k in need if k not in demo]
        if missing:
            raise SystemExit(f"demo is missing {missing} -- was it recorded by rollout/recorder.py?")
    return demo, n


def frame_to_action(demo, i):
    """One HDF5 frame -> the absolute-pose action dict apply_action() expects.

    right_ee_pose / left_ee_pose are (7,) = [quat_wxyz(4), pos(3)], exactly what
    mink.SE3 and _apply_arm_action_absolute() consume. total_buffer_updates is
    held constant: this is one continuous trajectory, not a re-planning policy,
    so the safety step reference should persist across the whole replay (it is
    reset once at episode start, and again on any live set_bias).
    """
    def pose(side):
        pos = np.asarray(demo[_POS.format(side=side)][i], dtype=float)
        quat = np.asarray(demo[_QUAT.format(side=side)][i], dtype=float)  # wxyz
        return np.concatenate([quat, pos])

    return {
        "left_ee_pose": pose("left"),
        "right_ee_pose": pose("right"),
        "left_gripper": float(demo[_GRIP.format(side="left")][i]),
        "right_gripper": float(demo[_GRIP.format(side="right")][i]),
        "total_buffer_updates": 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", help="recorded HDF5 episode (from rollout/recorder.py)")
    ap.add_argument("--safety-config", default="src/configs/safety.json")
    ap.add_argument("--bias-port", type=int, default=5560)
    ap.add_argument("--rate", type=float, default=30.0,
                    help="replay Hz. Demos are recorded at 30; slower is safer for a first run.")
    ap.add_argument("--host", default="localhost",
                    help="dummy -- no inference server is contacted, but the controller "
                         "still opens its (unused) action sockets.")
    args = ap.parse_args()

    demo, n = load_demo(args.demo)

    controller = PolicyController(
        hpc_host=args.host,
        enable_recording=False,
        safety_config=args.safety_config,
        bias_port=args.bias_port,
        task="replay",
    )

    print("\nController up (arms homed). Press 's' to START replay, "
          "'e' to end, 'q' to quit.")
    print("Adjust live from another shell:  python src/set_bias.py --z -0.02\n")

    # Wait for the keyboard controller to mark the episode active ('s').
    while not controller.episode_manager.is_active():
        if controller.stop_event.is_set():
            controller.stop()
            return
        time.sleep(0.1)

    print(f"replaying {n} frames at {args.rate} Hz")
    dt = 1.0 / args.rate
    try:
        for i in range(n):
            if controller.stop_event.is_set() or not controller.episode_manager.is_active():
                print(f"interrupted at frame {i}/{n}")
                break
            controller.apply_action(frame_to_action(demo, i))
            if i % 30 == 0:
                print(f"  frame {i}/{n}")
            time.sleep(dt)
        else:
            print("replay complete")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
