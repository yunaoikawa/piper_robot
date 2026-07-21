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

CONTACT REACTION:
  A calibrated per-joint torque watchdog is required for normal live replay.
  Use --calibrate-torque once on a supervised known-good replay, then pass the
  resulting file with --torque-config.

Usage (on pasteur, in the robot-control env):
  python replay_demo.py path/to/episode.hdf5 --safety-config src/configs/safety.json
  # then press 's' to start, 'e' to end, 'q' to quit (keyboard controller)
  # in another shell, adjust live:  python src/set_bias.py --z -0.02
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import zmq

from robot.cone_e import WORKSPACE_MAX, WORKSPACE_MIN
from rollout.controller import PolicyController
from rollout.lid_vision import VisionProfile, inspect_lid, render_inspection
from rollout.safety import MAX_STEP_M
from rollout.torque_safety import TorqueCalibrator, TorqueWatchdog


# HDF5 field names written by rollout/recorder.py.
_POS = "{side}_ee_pos"
_QUAT = "{side}_ee_quat"     # stored wxyz -- matches mink.SE3 / the action wire format
_GRIP = "{side}_gripper"     # binary open/close channel

ARM_TRANSLATION_EPS_M = 0.001
ARM_ROTATION_EPS_DEG = 1.0
ARM_GRIPPER_EPS = 0.1
DEFAULT_MAX_START_DISTANCE_M = 0.040
DEFAULT_MAX_START_ANGLE_DEG = 15.0
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


def _quat_angle_deg(q0, q1):
    """Shortest angular distance between two wxyz quaternions."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def detect_active_arms(demo):
    """Return arms with meaningful pose or gripper motion in the demo."""
    active = []
    metrics = {}
    for side in ("left", "right"):
        pos = np.asarray(demo[_POS.format(side=side)], dtype=float)
        quat = np.asarray(demo[_QUAT.format(side=side)], dtype=float)
        grip = np.asarray(demo[_GRIP.format(side=side)], dtype=float)
        translation = float(np.max(np.linalg.norm(pos - pos[0], axis=1)))
        rotation = max(_quat_angle_deg(quat[0], q) for q in quat)
        gripper = float(np.ptp(grip))
        metrics[side] = {
            "translation_m": translation,
            "rotation_deg": rotation,
            "gripper_range": gripper,
        }
        if (translation > ARM_TRANSLATION_EPS_M
                or rotation > ARM_ROTATION_EPS_DEG
                or gripper > ARM_GRIPPER_EPS):
            active.append(side)
    return active, metrics


def resolve_arms(selection, detected):
    if selection == "auto":
        if not detected:
            raise SystemExit("no active arm detected -- use --arms left/right/both to override")
        return detected
    if selection == "both":
        return ["left", "right"]
    return [selection]


def _max_step_from_config(path):
    if not path:
        return MAX_STEP_M
    with open(path) as f:
        return float(json.load(f).get("max_step_m", MAX_STEP_M))


def validate_demo(demo, n, active_arms, max_step_m):
    """Offline validation. Raises SystemExit before any hardware is touched."""
    errors = []
    if n < 2:
        errors.append(f"demo has {n} frame(s); need at least 2")
    for key, value in demo.items():
        if len(value) != n:
            errors.append(f"{key} has {len(value)} frames, expected {n}")

    for side in active_arms:
        pos = np.asarray(demo[_POS.format(side=side)], dtype=float)
        quat = np.asarray(demo[_QUAT.format(side=side)], dtype=float)
        grip = np.asarray(demo[_GRIP.format(side=side)], dtype=float)
        if not all(np.all(np.isfinite(x)) for x in (pos, quat, grip)):
            errors.append(f"{side}: non-finite pose or gripper value")
            continue
        qnorm = np.linalg.norm(quat, axis=1)
        if np.any(np.abs(qnorm - 1.0) > 1e-3):
            errors.append(f"{side}: quaternion norm outside 1±0.001")
        outside = np.any((pos < WORKSPACE_MIN) | (pos > WORKSPACE_MAX), axis=1)
        if np.any(outside):
            errors.append(f"{side}: {int(np.count_nonzero(outside))}/{n} positions outside workspace")
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        max_step = float(np.max(steps)) if len(steps) else 0.0
        if max_step > max_step_m:
            errors.append(
                f"{side}: max step {max_step * 1000:.1f}mm exceeds "
                f"{max_step_m * 1000:.1f}mm"
            )
        print(f"  {side}: max step {max_step * 1000:.1f}mm; "
              f"range {np.round(pos.min(0), 3)} .. {np.round(pos.max(0), 3)}")

    if errors:
        raise SystemExit("demo preflight FAILED:\n  " + "\n  ".join(errors))


def validate_start_pose(controller, demo, active_arms,
                        max_distance_m=DEFAULT_MAX_START_DISTANCE_M,
                        max_angle_deg=DEFAULT_MAX_START_ANGLE_DEG):
    """Refuse a first-frame jump after the controller has homed the arms."""
    errors = []
    for side in active_arms:
        current = getattr(controller.cone_e, f"get_{side}_ee_pose")()
        target_pos = np.asarray(demo[_POS.format(side=side)][0], dtype=float)
        target_quat = np.asarray(demo[_QUAT.format(side=side)][0], dtype=float)
        distance = float(np.linalg.norm(target_pos - current.translation()))
        angle = _quat_angle_deg(target_quat, current.rotation().wxyz)
        print(f"  {side} start delta: {distance * 1000:.1f}mm, {angle:.1f}deg")
        if distance > max_distance_m:
            errors.append(
                f"{side}: start distance {distance * 1000:.1f}mm exceeds "
                f"{max_distance_m * 1000:.1f}mm"
            )
        if angle > max_angle_deg:
            errors.append(
                f"{side}: start angle {angle:.1f}deg exceeds {max_angle_deg:.1f}deg"
            )
    if errors:
        raise SystemExit("live start preflight FAILED:\n  " + "\n  ".join(errors))


def frame_to_action(demo, i, active_arms=("left", "right")):
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

    action = {"total_buffer_updates": 1}
    for side in active_arms:
        action[f"{side}_ee_pose"] = pose(side)
        action[f"{side}_gripper"] = float(demo[_GRIP.format(side=side)][i])
    return action


def longest_grip_start(demo, active_arms=("left", "right"), closed_threshold=0.5):
    """Start of the longest contiguous closed-gripper run across active arms."""
    best = None
    for side in active_arms:
        closed = np.asarray(demo[_GRIP.format(side=side)], dtype=float) < closed_threshold
        start = None
        for i, value in enumerate(np.r_[closed, False]):
            if value and start is None:
                start = i
            elif not value and start is not None:
                candidate = (i - start, -start, side, start)
                if best is None or candidate > best:
                    best = candidate
                start = None
    return None if best is None else {"arm": best[2], "frame": best[3], "length": best[0]}


def read_joint_torques(controller, active_arms):
    return {
        arm: np.asarray(getattr(controller.cone_e, f"get_{arm}_joint_torque")(), dtype=float)
        for arm in active_arms
    }


def check_torque(controller, active_arms, watchdog=None, calibrator=None):
    samples = read_joint_torques(controller, active_arms)
    for arm, values in samples.items():
        if calibrator is not None:
            calibrator.add(arm, values)
        if watchdog is not None and not watchdog.check(arm, values):
            print(f"[torque] STOP: {watchdog.tripped}", flush=True)
            return False
    return True


def camera_rgb_to_bgr(image):
    """Rotate a Record3D frame into replay display coordinates and convert color."""
    return cv2.cvtColor(np.rot90(image, k=3), cv2.COLOR_RGB2BGR)


def _save_checkpoint_images(controller, directory, frame, sequence, vision_profile=None):
    """Save head/right-wrist views while the arm holds its checkpoint pose."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    sources = {
        "head": controller.camera,
        "right": controller.right_wrist_camera,
    }
    for label, camera in sources.items():
        if camera is None:
            continue
        image, _, _ = camera.get_latest_frame()
        if image is None:
            continue
        image = camera_rgb_to_bgr(image)
        path = directory / f"frame_{frame:04d}_{sequence:02d}_{label}.png"
        # Both head and wrist managers expose Record3D RGB frames.
        cv2.imwrite(str(path), image)
        paths[label] = str(path)
        if label == "right" and vision_profile is not None:
            result = inspect_lid(image, vision_profile)
            overlay = render_inspection(image, result)
            overlay_path = directory / f"frame_{frame:04d}_{sequence:02d}_right_overlay.png"
            cv2.imwrite(str(overlay_path), overlay)
            paths["right_overlay"] = str(overlay_path)
            paths["vision"] = {
                "ok": bool(result["ok"]),
                "reason": result.get("reason"),
                "edge_points": int(result.get("edge_points", 0)),
                "marker": result.get("marker").tolist() if result.get("marker") is not None else None,
            }
    return paths


def checkpoint_loop(controller, demo, frame, active_arms, port, directory,
                    vision_profile=None, watchdog=None, calibrator=None):
    """Hold at a frame until an external client resumes or aborts."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{port}")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    sequence = 0
    time.sleep(0.75)
    paths = _save_checkpoint_images(controller, directory, frame, sequence, vision_profile)
    print(f"\nCHECKPOINT frame {frame}: holding pose; images={paths}")
    print(f"Control: python src/replay_checkpoint.py --port {port} --status")

    try:
        while not controller.stop_event.is_set():
            if not check_torque(controller, active_arms, watchdog, calibrator):
                return False
            if not poller.poll(33):
                continue
            command = socket.recv_pyobj()
            name = command.get("command")
            if name == "status":
                reply = {
                    "status": "ok", "state": "holding", "frame": frame,
                    "active_arms": list(active_arms), "images": paths,
                    "bias": {k: v.tolist() for k, v in controller.xyz_bias.items()},
                    "torque_watchdog": watchdog.tripped if watchdog else None,
                }
            elif name == "snapshot":
                sequence += 1
                paths = _save_checkpoint_images(controller, directory, frame, sequence, vision_profile)
                reply = {"status": "ok", "frame": frame, "images": paths}
            elif name == "adjust":
                arm = command.get("arm", "right")
                if arm not in active_arms:
                    reply = {"status": "error", "message": f"{arm} is not active"}
                else:
                    requested = np.asarray(command.get("bias"), dtype=float).reshape(3)
                    applied = controller.set_bias(arm, requested)
                    controller.apply_action(frame_to_action(demo, frame, active_arms))
                    time.sleep(0.75)
                    sequence += 1
                    paths = _save_checkpoint_images(
                        controller, directory, frame, sequence, vision_profile)
                    reply = {
                        "status": "ok", "frame": frame,
                        "arm": arm, "bias": applied.tolist(), "images": paths,
                    }
            elif name == "resume":
                socket.send_pyobj({"status": "ok", "state": "resuming", "frame": frame})
                return True
            elif name == "abort":
                socket.send_pyobj({"status": "ok", "state": "aborting", "frame": frame})
                return False
            else:
                reply = {"status": "error", "message": f"unknown command {name!r}"}
            socket.send_pyobj(reply)
    finally:
        socket.close()
        context.term()
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", help="recorded HDF5 episode (from rollout/recorder.py)")
    ap.add_argument("--safety-config", default="src/configs/safety.json")
    ap.add_argument("--bias-port", type=int, default=5560)
    ap.add_argument("--rate", type=float, default=30.0,
                    help="replay Hz. Demos are recorded at 30; slower is safer for a first run.")
    ap.add_argument("--arms", choices=("auto", "left", "right", "both"), default="auto",
                    help="arms to command (default: detect meaningful motion automatically)")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and print the demo without connecting to robot hardware")
    ap.add_argument("--max-start-distance", type=float,
                    default=DEFAULT_MAX_START_DISTANCE_M, metavar="METRES")
    ap.add_argument("--max-start-angle", type=float,
                    default=DEFAULT_MAX_START_ANGLE_DEG, metavar="DEGREES")
    ap.add_argument("--checkpoint-frame", type=int, action="append", default=[],
                    help="hold after applying this frame; may be specified more than once")
    ap.add_argument("--checkpoint-port", type=int, default=5561)
    ap.add_argument("--checkpoint-dir", default="/tmp/pasteur_replay_checkpoints")
    ap.add_argument("--vision-profile", help="marker/transparent-edge JSON profile")
    torque = ap.add_mutually_exclusive_group()
    torque.add_argument("--torque-config", help="calibrated joint-torque watchdog JSON")
    torque.add_argument("--calibrate-torque", metavar="OUTPUT_JSON",
                        help="supervised known-good replay; write calibrated thresholds")
    ap.add_argument("--host", default="localhost",
                    help="dummy -- no inference server is contacted, but the controller "
                         "still opens its (unused) action sockets.")
    args = ap.parse_args()

    demo, n = load_demo(args.demo)
    vision_profile = VisionProfile.load(args.vision_profile) if args.vision_profile else None
    detected, metrics = detect_active_arms(demo)
    active_arms = resolve_arms(args.arms, detected)
    grip_goal = longest_grip_start(demo, active_arms)
    print("arm motion metrics:")
    for side, m in metrics.items():
        print(f"  {side}: translation={m['translation_m'] * 1000:.1f}mm, "
              f"rotation={m['rotation_deg']:.1f}deg, gripper_range={m['gripper_range']:.1f}")
    print(f"active arms: {', '.join(active_arms)} ({args.arms})")
    print(f"longest closed-gripper run: {grip_goal}")
    validate_demo(demo, n, active_arms, _max_step_from_config(args.safety_config))
    checkpoint_frames = set(args.checkpoint_frame)
    if vision_profile is not None:
        checkpoint_frames.add(vision_profile.goal_frame)
        print(f"vision goal checkpoint: frame {vision_profile.goal_frame}")
    invalid_checkpoints = [i for i in checkpoint_frames if not 0 <= i < n]
    if invalid_checkpoints:
        raise SystemExit(f"checkpoint frame(s) out of range 0..{n - 1}: {invalid_checkpoints}")
    print("offline demo preflight PASSED")
    if args.dry_run:
        return
    if not args.torque_config and not args.calibrate_torque:
        raise SystemExit("live replay requires --torque-config or supervised --calibrate-torque")

    watchdog = TorqueWatchdog.from_file(args.torque_config) if args.torque_config else None
    calibrator = TorqueCalibrator() if args.calibrate_torque else None
    if watchdog is not None:
        missing = sorted(set(active_arms) - set(watchdog.thresholds))
        if missing:
            raise SystemExit(f"torque config has no thresholds for active arm(s): {missing}")

    controller = PolicyController(
        hpc_host=args.host,
        enable_recording=False,
        safety_config=args.safety_config,
        bias_port=args.bias_port,
        task="replay",
    )

    try:
        print("live start preflight:")
        validate_start_pose(
            controller, demo, active_arms,
            max_distance_m=args.max_start_distance,
            max_angle_deg=args.max_start_angle,
        )
    except BaseException:
        controller.stop()
        raise

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
            if not check_torque(controller, active_arms, watchdog, calibrator):
                print(f"torque watchdog aborted at frame {i}/{n}; holding last target")
                break
            controller.apply_action(frame_to_action(demo, i, active_arms))
            if i % 30 == 0:
                print(f"  frame {i}/{n}")
            time.sleep(dt)
            if i in checkpoint_frames:
                if not checkpoint_loop(
                        controller, demo, i, active_arms,
                        args.checkpoint_port, args.checkpoint_dir,
                        vision_profile, watchdog, calibrator):
                    print(f"aborted at checkpoint frame {i}/{n}")
                    break
        else:
            print("replay complete")
    finally:
        if calibrator is not None:
            if calibrator.samples:
                calibrator.save(args.calibrate_torque)
                print(f"saved torque calibration to {args.calibrate_torque}: "
                      f"{ {arm: len(values) for arm, values in calibrator.samples.items()} }")
            else:
                print("no torque samples collected; calibration file not written")
        controller.stop()


if __name__ == "__main__":
    main()
