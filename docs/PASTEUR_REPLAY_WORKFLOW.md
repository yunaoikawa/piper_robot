# pasteur workflow: one-demo replay + Claude/Codex bias adjustment

Handoff for whoever (Codex, Claude Code, or a human) drives the robot on
**pasteur** without needing peacock. Written 2026-07-21.

## The idea

For a **fixed lab task** where the object barely moves, you may not need a
learned policy at all. Record **one** good demonstration, replay its
end-effector trajectory, and when the object sits a little off, **shift the
whole trajectory with a bias**. A vision model (Claude/Codex looking at the head
camera) can propose that bias. The learned policy (ACT / pi0.5) is kept for
closed-loop adaptation and for the ablation comparison later — not for daily runs
of a fixed task.

This is deliberately the *simplest thing that could work*. Know its limits
before trusting it.

For placements not seen in the teacher episode, use the AprilTag retargeting
path instead of searching a constant bias. The lid carries a 30 mm tag and the
workspace references use 60 mm tags. IDs and dictionary family are discovered
from the image; they do not need to be typed by the operator.

```bash
# Discover IDs and save an annotated image. This does not move the robot.
# Stop any other process holding Record3D cameras first (including cone_e).
python src/calibrate_apriltag_workspace.py --capture-head \
  --output src/configs/pasteur_lid_tags.json

# Register at least three fixed tag centers after measuring their robot XY.
python src/calibrate_apriltag_workspace.py --image head.png \
  --output src/configs/pasteur_lid_tags.json \
  --fixed ID X Y --fixed ID X Y --fixed ID X Y \
  --reference-lid X Y YAW_RAD \
  --reference-wrist-corners X0 Y0 X1 Y1 X2 Y2 X3 Y3

# Unseen planar placement: head-tag retarget + wrist closed-loop servo.
python replay_demo.py demo.hdf5 --tag-profile src/configs/pasteur_lid_tags.json \
  --auto-align --torque-config src/configs/pasteur_lid_torque.json
```

The shipped tag profile is deliberately incomplete until the fixed-tag robot
coordinates and one successful reference grasp are registered. `--auto-align`
fails closed when calibration, three fixed tags, or the lid tag is missing.
All four black edges of at least three 60 mm fixed tags must be inside the head
image; a tag clipped by the image boundary or hidden by an arm cannot be decoded.

## What is grounded in measurement (don't re-lit­igate these)

From `outputs/lab/act/horizon/EVAL_RESULTS.md`:

- **Object placement varies only ~2–3 cm** across demos of a given task. That is
  why a fixed trajectory + small bias is viable at all.
- **A VLM's position accuracy is ~0.6 cm, and it is a noise floor** — roughly
  independent of how far the object moved. It only beats "just use the demo
  trajectory unchanged" once the object is off by **more than ~5 cm**. Below
  that, the demo is already close and the bias is a fine-tune.
- **At full 960×640 resolution the VLM cannot see the difference between
  placements.** Lids 2–4 cm apart looked identical until the image was cropped
  to the ROI, contrast-stretched, and upscaled ~5×. **A bias-from-vision step
  MUST preprocess the frame this way** — do not hand a raw frame to the model and
  expect cm-level judgement.

## Hard limits (design constraints, not opinions)

1. **Replay is open-loop.** The bias TRANSLATES the whole path; it does not
   re-time or re-shape it. Object rotation, moves > a few cm, or a grasp that
   needs mid-motion correction → replay will not adapt. That is the policy's job.
2. **Torque reaction must be calibrated.** The replay reads
   `piperlib.JointState.torque` and stops target submission after five consecutive
   per-joint threshold violations. Thresholds are hardware/task-specific, so a
   supervised known-good calibration is mandatory. **Keep a hand on the stop.**

## Architecture

```
 trajectory source ─┬─ replay_demo.py     (recorded HDF5)        <- use this
                    └─ cloud_inference_control_collect_v2.py (ZMQ policy server)
                              │
                     PolicyController.apply_action()   (rollout/controller.py)
                              │
                    xyz_bias  (per-arm, robot frame, live-settable; finite values)
                              │
                    SafetyLayer (rollout/safety.py)
                       • keep-out zones            (src/configs/safety.json)
                       • per-step motion cap 40 mm (demo max was 30 mm)
                       • reject → arm holds
                              │
                    cone_e.set_*_ee_target()  ->  clamp_ee_target()  (robot/cone_e.py)
                       • workspace box, calibrated for commanded active-arm frames
```

Both trajectory sources go through the *same* bias + safety + clamp path, so
anything you tune on replay carries over to the policy and vice-versa.

## Files (all should now be on pasteur)

Run the checker first:

```bash
bash src/check_setup.sh
```

It verifies presence AND that `controller.py` is the **merged** version. Context:
on 2026-07-21 the client files were briefly overwritten during a peacock→pasteur
copy; the merged `controller.py` keeps **both** pasteur's camera fixes
(`load_camera_map`, 480×640 resize, 2 Hz publish) **and** the new safety/bias
layer. The checker fails loudly if you have a half-version.

| file | role | origin |
|---|---|---|
| `rollout/controller.py` | **merged**: camera fixes + bias + safety + set_bias | merged |
| `rollout/safety.py` | keep-out + per-step cap; rejects → hold | new |
| `robot/cone_e.py` | workspace clamp re-enabled, bounds recalibrated | new |
| `src/set_bias.py` | change the live bias, no restart | new |
| `src/configs/safety.json` | keep-out zones (ships EMPTY) + step cap | new |
| `replay_demo.py` | replay one HDF5 demo through the pipeline | new |
| `src/check_setup.sh` | this verification | new |
| `robot/camera_id.py`, `robot/camera_map.json` | camera index map | pasteur's own |

## Daily use

**0. Verify + start the arm RPC server** (however you normally start `cone_e` /
the robot bringup on pasteur — unchanged by any of this).

**1. Record one demo** (only once per task, or when the setup changes) with your
existing teleop + `--record` path. It writes an HDF5 with
`{left,right}_ee_pos/_ee_quat/_gripper` + `timestamps`.

**2. Replay it:**
```bash
python replay_demo.py path/to/episode.hdf5 --dry-run
# First supervised known-good run (creates per-joint thresholds):
python replay_demo.py path/to/episode.hdf5 --rate 15 \
  --calibrate-torque src/configs/pasteur_lid_torque.json
# Normal run: torque monitoring is mandatory, and the vision profile adds the
# confirmed grip-start checkpoint automatically.
python replay_demo.py path/to/episode.hdf5 --rate 15 \
  --torque-config src/configs/pasteur_lid_torque.json \
  --vision-profile src/configs/pasteur_lid_vision.json
# 's' start, 'e' end, 'q' quit.  Start at --rate 15 for a first run, then 30.
```

Replay defaults to `--arms auto`: an arm with no meaningful pose or gripper
change is omitted from every action and holds its current pose. This matters for
right-arm-only demos, whose recorded left-arm home pose may sit outside the
active-task workspace. Use `--arms left|right|both` only to override detection
deliberately. Before connecting to the robot, `--dry-run` validates quaternions,
workspace bounds, and consecutive steps. A live run also aborts if the first
target is more than 40 mm or 15 degrees from the post-home pose.

For supervised contact alignment, pause just before the grasp and adjust in
small increments while the arm holds its pose:

```bash
python replay_demo.py demo.hdf5 --rate 15 --checkpoint-frame 70
python src/replay_checkpoint.py --status
python src/replay_checkpoint.py --bias 0 0 -0.005  # reapply frame + save new images
python src/replay_checkpoint.py --resume           # or --abort
```

Checkpoint images and marker/transparent-edge overlays are written under
`/tmp/pasteur_replay_checkpoints` by default. Bias size is not arbitrarily
clamped; live replay therefore requires a calibrated torque watchdog.

**3. Adjust when the object is off** — from another shell, live, no restart:
```bash
python src/set_bias.py                 # show current bias + safety rejection count
python src/set_bias.py --z -0.02       # 2 cm deeper
python src/set_bias.py --y  0.03       # 3 cm in +y
python src/set_bias.py --reset
```
A live change resets the safety step reference, so the trajectory jump it causes
won't be falsely rejected. Bias values must be finite but have no magnitude cap.

**4. Vision-in-the-loop.** The right-camera profile first isolates the blue
fiducial (H=106..115, excluding the teal gripper), translates the confirmed
teacher ellipse with that marker, crops the narrow expected edge band,
contrast-stretches and enlarges it 4×, then fits only nearby edge pixels. The
checkpoint response includes the raw and overlay image paths; display both to
the operator before resume. An uncertain or missing edge is a stop, not a guess.

For the AprilTag path, frames 0–60 smoothly introduce the object transform,
frames 60–140 retain it through grasp and initial transport, and frames 140–190
smoothly return to the teacher trajectory so the fixed destination is not
shifted. Frame 81 numerically probes X/Y/yaw and converges the wrist tag to the
stored successful-grasp corners before closing. Each correction is capped at
2 mm / 2 degrees and failure to converge prevents resume.

## Regression check — DO THIS FIRST, before any real task

The live path now includes calibrated torque monitoring and marker-anchored
checkpoint inspection. Confirm they do not reject a known-good task. Full version in
`outputs/lab/act/horizon/REGRESSION_CHECK.md`; the replay-only short form:

1. `bash src/check_setup.sh` → all pass.
2. Replay a known-good demo at `--rate 15`, **bias 0**. Watch the log for:
   - `[Workspace] Position clamped` → should NOT appear in normal motion.
   - `[safety] REJECTED` → should NOT appear. If `step NNmm > 40mm` fires during
     fast transport, raise `max_step_m` to 0.06 in `src/configs/safety.json`.
3. Confirm `python src/set_bias.py` answers (proves the bias thread is up).
4. Apply the task's known-good bias (e.g. petri2bench −0.025 in z) and confirm
   the arm reaches the same depth it used to.

**Abort and revert if:** the arm drifts during replay (bias applied per-step
somewhere it shouldn't be), clamp fires in normal motion, or safety rejects
repeatedly. Revert client files with `git checkout -- <file>`; `rollout/safety.py`
is inert if unused.

## TODO (in priority order)

1. **Calibrate the implemented torque watchdog** — run a supervised known-good
   replay with `--calibrate-torque`, review the generated per-joint thresholds,
   then use that JSON with `--torque-config` for normal live replay.
2. **Confirm the marker/edge profile** — checkpoint inspection uses the blue
   fiducial to register a narrow ROI, enlarges it 4×, and overlays the expected
   and detected transparent-lid edges for operator confirmation.
3. **Populate keep-out zones** in `src/configs/safety.json` once real forbidden
   volumes are measured (use `robot/test_boundaries.py` EXPLORE to read live EE
   positions). Ships empty because a wrong zone is worse than none.
4. Ablation later: same task via `replay_demo.py` vs the ACT/pi0.5 policy, to
   quantify what closed-loop buys over one-demo-plus-bias.

## Gotchas

- `replay_demo.py` still opens the (unused) ZMQ action sockets via
  PolicyController — harmless, no server needed, `--host` is a dummy.
- Recorded quaternions are **wxyz** (matches `mink.SE3` and the action wire
  format); `frame_to_action` relies on this.
- The demo's gripper channel is the binary open/close, replayed as-is.
- peacock is not needed for any of this. It stays the training / policy-serving
  box; ignore it for the replay workflow.
