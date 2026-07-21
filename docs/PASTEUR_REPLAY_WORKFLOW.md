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
2. **No collision reaction.** The arms *do* expose torque
   (`piperlib.JointState` has `.torque`, `.vel`), but **nothing reads it yet**.
   Every safety layer here is preventive/geometric. **Keep a hand on the stop.**
   Wiring a torque watchdog is the top TODO below.

## Architecture

```
 trajectory source ─┬─ replay_demo.py     (recorded HDF5)        <- use this
                    └─ cloud_inference_control_collect_v2.py (ZMQ policy server)
                              │
                     PolicyController.apply_action()   (rollout/controller.py)
                              │
                    xyz_bias  (per-arm, robot frame, live-settable, ±0.06 m cap)
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
python replay_demo.py path/to/episode.hdf5 --safety-config src/configs/safety.json --rate 15
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

Each adjustment is limited to 10 mm from the previous bias. Checkpoint images
are written under `/tmp/pasteur_replay_checkpoints` by default.

**3. Adjust when the object is off** — from another shell, live, no restart:
```bash
python src/set_bias.py                 # show current bias + safety rejection count
python src/set_bias.py --z -0.02       # 2 cm deeper
python src/set_bias.py --y  0.03       # 3 cm in +y
python src/set_bias.py --reset
```
Bias is clamped to ±0.06 m. A live change also resets the safety step reference,
so the trajectory jump it causes won't be falsely rejected.

**4. Vision-in-the-loop (optional, the "Claude/Codex adjusts" part).** Grab the
head-camera frame, **crop to the object ROI + contrast-stretch + upscale** (see
the measurement note above — raw frames don't work), show it to the model, ask
for a lateral offset RELATIVE to the demo, translate that to a bias, and call
`set_bias`. Do this **once per attempt while the arm is stationary at the
start** — never inside the motion (a VLM takes 1–3 s; the loop is 30 Hz), and
supervise it (no collision sensing yet).

## Regression check — DO THIS FIRST, before any real task

Three things changed in the live control path (clamp re-enabled + recalibrated,
bias moved off the server, safety layer can now reject). Confirm they didn't
break a known-good task. Full version in
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

1. **Torque watchdog** — the one real safety upgrade available. Read
   `piper.get_joint_state().torque` (add a `ConeE.get_*_joint_torque()` method;
   `robot/rpc.py` is a generic getattr proxy so the client sees it automatically),
   compare against the demo torque distribution, and stop on a sustained
   exceedance. Calibrate the threshold from a normal replay (record torque while
   replaying a good demo, take the max, add margin — same method as the 40 mm cap).
2. **Vision preprocessing helper** — a function that crops the ROI, contrast-
   stretches, and upscales the head frame, so the bias-from-vision step is
   reliable. Without it the model cannot resolve cm-level offsets.
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
