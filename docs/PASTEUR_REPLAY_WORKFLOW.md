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
The confirmed lid tag is ID 11. Once stored, calibration always reuses that ID
instead of guessing from apparent tag size; ID 12 on the microscope is fixed.
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
2. **Torque is telemetry, not a general contact stop.** Absolute joint torque
   varies with configuration and acceleration. Stopping target submission also
   leaves the controller holding its last target, which can sustain contact.
   `--torque-config` remains available for characterized tasks, but is optional.
   **Keep a hand on the stop.**

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
                    cone_e.set_*_ee_target()
                       • IK configuration/joint limits
                       • discontinuous joint-branch rejection
```

Both trajectory sources go through the same bias, explicit keep-out, jump,
IK-limit, MuJoCo/ESDF, and torque gates.  The former demo-min/max Cartesian box
was removed because it prevented generalization without representing a
physical obstacle.

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
| `robot/cone_e.py` | arm RPC; no demo-derived Cartesian workspace box | pasteur |
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
# Normal run: torque monitoring is optional, and the vision profile adds the
# confirmed grip-start checkpoint automatically.
python replay_demo.py path/to/episode.hdf5 --rate 15 \
  --vision-profile src/configs/pasteur_lid_vision.json
# 's' start, 'e' end, 'q' quit.  Start at --rate 15 for a first run, then 30.
```

Replay defaults to `--arms auto`: an arm with no meaningful pose or gripper
change is omitted from every action and holds its current pose. This matters for
right-arm-only demos, whose recorded left-arm home pose must not be replayed.
Use `--arms left|right|both` only to override detection deliberately. Before
connecting to the robot, `--dry-run` validates quaternions, MuJoCo IK/joint
limits, ESDF clearance, and consecutive steps. A live run also aborts if the
first target is more than 40 mm or 15 degrees from the post-home pose.

For supervised contact alignment, pause just before the grasp and adjust in
small increments while the arm holds its pose:

```bash
python replay_demo.py demo.hdf5 --rate 15 --checkpoint-frame 70
python src/replay_checkpoint.py --status
python src/replay_checkpoint.py --bias 0 0 -0.005  # reapply frame + save new images
python src/replay_checkpoint.py --resume           # or --abort
```

For a misplaced lid, keep frame 81 active and run the image-Jacobian loop. It
probes X/Y once, then advances by at most 25 mm per observation without homing
between attempts. It stops at the checkpoint; it never resumes the grasp.

```bash
python src/servo_blue_cross.py --port 5561
```

Checkpoint images and marker/transparent-edge overlays are written under
`/tmp/pasteur_replay_checkpoints` by default. Bias size is not arbitrarily
clamped. Torque can be logged or calibrated, but is not required.

For geometry-based generalization, use the demo-relative controller. It maps
the fixed 60 mm workspace tags and blue lid fiducial into robot XY, approaches
the successful grasp pose, then estimates the wrist image Jacobian from
measured EE motion. It never requires an AprilTag on the circular lid:

```bash
python src/run_demo_relative_servo.py --dry-run
# After reviewing the automatically saved overlays:
python src/run_demo_relative_servo.py
```

The first live command pauses before demonstrated contact. Rerun with
`--auto-contact` only after reviewing the head/right overlays under
`/tmp/demo_relative_servo`.

### AprilTag-free 3D servo (preferred experimental path)

The live 3D path does not detect or require AprilTags. Record3D depth is
temporally filtered, a local bench plane is fitted around the live SAM lid
mask, and the grasp ray is intersected with that plane. This avoids treating
the LiDAR return through a transparent lid as the lid's physical height.
Historical/live point clouds can be registered with ICP to detect a shifted
head camera. One fixed 60 mm tag may remain available for exceptional manual
recalibration, but it is not part of normal execution and no tag is placed on
the lid.

The default command is observation-only and saves both head and right-camera
SAM overlays. It sends no arm target:

```bash
python src/run_staged_sam_pregrasp.py
```

Raw and contrast-enhanced SAM inputs are saved even when recognition fails.
Software enhancement cannot recover detail from a nearly black, quantized
Record3D frame: zero SAM candidates is a hard observation failure, so restore
camera exposure/lighting and rerun the observation-only command.

The dry-run report also measures depth geometry at the detected lid. For a
future 3D approach or descent, aim the head camera within 40 degrees of the
bench normal (20--30 degrees is preferable) and make the native LiDAR pixel
footprint no larger than 7 mm at the lid. A closer, more top-down view improves
both metrics. These depth/support-plane metrics are recorded in every one-shot
horizontal probe, but they are advisory for UV-only horizontal calibration and
do not block that single X/Y command. The current saved scene measures about
48 degrees and 5.0 x 6.4 mm: it is rejected for a future 3D approach/descent,
not solely for a signed UV horizontal probe. The stricter 3D/descent gate
remains in the 3D execution path. Repositioning the camera still invalidates
the fixed-view/ICP reference; capture a new reference cloud after the mount is
fixed.

The selected lid and gripper masks and boxes must also remain at least 10 px
inside every head-image boundary. A clipped mask is a hard camera-placement
failure, even if SAM is otherwise confident. Reframe the camera and repeat the
observation-only run; never fit or apply a calibration from a clipped frame.

After reviewing those images, authorize exactly one signed horizontal probe.
Use a fresh operator-issued token and a unique, not-yet-existing output
directory for every invocation:

```bash
TOKEN='fresh-operator-approval-token-for-this-command'
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(python -c \
  'import uuid; print(uuid.uuid4().hex)')"
OUT="/var/tmp/pasteur-horizontal-probes/${RUN_ID}"
test ! -e "${OUT}"
python src/run_staged_sam_pregrasp.py \
  --execute-horizontal \
  --single-probe-axis x \
  --single-probe-m +0.006 \
  --motion-token "${TOKEN}" \
  --output-dir "${OUT}"
```

Choose `x` or `y` and use the sign of `--single-probe-m` to select the
direction. `--execute-horizontal` without `--single-probe-axis` is rejected:
the removed legacy path could issue several calibration and servo motions
under one approval. One token now authorizes one horizontal command, captures
the before/after head and right-camera evidence, writes one immutable
`probe_record.json`, and stops. It cannot request a downward move, command the
left arm, or home either arm. A six-joint positive torque limit is mandatory
and its exact values are recorded with each stationary check.

Build the local model offline from at least three committed, usable-for-fit
probe records whose measured XY motions have rank 2. Keep at least two
additional physical records completely out of the fit; the held-out set must
independently have horizontal rank 2. Across fit plus held-out records, cover
both signs of both X and Y. Every one of those records requires its own fresh
token and unique output directory. A model is motion-eligible only after the
held-out gate accepts it.

Before every later solve, also require the model's current-applicability check
to accept the exact context fingerprint, fixed-camera registration, current EE
position and orientation, and local feature range. Repeat the clip gate on the
current SAM observation. Any rejection means stop and capture a new local
calibration; do not silently reuse an old image-to-motion mapping.

UV is the default feature for horizontal control. Record3D depth is not used
to authorize or calculate the UV horizontal command: its stationary noise is
too large for that role. Synchronized, filtered depth is retained for support
plane, clearance, and eventual descent geometry only. A failed recorded
depth-geometry quality result therefore does not waive or replace the hard
fixed-view ORB, SAM image/mask/ROI margin, freshness, torque, or one-shot claim
gates. Descent remains a separate operator-confirmed stage. Point-cloud
proximity is configured as a warning rather than a hard stop in
`src/configs/pasteur_lid_scene3d.json`.

For offline camera-shift validation, first save a reference cloud, then compare
a later capture. `registration.accepted=false` means stop and recalibrate; it
must not silently fall back to an old image-to-motion bias.

```bash
python src/reconstruct_head_pointcloud.py \
  --rgb reference.png --depth reference_depth.npy \
  --profile src/configs/pasteur_lid_scene3d.json \
  --output-dir /tmp/head_reference
python src/reconstruct_head_pointcloud.py \
  --rgb live.png --depth live_depth.npy \
  --profile src/configs/pasteur_lid_scene3d.json \
  --reference-points /tmp/head_reference/head_scene_points.npy \
  --output-dir /tmp/head_live
```

For offline collision-map development, a saved RGB/depth pair can also produce
a projective TSDF, a conservative ESDF, and a triangle surface mesh without
connecting to either robot or camera:

```bash
python src/reconstruct_scene_esdf.py \
  --rgb saved_head.png --depth saved_head_depth.npy \
  --profile saved_head_profile.json \
  --output-dir /tmp/head_esdf \
  --voxel-size 0.01 --truncation 0.03 \
  --support-normal-file support_plane_normal.npy \
  --support-offset 0.6252082262814739
python -m http.server 8765 --bind 0.0.0.0 --directory /tmp/head_esdf
```

Open `http://<tailscale-ip>:8765/esdf.html` on the phone. The viewer uses a
polygon mesh for surfaces, colors observed free-space clearance red through
green, and shows the unknown-space frontier in purple. Unknown or occluded
space remains `NaN` in `scene_esdf.npz`; collision checking must reject it
rather than treating it as free. A single old frame has no saved Record3D
confidence or pose, so it is suitable for pipeline and viewer validation, not
for authorizing arm motion.

When lossless SAM diagnostic overlays and their synchronized RGB-D frame are
available, recover the semantic masks while building the volume:

```bash
python src/reconstruct_scene_esdf.py \
  --rgb synchronized_rgb.png --depth synchronized_depth.npy \
  --profile saved_head_profile.json --output-dir /tmp/head_semantic \
  --sam-source-rgb sam_input.png \
  --sam-lid-overlay sam_lid_overlay.png \
  --sam-robot-overlay sam_robot_overlay.png
python src/render_semantic_mujoco.py \
  --scene-mesh /tmp/head_semantic/scene_mesh_levelled.npz \
  --output-dir /tmp/head_semantic/mujoco
```

The robot mask is retained as a semantic observation but excluded from the
static ESDF. Complete robot geometry comes from the existing MuJoCo CAD; this
avoids both sparse single-view arm surfaces and double-counting the robot as a
static obstacle. Do not fuse a SAM RGB frame with depth from another time.
The renderer intentionally places the capture beside the CAD whenever the
saved frame lacks synchronized joint state or a camera-to-robot extrinsic.
`semantic_comparison_UNREGISTERED.mjcf` is then a visual diagnostic, not a
clearance model. The measured whole-scene triangle mesh is visual-only because
using one non-convex scene mesh as a MuJoCo collision geom would create a
misleading convex-hull collision volume.

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

## Generic SAM target grasp window

The autonomous path no longer closes from an absolute right-camera pixel
coordinate. `rollout/grasp_window.py` detects the light finger pad beside the
cyan tool body on every frame and expresses the SAM target mask in that local
tool frame. This makes translation and image resolution irrelevant. The
successful demonstration defines both a forward square ("white window") and
target-mask quantiles. Run their labelled offline ablation with:

```bash
python src/run_grasp_window_ablation.py \
  --manifest src/configs/pasteur_grasp_window_ablation.json \
  --output-dir /tmp/pasteur_grasp_window_ablation
```

The current small dataset selects the fail-closed `HYBRID` conjunction. The
window is an insertion test, not proof of contact: an image can match while the
tool is still floating. `src/run_autonomous_sam_lid_grasp.py` therefore closes
only when all of these independent conditions pass:

1. a fresh SAM target mask matches the demonstrated tool-relative window;
2. the light finger pad has reached the target in normalized image distance;
3. the measured end-effector orientation matches the demonstrated grasp;
4. MuJoCo's gripper mesh is within the configured two-sided tolerance of the
   RGB-D support plane;
5. closure stabilizes above the empty-close aperture; and
6. the target follows the two-millimetre verification lift.

When only the support-distance gate remains, the controller descends along the
observed plane normal in at most two-millimetre, MuJoCo-validated steps. A
simultaneous motion stall and torque change before the support plane is treated
as early fingertip contact; the controller lifts four millimetres to release
it and holds. It does not guess a lateral direction until a wrist-image
Jacobian for the current camera mounting has been validated.

The state transitions are in
`rollout/autonomous_grasp_state_machine.py`, so these decisions do not depend
on an interactive Codex session. The task schema calls the lid a generic
thin-planar target; AprilTags are not required at runtime. The present camera
to robot calibration is intentionally unaccepted, so live execution remains
fail-closed until a new multi-pose calibration is explicitly validated.

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

1. **Characterize torque before using it as a stop** — recovery teleop now
   records slew-limited residual and absolute-envelope warnings without stop
   authority. Continue collecting multiple poses and speeds with
   `--calibrate-torque`, especially for a dedicated left-arm envelope. A later
   autonomous stop must combine model residual with tracking stall/contact
   evidence rather than treating this warning alone as collision.
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
