# Pasteur incubator-door opening

This workflow retargets verified teleoperation demonstrations to the current
incubator pose, closes the right gripper on the recessed door handle, proves
the grasp with a 5 mm pull, and then replays the door-opening motion slowly.
The individual stages are intentionally restartable so that a failed visual or
mechanical check does not require homing or repeating successful motion.

## Perception rule: shadows are not gripper geometry

The black horizontal or claw-shaped regions around the handle can be shadows.
Their position changes with the lab light and is not evidence for jaw pose,
contact, or handle occupancy.  Never compare black regions with a successful
demonstration or use their area as a close trigger.

The current door orientation is estimated from RGB-D planar consensus instead:

1. fixed AprilTag 3 registers the head RGB-D frame to the robot frame;
2. the tag attached to the incubator only seeds a broad region of interest;
3. low-saturation points on the vertical incubator face are fitted with RANSAC;
4. the median plane normal over the bundle supplies the current door yaw;
5. a high yaw spread, too few inliers, or poor plane residual stops retargeting.

The tag is not treated as an obstacle and does not define the door plane.  The
fit uses thousands of depth points, so a moved lamp or a cast shadow does not
rotate the planned gripper.

```bash
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/estimate_incubator_door_plane.py \
  --capture /absolute/path/to/head_rgbd_bundle \
  --output /absolute/path/to/door_plane.json
```

The accepted Pasteur regression bundle on 2026-08-08 contained eight frames.
It produced about 4,100 inliers per frame, a median door-normal yaw of about
-5.4 degrees, 0.40 degree standard deviation, and roughly 2.2--2.8 mm plane
RMS.  These are reference quality figures, not hard-coded target coordinates.

## Demonstration compilation

Only visually verified door-opening recordings are compiled.  The compiler
selects a medoid trajectory, measures contact/proof/release landmarks, and
trains the small rigid-parent feature correction used by the right camera.

```bash
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/compile_incubator_door_demos.py
```

The red EYELA label is a rigid-parent alignment feature, not the handle.  It
may correct small translation after plane-yaw retargeting.  It must not override
depth geometry or serve as grasp-success evidence.

## Restartable execution stages

All live commands require an explicit, unique output directory.  The physical
right arm is selected through `robot/camera_map.json`; do not infer arm identity
from model body prefixes or image left/right.

```bash
PY=/home/admin/miniforge3/envs/robot-test/bin/python
RUN=/absolute/unique/run/directory

# Read-only evidence from head/right/left cameras and robot state.
$PY src/run_incubator_door_demo.py observe --output-dir "$RUN/observe"

# Move to the demo-relative open-jaw preclose after applying measured door yaw.
$PY src/run_incubator_door_demo.py aligned-yaw-preclose \
  --world-yaw-deg CURRENT_MINUS_DEMO_YAW \
  --output-dir "$RUN/preclose"

# Optional small right-camera correction, then open-jaw contact.
$PY src/run_incubator_door_demo.py visual-align-step \
  --aligned-state "$RUN/preclose/aligned_state.json" \
  --output-dir "$RUN/visual"
$PY src/run_incubator_door_demo.py aligned-contact \
  --aligned-state "$RUN/visual/aligned_state.json" \
  --output-dir "$RUN/contact"

# Close once, prove retention with 5 mm pull, verify while stationary, and pull.
$PY src/run_incubator_door_demo.py close-verify \
  --contact-state "$RUN/contact/aligned_contact.json" \
  --output-dir "$RUN/close"
$PY src/run_incubator_door_demo.py proof-pull \
  --contact-state "$RUN/close/contact_state.json" \
  --output-dir "$RUN/proof"
$PY src/run_incubator_door_demo.py reverify-proof \
  --proof-state "$RUN/proof/proof_state.json" \
  --output-dir "$RUN/reverify"
$PY src/run_incubator_door_demo.py open-door \
  --proof-state "$RUN/reverify/proof_state.json" \
  --output-dir "$RUN/open"
```

Paths are rotated and translated from the measured contact state; they are not
replayed at the absolute demonstration pose.  Cartesian targets are retried to
settle controller bias before contact.  Closing is accepted only when aperture
is stable and above the characterized empty-close bound.  The 5 mm proof must
retain sufficient aperture before the full pull begins.  During the full pull,
aperture is checked at sparse trajectory checkpoints rather than adding a slow
30 Hz vision loop.

If the grasp becomes empty, run the recovery stage.  It retreats 15 mm before
opening the jaws; it does not home through the door or continue pulling.

```bash
$PY src/run_incubator_door_demo.py recover-empty-close \
  --output-dir "$RUN/recover"
```

## Regression checks

```bash
/home/admin/miniforge3/envs/robot-test/bin/python -m pytest -q \
  tests/test_incubator_door_demo.py \
  tests/test_incubator_door_visual.py \
  tests/test_incubator_door_plane.py
```

The tests cover demonstration landmarks, SE(3) retargeting, feature alignment,
world-yaw correction, and synthetic vertical-plane recovery.  A future change
must preserve the shadow-independent plane fit and the close/proof/pull gates.

## Closing an open door

Closing is not the reverse of the grasped opening trajectory.  Verified
Peacock data shows that the physical right arm keeps both jaws fully open and
pushes the lower inside edge of the door.  The opening grasp path is about
15 cm too high for this contact and can move the arm without moving the door.

The accepted reference is copied from Peacock as:

```text
/home/yoikawa/src/robot/data/raw/horizon/door_close/
  door_close_20260703_163736.{hdf5,head.mp4,left.mp4,right.mp4}
```

On Pasteur it lives under
`data/reference/pasteur/incubator/incoming/door_close/`.  The loader rejects a
reference if the inactive arm moves or if either jaw closes.  For the current
fixed Pasteur installation, use the demonstrated robot coordinates directly;
do not apply the opening-handle registration rotation.  That rotation moved
the low pusher beside the open panel and produced a collision-free but empty
motion.  The raw 140-frame Cartesian close trajectory completed successfully
on 2026-08-08 with the jaws open.

```bash
$PY src/run_incubator_door_demo.py close-door-demo \
  --output-dir "$RUN/close-door"
```

Success must be confirmed by a fresh image in which the incubator interior is
no longer visible and the outer door face/control panel is visible.  Shadows
remain non-evidence.  Do not add an extra terminal push after visual closure.

## Codexless endpoint workflow

`run_incubator_door_autonomy.py` now owns the normal stage transitions.  Codex
is not part of its perception or control loop.  With no `--execute` flag it
only prints the evidence-gated plan and cannot connect to or move the robot:

```bash
$PY src/run_incubator_door_autonomy.py open
$PY src/run_incubator_door_autonomy.py closed
```

To run against Pasteur, give one unique run directory.  Every motion, image,
decision, subprocess output and failure is appended atomically to
`journal.json`:

```bash
$PY src/run_incubator_door_autonomy.py open --execute \
  --output-dir data/runs/pasteur/incubator_auto_open_001

$PY src/run_incubator_door_autonomy.py closed --execute \
  --output-dir data/runs/pasteur/incubator_auto_close_001
```

The initial and final door states are not inferred from a missing tag or from
black image regions.  The controller:

1. registers a fresh head RGB-D bundle to two verified endpoint observations;
2. learns the moving-panel region from the open-vs-closed metric depth change;
3. intersects the depth change with the low-saturation closed vertical-plane
   inliers, excluding the blue gripper and most arm geometry;
4. compares the live depth independently with both endpoint prototypes;
5. expresses minimum support in reference-tag areas instead of image pixels;
6. reports `unknown` and sends no further command when the two endpoints are
   ambiguous.

Opening estimates the closed face plane, moves to the calibrated open-jaw
preclose, permits only a bounded number of normalized right-image corrections,
closes once, proves the grasp with 5 mm motion, re-verifies it while
stationary, and then runs the demonstrated pull.  An empty or lost grasp first
retreats before reopening.  No 30 Hz vision loop is inserted into the pull.

Closing deliberately selects the separate low, open-jaw Peacock push demo.
It never reverses the opening trajectory and never sends an unverified extra
terminal push.  A final registered RGB-D observation must classify as closed.

The state machine and endpoint classifier live in
`rollout/articulated_appliance.py`; the Pasteur profile only supplies the
incubator-specific demonstrations, endpoint observations and calibrated
tolerances.  To reuse the flow for another hinged or sliding appliance,
record and visually verify both endpoint RGB-D observations, provide a rigid
parent marker plus fixed registration markers, and supply distinct open and
close demonstrations.  Pixel coordinates and shadow colour are not part of
the reusable contract.

Run its offline regressions with:

```bash
$PY -m pytest -q tests/test_articulated_appliance.py \
  tests/test_incubator_door_demo.py \
  tests/test_incubator_door_plane.py \
  tests/test_incubator_door_visual.py \
  tests/test_incubator_door_close.py
```

## Portable appliance frame for another lab

The Pasteur coordinates above are a verified installation profile, not the
portable task definition.  Cross-lab execution uses the completed incubator
box as a canonical semantic frame.  An end-effector sample is compiled as
`T_appliance_ee`; at runtime it becomes
`T_robot_appliance_live @ T_appliance_ee`.  Thus a translated or rotated
incubator moves both the opening preclose and every sample of the low open-jaw
closing path.

AprilTags are optional local trackers.  Their ids and physical locations do
not need to match between labs.  During enrollment, SAM plus RGB-D first fits
the incubator volume and establishes `T_robot_appliance`.  If a tag is useful,
the enrollment stores that lab's measured
`T_appliance_tag = inv(T_robot_appliance) @ T_robot_tag`.  Later observations
recover the appliance as
`T_robot_tag_live @ inv(T_appliance_tag)`.  Neither tag id nor tag pose appears
in the portable trajectory.  Without a tag, acquire another SAM plus RGB-D
scene instead.

The first enrollment step is read-only and never contacts the robot:

```bash
$PY src/enroll_appliance_frame.py \
  --scene /lab/site/scene.json \
  --robot-scene-calibration /lab/site/accepted_robot_scene.json \
  --semantic-name incubator \
  --tag-observation /lab/site/optional_tag_observation.json \
  --tag-id 41 \
  --output /lab/site/incubator_enrollment.json
```

`accepted_robot_scene.json` must independently accept the convention
`p_robot = T_robot_scene @ p_scene`.  A gravity-levelled Record3D scene is not
silently treated as the camera or robot frame.  The semantic volume fit,
confidence, and collision readiness must also pass.  `--inspection-only` may
write diagnostics from an incomplete scene, but its output explicitly has no
motion authority.

Create the bounded registration from the reference demo installation to the
current lab.  Give `--current-tag-observation` only for the fast locally
enrolled tag path; omit it to use the current SAM/RGB-D enrollment directly:

```bash
$PY src/prepare_appliance_registration.py \
  --reference-enrollment /reference/incubator_enrollment.json \
  --current-enrollment /lab/site/incubator_enrollment.json \
  --current-tag-observation /lab/site/tag_now.json \
  --output /lab/site/incubator_registration.json
```

Registration rejects excessive translation, yaw, tilt, and reconstructed size
change.  Pass only the accepted artifact to either motion stage:

```bash
$PY src/run_incubator_door_demo.py \
  --appliance-registration /lab/site/incubator_registration.json \
  --output-dir "$RUN/preclose" aligned-yaw-preclose \
  --aligned-state /reference/aligned_contact.json

$PY src/run_incubator_door_demo.py \
  --appliance-registration /lab/site/incubator_registration.json \
  --output-dir "$RUN/close" close-door-demo
```

The registration already contains the cross-lab appliance yaw.  During the
Codexless opening workflow, the live RGB-D door-plane yaw is used only as a
small residual correction; it is not added a second time.

For Codexless orchestration, set `autonomy.appliance_registration` in the
site-local profile.  If it is absent, the existing Pasteur profile retains its
verified identity registration.  A portable site profile must never copy
Pasteur's identity assumption.

The portable contract and arbitrary-tag-placement regressions are in
`rollout/appliance_frame.py`, `tests/test_appliance_frame.py`, and
`tests/test_prepare_appliance_registration.py`.
