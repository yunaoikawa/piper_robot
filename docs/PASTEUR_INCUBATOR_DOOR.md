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
