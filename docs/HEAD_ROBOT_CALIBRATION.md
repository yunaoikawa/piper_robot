# Fixed head RGB-D to robot calibration

This workflow connects a rigidly mounted Record3D LiDAR camera, exact
bimanual Piper CAD, and the saved semantic lab scene without AprilTags.  The
camera remains fixed throughout.  Camera capture and robot commands run in
separate processes so the capture manifest truthfully records
`commands_sent=false`.

## 1. Preflight without motion

Only one process may own the arms.  Stop any rollout or teleoperation process,
then start the RPC server attached to the measured joints:

```bash
PYTHONPATH=. python -m robot.cone_e --attach-current
```

`--attach-current` is mandatory: it holds the measured pose and does not home.
Check the keyboard controller without sending commands:

```bash
PYTHONPATH=. python src/calibration_keyboard_jog.py \
  --check-only \
  --allow-symmetric-left-torque-fallback
```

The left fallback is explicit because the current torque file contains a
measured right-Piper envelope.  It copies that conservative envelope only for
the mechanically identical left Piper and refuses to start if the stationary
left torque already exceeds it.  Replace it with measured left thresholds
when those are available.

## 2. Collect five stopped poses

Keep Record3D USB streaming enabled.  In the capture terminal:

```bash
PYTHONPATH=. python src/capture_record3d_multiview.py \
  --operator-action move-robot --robot-state \
  --condition fixed_head_robot_calibration \
  --view pose_0_baseline \
  --view pose_1_left \
  --view pose_2_right \
  --view pose_3_both \
  --view pose_4_holdout
```

In a second terminal:

```bash
PYTHONPATH=. python src/calibration_keyboard_jog.py \
  --allow-symmetric-left-torque-fallback
```

The jog UI never initializes or homes.  A direction key stages one 5 mm
robot-frame Cartesian move; it sends nothing until Enter confirms it.  The
historical ConeE Cartesian workspace is not mirrored for the left arm, so the
UI refuses any left Cartesian command that would be clamped.  Select a joint
with `[` / `]`, then use `-` / `+` for a slow 0.005 rad joint move instead.
Space holds the measured joint positions of both arms, Esc cancels a staged
move, and `q` holds and exits.  Holding never resends an EE target, because an
EE target outside the historical shared clamp could move instead of hold.

Pose requirements are geometric rather than saved pixel targets:

1. Capture the current baseline.
2. Move the left arm so at least two joints differ by 0.02 rad or more.
3. Move the right arm with the same minimum diversity.
4. Move both arms to a non-overlapping asymmetric view.
5. Make the holdout pose at least 0.03 rad from every fit pose for each arm.

At every prompt, stop completely before pressing Enter in the capture
terminal.  The burst is rejected if any joint changes more than 0.005 rad or
the camera moves more than 3 mm / 1 degree.

## 3. Calibrate and register the saved scene

The production entrypoint runs SAM robot masks, exact CAD fitting, independent
holdout gates, fixed-camera-to-saved-scene registration, and robot-frame
semantic/MuJoCo reconstruction:

```bash
PYTHONPATH=. MUJOCO_GL=egl python \
  src/run_head_robot_calibration_pipeline.py \
  --capture CAPTURE_DIR \
  --multiview-report \
    data/reconstructions/pasteur/record3d_20260730_215729_v3/multiview_report.json \
  --output-dir data/runs/pasteur/fixed_head_calibration \
  --sam-endpoint tcp://127.0.0.1:5562
```

The camera calibration accepts only when:

- train depth median / p90 are at most 10 / 25 mm;
- holdout depth median / p90 are at most 15 / 30 mm;
- robot union IoU is at least 0.70 and each arm IoU at least 0.60;
- independent-pose transform repeatability is within 5 mm / 1 degree;
- both arms and the holdout pass qpos-diversity gates.

The scene registration selects the strongest saved overlapping view using
metric ORB correspondences, removes robot and movable-object masks, refines
one rigid transform against static RGB-D, then validates it against the saved
static semantic volume.  It does not deform links or scene geometry.  The
saved multiview report hash is pinned, so the result cannot be applied to a
different scene.

## 4. Collision and pregrasp gates

The first semantic run creates a pending daily-scene revision.  Confirm the
objects in the phone UI, then rerun the same command with:

```text
--daily-scene OUTPUT/semantic/scene/daily_scene.json
--resume-confirmed
--require-collision-ready
```

Collision readiness requires the accepted registration, synchronized 12-joint
state, confirmed objects, compiled MuJoCo with pinned NYU grippers, and no
robot/environment penetration above tolerance.

The pipeline writes `pregrasp_motion_config.json` and an exact live dry-run
command in `pipeline_report.json`.  Run that command before any physical
pregrasp.  A dry-run uses fresh head SAM/RGB-D and right-wrist RGB to calculate
the lift-horizontal-descent trajectory but sends no arm or gripper command.
Physical execution remains a separate gate after the dry-run artifacts and
swept-volume result are accepted.
