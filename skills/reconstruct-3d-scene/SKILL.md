---
name: reconstruct-3d-scene
description: Build, inspect, or repair a semantic 3D scene and MuJoCo model from RGB-D captures using a SAM-first workflow. Use when Codex is asked to create a 3D space, reconstruct a bench or room, segment major objects before modelling, convert Record3D/depth data into completed objects, generate ESDF or MuJoCo geometry, or diagnose scene alignment and collision-readiness.
---

# Reconstruct a 3D Scene

Use the deterministic repository pipeline. Do not manually invent object
coordinates before inspecting the SAM overlay and reconstruction report.

## Workflow

1. Read [references/pipeline-contract.md](references/pipeline-contract.md).
   For end-to-end unattended replay, also follow
   [../../docs/SEMANTIC_SCENE_AUTOMATION.md](../../docs/SEMANTIC_SCENE_AUTOMATION.md).
2. Identify a synchronized RGB image and organized, levelled RGB-D mesh.
3. Select a scene profile and inspect its object catalog.
4. Run `src/build_semantic_scene.py`. Prefer live SAM3; use `--mask` only for
   already accepted masks or offline replay.
5. Inspect `sam_overlay.png`, `scene.json`, `index.html`, and `mujoco.html`.
6. If `operator_confirmation_required` is present, start
   `src/daily_scene_ui.py`, let the operator correct labels, then rerun with
   `--resume-confirmed`.
7. Treat `display_ready`, `collision_ready`, and `motion_ready` as separate
   authority levels. Never infer motion authority from a visually plausible
   model.
8. For explicitly authorized robot-assisted multiview acquisition, follow
   [../../docs/ACTIVE_MULTIVIEW_SCENE_CAPTURE.md](../../docs/ACTIVE_MULTIVIEW_SCENE_CAPTURE.md)
   and keep capture authority separate from reconstruction readiness.

## Existing-data offline replay

When the request is to rebuild the current Pasteur scene and reproduce a
recorded grasp while no new images or hardware motion are allowed, use the
script-only route in
[../../docs/OFFLINE_SCENE_REPLAY.md](../../docs/OFFLINE_SCENE_REPLAY.md):

```bash
MUJOCO_GL=egl \
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_pasteur_offline_replay.py \
  --config src/configs/pasteur_offline_replay_20260730.json \
  --output-dir OUTPUT
```

Do not manually transfer target pixels or joint values. The pipeline must pin
physical-right to semantic `right/` across sensing, carving, and replay, fit
the latest wrist RGB-D target, distinguish
the successful target episode from the latest post-drop episode, pass the
object-radius geometry gate, preserve exact stopped keyframes, and reconstruct
only the unrecorded intervals. A clear moving-arm path does not imply global
scene or hardware motion authority.

For the reviewed Pasteur scene plus a newer fixed-head dish/lid observation,
use the current-scene profile:

```bash
MUJOCO_GL=egl python src/run_pasteur_offline_replay.py \
  --config src/configs/pasteur_current_scene_20260731.json \
  --output-dir OUTPUT
```

This runs `src/update_current_semantic_objects.py` after the historical wrist
target fit and before trajectory reconstruction. It must preserve the accepted
static model, semantic physical-left/right identity, pinned NYU grippers, and
jaw-aligned semantic home. It updates only configured movable object bodies.
Unlabelled same-prompt SAM instances are assigned by normalized motion from the
accepted model, never image-left/right; ambiguous assignments fail closed.
Accepted-mask replay is allowed for exact regression. For a new capture, omit
accepted masks and provide `--sam-endpoint`.

## Tagless multiview and robot calibration

For a completed `piper_robot.multiview_semantic_scene/v1` report, run semantic
completion directly.  This route discovers the bench and raised platforms from
horizontal measured polygons, retains their occupied-cell silhouettes, fits
catalog primitives to SAM-labelled object points, and includes exact Piper CAD
with the configured NYU grippers:

```bash
MUJOCO_GL=egl python src/build_semantic_scene.py \
  --multiview-report OUTPUT/multiview_report.json \
  --profile src/configs/pasteur_semantic_scene.json \
  --output-dir COMPLETED_OUTPUT \
  --daily-scene COMPLETED_OUTPUT/daily_scene.json
```

For a capture-to-MuJoCo run with logs, hashes, validation, and optimization
metrics, prefer the Codex-free production entrypoint:

```bash
MUJOCO_GL=egl python src/run_semantic_scene_pipeline.py \
  --capture CAPTURE_DIR \
  --profile src/configs/pasteur_semantic_scene.json \
  --output-dir OUTPUT_DIR
```

Without an accepted camera-to-robot transform this is display-only.  Do not
promote the visually fitted robot bases to collision authority.

To obtain the transform without AprilTags, keep the head camera fixed and let
the operator teleoperate at least five distinct, fully stopped arm poses.  The
capture process reads qpos before and after each RGB-D burst but sends no robot
commands:

```bash
python src/capture_record3d_multiview.py \
  --operator-action move-robot --robot-state \
  --view baseline --view left_excitation --view right_excitation \
  --view both --view holdout
```

Then fit exact pose-specific CAD to the SAM robot masks:

```bash
python src/calibrate_head_robot_from_cad.py \
  --capture CAPTURE_DIR \
  --profile src/configs/pasteur_semantic_scene.json \
  --output CALIBRATION_JSON
```

The first poses are fit data and the last pose is a holdout.  The calibrator
also removes the eroded mask intersection that remains fixed across all poses;
this suppresses static clutter such as a microscope strut mistakenly labelled
as arm.  Accept only reports whose `accepted` field is true.  The fixed gates
cover depth residuals, union and per-arm silhouette overlap, holdout accuracy,
and independent-pose transform repeatability.

Require the production MJCF through `robot_calibration.model`; never fall back
to a scene approximation for camera calibration. Fit independently mounted
arms with separate base transforms. Associate SAM instances to physical arms
through temporal instance continuity plus arm-specific qpos excitation, never
through view names or image-left/image-right rules. Keep its historically
crossed production namespace (`physical left -> right_arm_*`, `physical right
-> left_arm_*`) separate from the semantic planning namespace (`left ->
left/`, `right -> right/`). Preserve RGB, depth, masks, and
intrinsics in one sensor coordinate system. Any gravity-up or phone-friendly
rotation is display-only.

Rebuild the multiview report after synchronized capture so it contains
`T_level_first_camera` and per-view qpos provenance, then pass the accepted
report to semantic completion with `--calibration-report`.  Collision
readiness additionally requires operator-confirmed objects, successful MuJoCo
compilation, pinned NYU grippers, and no robot/environment penetration above
the configured tolerance.  Motion readiness remains a separate downstream
planning decision.

### Depth ordering for touching SAM regions

When a foreground arm touches a background microscope in the RGB image, do
not use 2D connected components to decide their metric relationship. If the
head camera was fixed while the arms changed pose, follow
[../../docs/DEPTH_AWARE_SCENE_ALIGNMENT.md](../../docs/DEPTH_AWARE_SCENE_ALIGNMENT.md):
build a temporal far-depth envelope, split SAM candidates at depth
discontinuities, retain only components that move in front of the envelope,
and infer fixed base components from persistent 3D voxels. A persistent SAM
robot component is not automatically a base: reject it when its 3D center
lies inside a completed non-robot semantic volume such as a microscope. If
only one base remains observable, update only that unambiguous base and retain
the other reviewed pose. Never pull an unseen base toward a leaked static
component. Validate on a withheld view. Preserve the reviewed Z, yaw,
gripper, and home keyframe. This can refine display alignment, but motion
still requires explicit home-pose provenance and a zero-contact MuJoCo audit.

## Required behavior

- Run SAM before semantic object completion.
- Preserve the distinction between measured surfaces and inferred geometry.
- Use synchronized joint state and exact MJCF/CAD for articulated robots.
- Pin the configured end-effector variant. Fail generation if required
  end-effector geoms disappear or forbidden stock terminal links return.
- Use support, contact, gravity, dimension, and non-penetration constraints.
- Check robot/environment contacts at the configured initial/home keyframe.
  Any penetration above tolerance keeps collision and motion readiness false.
- Preserve measured support silhouettes and holes; never turn a partially
  observed support or a concave multipart object into one solid AABB.
- Do not use PCA alone to place an opaque box template when semantic ESDF data
  exists.  Optimize its gravity-aligned pose against the observed semantic
  surface and volume while strongly penalizing candidate volume in known free
  space; leave unknown space unpenalized and record before/after loss terms.
- Keep unknown or low-confidence objects visible and pending; do not silently
  force them into a known class.
- Store prompts, masks, model identity, scores, timestamps, depth quality, and
  completion sources in the scene manifest.
- Keep object-specific dimensions and aliases in the catalog, not Python.
- Keep image-resolution thresholds dimensionless or derived from image shape.
- Do not move robot hardware as part of reconstruction or validation. An
  explicitly authorized active-capture stage is separate: use only a
  previously validated collision-free camera trajectory, retain pressure
  stopping, and return new captures to this observation-only pipeline.

## Commands

Build with SAM3:

```bash
MUJOCO_GL=egl python src/build_semantic_scene.py \
  --capture CAPTURE_DIR \
  --rgb RGB_IMAGE \
  --mesh LEVELLED_MESH_NPZ \
  --profile src/configs/semantic_scene_default.json \
  --calibration-report CALIBRATION_REPORT_JSON \
  --output-dir OUTPUT_DIR \
  --daily-scene OUTPUT_DIR/daily_scene.json
```

Replay accepted masks by adding one or more
`--mask semantic_name=/absolute/mask.png` arguments.

After phone confirmation:

```bash
MUJOCO_GL=egl python src/build_semantic_scene.py \
  --profile PROFILE \
  --output-dir OUTPUT_DIR \
  --daily-scene OUTPUT_DIR/daily_scene.json \
  --resume-confirmed
```

Stop and report the exact failed gate when input provenance, depth, support,
MuJoCo compilation, or calibration authority is insufficient.
