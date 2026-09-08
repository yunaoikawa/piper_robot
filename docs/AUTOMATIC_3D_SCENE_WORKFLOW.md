# Automatic SAM-first 3D scene workflow

## Purpose

This workflow turns a synchronized RGB-D observation into semantic measured
surfaces, completed objects, a MuJoCo scene, ESDF-ready geometry, and
phone-viewable validation artifacts without requiring Codex to choose
coordinates interactively.

It is observation-only. It does not command robot hardware.

## End-to-end flow

1. Validate the capture provenance and select one sharp, unobstructed RGB-D
   frame. Preserve image, depth, intrinsics, timestamps, camera pose, joint
   state, and hashes.
2. Sweep the configured object catalog with SAM3 text prompts.
3. For each prompt, run a full-frame pass, create a candidate-derived ROI,
   enlarge it, run SAM again, and map the mask back to the original frame.
4. Resolve overlapping masks by confidence. Never use argument order or a
   fixed image region as semantic evidence.
5. Project accepted masks into the organized, levelled RGB-D mesh.
6. Discover horizontal supports from normals and metric height bands.
7. Complete each object according to its catalog policy:
   - `exact_cad`: articulated model and synchronized joint state are
     authoritative.
   - `template`: fit a known closed model under support and dimension
     constraints.
   - `primitive`: infer an oriented box or cylinder from robust metric bounds.
   - `observed_mesh`: preserve an irregular measured surface and generate only
     a conservative hidden-volume proxy.
8. Treat unclaimed interior RGB-D components above supports as unknown object
   candidates.
9. Evaluate confidence, supports, completed-volume intersections, MuJoCo
   compilation, configured end-effector identity, home-pose
   robot/environment penetration, calibration authority, and mobile
   rendering.
10. Continue automatically for known high-confidence objects. Route unknown
    or low-confidence instances to the daily phone UI and resume the same
    revision after correction.

## Why SAM is first

Depth alone separates surfaces but does not identify which disconnected
surfaces belong to one object. A semantic mask first limits geometry fitting
to the correct pixels and prevents a large nearby fixture from dominating
the fit. The measured surface is still retained, so an incorrect semantic
completion remains visibly auditable.

SAM segmentation is not equivalent to guaranteed identity. Prompt score,
mask stability, depth coverage, connectedness, dimension prior, support
plausibility, and multiview agreement are independent evidence. Unknown
regions remain unknown instead of being forced into the closest prompt.

## Commands

Start SAM3 on a GPU node:

```bash
python src/sam3_server.py --endpoint tcp://0.0.0.0:5562
```

Build a scene:

```bash
MUJOCO_GL=egl python src/build_semantic_scene.py \
  --capture /absolute/capture \
  --rgb /absolute/rgb.png \
  --mesh /absolute/scene_mesh_levelled.npz \
  --profile src/configs/semantic_scene_default.json \
  --calibration-report /absolute/calibration_report.json \
  --sam-endpoint tcp://SAM_HOST:5562 \
  --output-dir /absolute/output \
  --daily-scene /absolute/output/daily_scene.json
```

For an offline replay, replace `--sam-endpoint` with repeated accepted masks:

```bash
--mask robot=/absolute/robot.png \
--mask incubator=/absolute/incubator.png
```

Serve the phone confirmation UI:

```bash
python src/daily_scene_ui.py \
  --scene /absolute/output/daily_scene.json \
  --host 0.0.0.0 --port 8765
```

Resume after confirming the exact revision:

```bash
MUJOCO_GL=egl python src/build_semantic_scene.py \
  --profile src/configs/semantic_scene_default.json \
  --output-dir /absolute/output \
  --daily-scene /absolute/output/daily_scene.json \
  --resume-confirmed
```

## Artifact contract

Every output directory contains:

- `scene.json`: canonical provenance, objects, supports, completion methods,
  uncertainty, intersections, and readiness.
- `sam_overlay.png`: semantic masks and confidence over the source image.
- `masks/`: lossless instance masks including unknown candidates.
- `observed/`: SAM-labelled measured RGB-D surfaces.
- `index.html`: inferred and observed layers with mobile toggles.
- `scene.xml`: compiled MuJoCo scene.
- `mujoco.html`: phone-viewable MuJoCo geometry.
- `articulation.mp4`: simulation-only joint sanity check when the model has
  articulated joints and offscreen rendering is available.

The three readiness levels are deliberately separate:

- `display_ready` permits visual inspection only.
- `collision_ready` requires accepted identities, completed geometry without
  disallowed intersections, and successful MuJoCo compilation.
- `motion_ready` additionally requires an accepted camera-to-robot transform.

## Generalization rules

- Put semantic names, aliases, prompts, dimensions, transparency, support
  relationships, and model paths in the object catalog.
- Express image thresholds as fractions or derive them from masks and image
  shape. Do not encode task-specific pixel counts.
- Express geometry in metres and include its coordinate frame.
- Use a support/contact constraint before trusting a completed object's
  vertical position.
- Use exact kinematics for robots. Do not fit individual links independently.
- Distinguish a partial observed surface from a closed collision body.
- Preserve support-mask holes with measured meshes and tiled collision cells;
  an image-space occlusion or robot cut-out must not become solid merely
  because a world-space AABB spans it.
- Pin the requested end-effector in the profile and reject regenerated MJCF
  when required geoms vanish or forbidden stock terminal bodies reappear.
- Make the primary `mujoco.html` use the configured initial pose. Keep a
  synchronized-capture view as a separate artifact when its joint state
  differs.
- Preserve unknown space as collision-relevant unknown space.
- Use the current Pasteur capture as a regression fixture, not as the source
  of universal coordinates.

## Adding an object

1. Add the object to `scene_object_catalog.json`.
2. Supply several literal visual prompts, a completion policy, metric size
   range, support policy, and transparency.
3. Prefer a catalog template over new Python.
4. Add a new completion implementation only for geometry that cannot be
   represented by exact CAD, a template, a box, a cylinder, or an observed
   mesh.
5. Test clear, occluded, low-depth, wrong-prompt, and unsupported cases.

## Failure handling

- No SAM candidate: keep the residual RGB-D component unknown.
- Candidate too large for ROI enlargement: use the recorded full-frame result.
- Poor transparent-object depth: use contour, support-plane ray intersection,
  and catalog dimensions; require confirmation when evidence disagrees.
- Missing support: retain the observation but lower confidence.
- Completed bodies intersect: set collision and motion readiness false.
- The robot penetrates any environment geom at home: set collision and motion
  readiness false and list the body pair and penetration depth.
- MuJoCo compilation fails: keep measured artifacts and report the compiler
  error.
- Camera-to-robot transform is unaccepted: permit display but never motion.
- Daily scene revision changes: invalidate prior collision and motion results.
