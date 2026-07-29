---
name: reconstruct-3d-scene
description: Build, inspect, or repair a semantic 3D scene and MuJoCo model from RGB-D captures using a SAM-first workflow. Use when Codex is asked to create a 3D space, reconstruct a bench or room, segment major objects before modelling, convert Record3D/depth data into completed objects, generate ESDF or MuJoCo geometry, or diagnose scene alignment and collision-readiness.
---

# Reconstruct a 3D Scene

Use the deterministic repository pipeline. Do not manually invent object
coordinates before inspecting the SAM overlay and reconstruction report.

## Workflow

1. Read [references/pipeline-contract.md](references/pipeline-contract.md).
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
