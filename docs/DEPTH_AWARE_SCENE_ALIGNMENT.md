# Depth-aware scene and robot alignment

This workflow refines the reviewed Pasteur semantic scene without moving
hardware. It exists to prevent a common 2D failure: a foreground Piper link
and a background microscope touch in the image, so a SAM union incorrectly
treats them as one object.

## Contract

- RGB segmentation proposes semantic candidates; it does not determine
  metric ordering.
- A fixed metric AprilTag connects the earlier Record3D reconstruction to the
  stopped current RGB-D capture.
- The current head camera must remain fixed while the arms change pose.
- For every pixel, the farthest valid depth across the stopped views is the
  temporal background envelope.
- A SAM component is retained as robot foreground only when part of the
  depth-connected component is measurably closer than that envelope.
- Stationary 3D voxels present in at least three training views identify the
  two fixed robot bases. A component must also reach the known base support
  plane; persistent upper-link fragments are rejected.
- The fifth view is a holdout. Both fitted base centers must have nearby
  depth-cleaned robot points in that view.
- Base XY may be corrected independently because these are two separately
  mounted robots. Reviewed base Z, upright yaw, NYU gripper geometry, and the
  MuJoCo `home` keyframe are retained.
- The microscope geometry is never moved to improve robot clearance.
- A trajectory is authorized only after the camera registration and robot
  fit pass and MuJoCo reports zero robot/environment contacts at home.
- All outputs are display-only. This pipeline imports no robot client and
  records `commands_sent: false`.

## Command

```bash
PYTHONPATH=. MUJOCO_GL=egl conda run -n robot-test \
  python src/refine_scene_robot_alignment.py \
  --reference-report \
  data/reconstructions/pasteur/record3d_20260730_215729_v3/multiview_report.json \
  --reference-capture \
  data/captures/pasteur/record3d_multiview_import/2026-07-30/20260730T050529.878846Z_head_record3d_exr_d7d3a413 \
  --current-capture \
  data/captures/pasteur/2026-07-30/20260730T065216.967597Z_head_fixed_head_robot_calibration_41c5df92 \
  --robot-mask-dir \
  data/runs/pasteur/sam_robot_industrial_probe/accepted_union \
  --scene-json \
  data/reconstructions/pasteur/record3d_20260730_215729_semantic_mujoco_v1/scene.json \
  --scene-model robot/pasteur-calibrated-scene/scene.mjcf \
  --positioned-robot robot/pasteur-calibrated-scene/positioned_robot.mjcf \
  --output-dir /tmp/pasteur_scene_alignment \
  --baseline-is-home \
  --lid-center-px 647.653164556962 1009.0962025316455 \
  --support-plane-z-m -0.5660460087444851
```

The current-frame lid center is operator-confirmed because the SAM service
was unavailable for this capture. The JSON records this explicitly; it must
not be described as a current-frame SAM result.

## Outputs and gates

- `depth_layer_robot_masks/`: foreground-only robot masks after temporal
  depth separation.
- `depth_layer_robot_montage.png`: the same masks overlaid in green on all
  stopped RGB views, generated automatically for human review.
- `latest_lid_alignment.png`: Tag-bridged lid circle on the current RGB
  frame.
- `positioned_robot.mjcf` and `scene.mjcf`: derived models; the reviewed
  canonical files are not overwritten.
- `alignment_report.json`: Tag repeatability, rejected/accepted depth
  components, independent base translations, holdout errors, collision
  ablation, and the trajectory gate.
- `latest_lid_scene.json`: metric dynamic object input for planning.

Render a phone-readable, stopped audit:

```bash
PYTHONPATH=. MUJOCO_GL=egl conda run -n robot-test \
  python src/render_scene_alignment.py \
  --model /tmp/pasteur_scene_alignment/scene.mjcf \
  --object-scene /tmp/pasteur_scene_alignment/latest_lid_scene.json \
  --alignment-report /tmp/pasteur_scene_alignment/alignment_report.json \
  --output /tmp/pasteur_scene_alignment/home_scene_display_only.mp4 \
  --report /tmp/pasteur_scene_alignment/render_report.json
```

If contacts remain, the output is still useful as an alignment diagnostic,
but it is not a trajectory. Do not remove the collision gate merely because
the RGB silhouettes overlap correctly. First determine whether the remaining
contact comes from base pose, reviewed home kinematics, or an over-completed
semantic collision volume.
