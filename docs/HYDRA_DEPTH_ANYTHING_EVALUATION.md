# Hydra / depth-anything.cpp evaluation for Pasteur

Date: 2026-08-04

## Decision

Both projects are useful, but neither belongs in the shortest path for the
first physical lid-grasp retry.

- Use Hydra to estimate the fixed transform between the head RGB-D camera and
  the Piper base/MuJoCo frame after the camera stream is reliable.
- Use depth-anything.cpp as a confidence-gated depth and multi-view supplement
  for the RGB-only wrist cameras and for offline scene completion.
- Keep the current fast lid-grasp path independent of both. It already uses a
  fixed calibrated MuJoCo route, jaw-level constraints, a lateral-motion veto,
  one continuous descent, one close, and a vertical verification lift.

## Hydra

Source: Martin Huber et al., “Hydra: Marker-Free RGB-D Hand-Eye
Calibration,” arXiv:2504.20584v1, 2025.

Hydra aligns forward-kinematic robot meshes from several measured joint
configurations with SAM 2-segmented RGB-D robot point clouds. It initializes
the camera-to-base transform by centroid alignment and Kabsch-Umeyama, then
refines it with robust point-to-plane ICP on SE(3), using Huber IRLS weights.

The paper reports approximately 90% successful calibration from three robot
configurations, about 0.8 seconds of registration time for nine
configurations, and roughly 5 mm quasi-task-space error. Six varied
configurations are the preferred Pasteur default because the reported success
rate generally saturates by six.

### Pasteur integration

1. Capture six collision-checked right/left configurations with synchronized
   head RGB, registered metric depth, and measured joint state.
2. Render the exact production Piper CAD at each measured configuration.
3. Segment only robot links. Erode unreliable mask boundaries and reject
   pixels with invalid or low-confidence depth.
4. Fit one proper rigid transform with determinant +1. Reflections are never
   admissible; this explicitly prevents the previous left/right inversion.
5. Validate on held-out homing images by mesh reprojection, support-plane
   height, arm identity, and robot/incubator/table nonpenetration.
6. Version the accepted transform with camera serials, image hashes, capture
   time, joint-state hashes, residual statistics, and a camera-moved invalidation
   flag.

Hydra calibrates coordinates; it does not detect the lid, produce a grasp, or
prove collision safety. Its reported image-plane-derived task metric does not
fully measure camera-axis error, so the transform must also be checked against
metric support geometry.

## depth-anything.cpp

Source: localai-org/depth-anything.cpp, a C++17/ggml implementation of Depth
Anything 3 and Depth Anything V2, reviewed at commit
`2028b47ac75a8659c6a9aa617baf09be193eb55f`.

The implementation can emit depth, confidence, predicted camera pose,
intrinsics, point clouds, GLB and COLMAP data. It supports single-image and
multi-view inference, quantized GGUF models, and CPU/CUDA/Metal/Vulkan
backends.

### Pasteur integration

1. First run it only on saved wrist-camera images and videos.
2. Compare its prediction with synchronized Record3D metric depth on opaque
   surfaces: robot links, both platforms, incubator, microscope, and bench.
3. Measure near-workspace scale error, plane residual, temporal jitter,
   discontinuity around jaws, and held-out multi-view consistency.
4. Align relative multi-view output to metric scale using several known Piper
   link dimensions and the observed support plane. Do not infer scale from one
   object or one frame.
5. Fuse only high-confidence, multi-view-consistent points. Low-confidence and
   unseen space stays unknown and therefore occupied for planning.
6. Feed the accepted supplement into semantic scene completion and ESDF
   generation, never directly into a low-level arm command.

The upstream benchmark quotes about 320–400 ms per image on a Ryzen 9
9950X3D CPU and about 47 ms on an NVIDIA GB10. These are reference-machine
numbers, not Pasteur guarantees. The current host reports an NVIDIA
driver/library mismatch, so CUDA deployment requires repair or execution on a
working GPU node.

The multi-view path is coherent but presently relative-scale; the nested
metric model's metric-scale path is primarily single-image. Predicted camera
pose also does not replace robot hand-eye calibration. Hydra (or another
robot-grounded fit) remains responsible for the camera-to-Piper transform.

## Transparent lid policy

Neither learned monocular depth nor consumer LiDAR is authoritative on the
transparent lid. Represent the target as a known thin circular object whose
pose is estimated from the visible rim/blue mark, the support plane, and
multi-view evidence. Do not put a predicted transparent surface into a hard
contact or clearance decision without an independent geometric check.

For grasp execution, retain these independent signals:

- calibrated MuJoCo collision audit for the gross route;
- production FK and measured EE pose for jaw level;
- right wrist image for lid-between-jaws and lateral-motion checks;
- measured gripper aperture for non-empty closure;
- pressure/contact evidence for the final physical stop.

## Proposed later ablation

Evaluate the same held-out captures in four modes:

1. Record3D metric depth only;
2. depth-anything.cpp only;
3. depth-anything.cpp plus semantic masks;
4. confidence-gated fusion of Record3D, depth-anything.cpp, known CAD, and
   multi-view consistency.

Promote mode 4 only if it improves opaque-object completeness without
increasing support-plane error, robot-mesh reprojection error, or false free
space around the grippers. Until then it is visualization/advisory data only.

## One-time lab capture checklist

The following data should be collected while the operator is next in the lab.
It is intentionally separate from the first physical lid-grasp retry and does
not require autonomous calibration motion.

### Before capture

- Install diffuse fixed lighting that illuminates the bench without a hard
  reflection across the transparent lid.
- Enable Developer Mode on all three camera phones, complete the required
  reboot/confirmation, disable Auto-Lock, connect power, reopen Record3D, and
  enable USB Streaming.
- Keep mirroring disabled and keep the head camera mechanically fixed for the
  entire calibration session.
- Start one persistent receiver and leave it connected. Do not poll by opening
  and closing a new Record3D session for each snapshot.
- Capture immutable camera UDIDs and app/version/settings in the manifest.

### Dataset A: fixed-head robot calibration (about five minutes)

Use one continuous head Record3D connection. The capture script sends no robot
commands; the operator moves the robot through the existing teleoperation path
and presses Enter only after it is fully stopped.

```bash
python src/capture_record3d_multiview.py \
  --operator-action move-robot --robot-state \
  --condition hydra_depth_anything_20260805 \
  --frames-per-view 9 \
  --view home_baseline \
  --view physical_left_excitation \
  --view physical_right_excitation \
  --view both_spread \
  --view wrist_orientation_excitation \
  --view holdout
```

The poses must be visibly different, collision-free, and jointly show as much
of both Piper meshes as practical. The last pose is never used to fit the
transform; it is the holdout. Each view stores lossless RGB, metric depth,
confidence, intrinsics, Record3D pose, source hashes, and read-only 12-joint
state before and after its burst. The burst fails if the phone or robot moves.

### Dataset B: RGB-only wrist multi-view

Record the normal lid-grasp trials with the existing teleoperation recorder so
the head, left wrist, right wrist, joint state, EE pose, and gripper aperture
are retained together. Also record one short observation-only episode in which
one arm at a time is manually moved through 8–12 stopped or very slow,
overlapping views of the bench. Include small azimuth and elevation changes;
do not make fast circles. The second arm stays still.

The future Depth Anything importer should extract sharp stopped frames from
the wrist MP4 files, retain the corresponding HDF5 joint/EE state, and record
that wrist depth is learned rather than measured. These images are useful for
scene completion and a relative multi-view model, but never become collision
authority without metric/CAD validation.

### Minimum useful result

Do not spend grasp time repeating a marginal optional view. The minimum useful
future dataset is:

- one fixed, lit head camera identity;
- home plus four varied stopped fit poses and one stopped holdout;
- nine synchronized head RGB-D frames per pose;
- complete before/after 12-joint snapshots;
- one recorded physical grasp attempt with all three RGB streams;
- one right-wrist and one left-wrist overlapping scene sweep;
- a visible known-scale Piper link, support plane, and 90 mm lid in the scene.

If only one extra collection can be completed, Dataset A has priority because
it enables a robot-grounded camera transform and also supplies metric
ground-truth frames for evaluating learned depth.

## References

- https://arxiv.org/abs/2504.20584
- https://github.com/localai-org/depth-anything.cpp
- https://github.com/bytedance-seed/depth-anything-3
