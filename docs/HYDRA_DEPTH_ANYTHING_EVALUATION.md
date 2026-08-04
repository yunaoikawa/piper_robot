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

## References

- https://arxiv.org/abs/2504.20584
- https://github.com/localai-org/depth-anything.cpp
- https://github.com/bytedance-seed/depth-anything-3
