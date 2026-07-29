# Active multiview capture for the lab scene

## Goal

Reduce the ambiguity left by one head RGB-D frame without allowing a
photorealistic reconstruction to become collision authority by accident.
The scene keeps four distinct layers:

1. exact articulated CAD plus synchronized joint state for both Piper arms;
2. SAM-labelled RGB-D observations and temporal instance tracks;
3. a multiview appearance/surface model for inspection and pose refinement;
4. a measured TSDF/mesh and conservative ESDF for collision checking.

The Gaussian layer may refine camera poses and render unseen views. It does
not replace the metric collision mesh or ESDF.

## Capture gate

Active capture is a separate operation from reconstruction. It may run only
after all of the following are true:

- the operator explicitly authorized robot-assisted capture;
- the requested end-effector variant passed its model regression check;
- the proposed scan trajectory is collision-free in the current conservative
  ESDF, including swept-volume checks;
- the initial keyframe has no unexplained robot/environment penetration;
- pressure stopping remains enabled;
- only one camera-bearing arm moves at a time, at the configured scan speed.

If the current scene fails any gate, collect by manually repositioning the
camera or robot, or repair the static model first. Do not use arm motion to
validate a questionable model.

## Recommended acquisition

Use 8–12 stopped views rather than continuous fast circles. Each neighboring
view should retain substantial overlap while changing azimuth or elevation
enough to reveal an occluded surface.

For every stopped view, save:

- lossless RGB and metric depth;
- camera intrinsics and depth scale;
- monotonic RGB/depth timestamps;
- all arm joint positions at the exposure time;
- the camera-to-end-effector transform and its calibration identity;
- pressure-stop state;
- source hashes.

Capture the fixed head RGB-D view first. Then move the left and right hand
cameras one at a time through prevalidated viewpoints. Render the exact robot
CAD into each camera and combine that self-mask with SAM so moving robot pixels
do not enter the static lab reconstruction.

## Registration

Optimize one shared lab transform and the hand-eye residuals jointly; never
fit individual robot links. The objective combines:

- SAM robot surface to forward-kinematic CAD distance;
- static-scene RGB-D reprojection and point-to-plane residuals;
- broad support-plane consistency;
- incubator/template edge alignment;
- camera-pose smoothness and hand-eye priors;
- non-penetration penalties.

Keep vertical base height locked to the measured mount/support relation unless
independent multiview depth provides enough evidence to revise it. Reject a
lower visual residual if it worsens the home-pose penetration gate.

## Reconstruction choices

- Classic 3D Gaussian Splatting is useful for fast, photorealistic novel-view
  inspection, but its anisotropic volumes are not a collision mesh.
- SplaTAM is a relevant RGB-D baseline because it jointly tracks an unposed
  RGB-D camera and builds a Gaussian map.
- 2D Gaussian Splatting is preferable when extracting a consistent surface is
  important.
- Triangle-based splatting is promising for a directly editable mesh, but is
  newer and should remain an experimental backend until it passes the saved
  Pasteur regression captures.

The production path should fuse RGB-D into a projective TSDF, extract a
polygon mesh, simplify it without closing observed holes, then build the ESDF.
A Gaussian or triangle-splat model is an additional visual/registration layer.

## Next-best-view loop

1. Fuse all accepted frames and update per-object coverage.
2. Compute unknown frontiers and low-confidence SAM/depth regions.
3. Sample camera poses reachable by the camera-bearing arm.
4. Reject poses and swept paths that enter the conservative ESDF.
5. Score the remaining poses by expected newly visible surface, viewing angle,
   depth range, and motion cost.
6. Execute the best validated pose slowly, stop, capture, and re-register.
7. Stop when coverage gain is small or any calibration/contact gate fails.

This loop generalizes to new object layouts because its objective is metric
coverage and uncertainty reduction, not a saved pixel target.

## Primary references

- Kerbl et al., [3D Gaussian Splatting for Real-Time Radiance Field
  Rendering](https://arxiv.org/abs/2308.04079)
- Keetha et al., [SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D
  SLAM](https://arxiv.org/abs/2312.02126)
- Huang et al., [2D Gaussian Splatting for Geometrically Accurate Radiance
  Fields](https://arxiv.org/abs/2403.17888)
- Fry et al., [Triangle Splatting SLAM](https://arxiv.org/abs/2605.31419)
