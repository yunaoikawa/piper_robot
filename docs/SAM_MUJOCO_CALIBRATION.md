# SAM-first MuJoCo calibration

The MuJoCo lab is a nominal model of geometry, articulation, materials, and contact
sites. It is not the source of truth for the current position of the camera, robot,
fixtures, or labware.

## Authority order

1. Synchronized joint state poses the complete robot CAD.
2. Quality-gated SAM instances plus synchronized metric depth pose movable objects.
3. SAM-labelled measured surfaces provide the static polygon and ESDF layers.
4. Nominal MJCF geometry fills unobserved shape only and supplies collision proxies.
5. Unobserved space remains unknown and is not treated as free.

The saved MacBook WetRobo scene at
`robot/piper-mujoco/xml/lab-scene.xml` contributes table/incubator/flask geometry,
materials, synthetic cameras, and gripper contact sites. Its object poses are nominal.
In particular, the incubator was deliberately moved within simulated arm reach and is
not a measurement of the real bench.

## Coordinate frames

`src/reconstruct_scene_esdf.py` stores:

- `T_level_camera` in `scene_esdf.npz`
- vertices in the support-plane-levelled frame
- semantic labels in `scene_mesh_levelled.npz`

An external calibration provider may supply `T_robot_camera` (`robot <- camera`).
The only valid composition for a levelled observation is:

```text
T_robot_level = T_robot_camera @ inverse(T_level_camera)
```

AprilTags are one optional way to obtain `T_robot_camera`; hand-eye calibration or a
validated geometry registration may provide the same transform. SAM does not remove
the need to relate the camera and robot coordinate frames.

Without an accepted transform, the renderer deliberately shows the capture beside the
nominal model. It must not overlay it, update a MuJoCo body, or use it for robot-frame
clearance.

### Display-only rough fixture alignment

Historical data can still be overlaid approximately for human inspection when a
recognizable fixture supplies one point and one table-plane heading. This is a
different mode from calibration and stays ineligible for collision checking, motion,
or a robot-frame `BenchState`.

For example, a levelled capture whose incubator front-face bottom center is measured at
`[0.397, 0.751, 0]` and whose front-to-back direction is level-frame `+Y` can be placed
near the nominal incubator front face as follows:

```json
{
  "schema_version": 1,
  "display_only": true,
  "method": "incubator_front_face_and_support_plane",
  "anchor": {
    "name": "incubator_front_face_bottom_center",
    "source_xyz_m": [0.397, 0.751, 0.0],
    "target_xyz_m": [0.35, 0.0, 0.0]
  },
  "source_heading_xy": [0.0, 1.0],
  "target_heading_xy": [1.0, 0.0]
}
```

Pass that JSON with `--rough-alignment`. The resulting MJCF, image banner, and report
all say `ROUGH_ALIGNED_DISPLAY_ONLY`; `spatially_registered`,
`robot_collision_ready`, and `motion_scene_ready` remain false. A rough alignment and
`--robot-from-camera` are mutually exclusive so heuristic evidence cannot be mistaken
for an external calibration.

## Semantic collision layers

- `robot_surface`: dynamic; excluded from the static ESDF and replaced by synchronized
  robot CAD.
- `lid_surface`: dynamic manipulation target; excluded from the static ESDF and checked
  separately at its calibrated pose.
- `background_surface`: measured static obstacle.
- unknown: collision-conservative, never silently free.

Robot masks win pixels that overlap a target mask because the robot is the visible
foreground. The overlap count remains in the artifact report for diagnosis.

For a transparent petri lid, depth inside the mask often belongs to the bench or
microscope below it. The lid footprint is therefore recovered by intersecting SAM-mask
camera rays with the fitted support plane. A closed-set catalog shape check rejects a
mask whose measured footprint is inconsistent with the selected labware (for example,
the microscope accidentally labelled as a lid). The catalog supplies dimensions and
symmetry; it does not supply the observed position.

The measured polygon mesh is visual-only in MuJoCo. A single-view open surface becomes
an incorrect convex collision hull if added as one mesh geom. Static clearance remains
in the ESDF; known robot and fixture shapes use explicit MuJoCo collision proxies.

## Legacy saved overlays

The current offline artifact recovers masks from lossless blended SAM diagnostics. Its
RGB and depth files are synchronized, but it lacks:

- a camera-to-robot transform,
- synchronized joint positions,
- raw SAM scores, model version, and instance metadata,
- a Record3D confidence image.

It is therefore useful for checking segmentation, support-plane levelling, polygon
quality, and calibration gates, but not for motion or robot-frame collision checking.
New captures should save raw instance masks, score, prompt/model version, capture ID,
RGB/depth/qpos timestamps, camera profile ID, and the accepted transform provenance.

## Portable model note

The MacBook commit originally referenced an untracked MuJoCo Menagerie asset directory.
The checked-in Piper XML uses the repository's own STL meshes so a clean checkout can
compile it. These meshes restore portability; they do not make the nominal arm base or
fixture poses measured truth.
