# RGB-D physical gripper levelling

The dish transport pipeline must not equate a level production EE frame with
a level physical blue jaw.  The physical attachment currently has a visible
offset that production FK does not encode.

`src/run_rgbd_gripper_level.py` measures the elongated blue jaw in the raw
head RGB-D frame.  Record3D gravity and measured horizontal support normals
are checked independently.  No accepted camera-to-robot transform is needed.
The current blue-curve PCA feature is diagnostic only: automatic correction is
disabled in the profile until the flat grey contact-pad feature replaces it.

The execution path uses production FK only to choose a joint direction in the
nullspace of XYZ translation and cross-jaw roll.  It then performs a stopped
1.2-degree probe, returns to the exact starting branch, verifies the signed
RGB-D response and repeatability, and only then applies at most 1.5 degrees of
empirically scaled correction.  Each stopped measurement is the median of a
15-frame depth burst and fails closed when its median absolute deviation is
above 0.6 degrees.  A correction that does not improve the physical angle is
rolled back.  There is no object motion or closure.

Every joint waypoint republishes its endpoint at 30 Hz and requires four
consecutive measured samples within 0.5 degrees before any RGB-D capture.
This prevents controller lag from being mistaken for a stopped probe or a
failed return.
The endpoint loop can integrate steady directional error into at most three
degrees of command bias, then latches the measured stopped state.  This is for
controller deadband/gravity rejection and does not relax the measured endpoint
gate.

Observation only:

```bash
python src/run_rgbd_gripper_level.py \
  --output-dir data/runs/pasteur/rgbd_gripper_level/observe
```

Explicitly authorized probe and correction:

```bash
python src/run_rgbd_gripper_level.py \
  --output-dir data/runs/pasteur/rgbd_gripper_level/live \
  --execute
```
