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

## Side-view repeatability reference

Once an observer wrist has acquired a user-confirmed side view, save its blue
jaw silhouette and both measured EE poses. The clicked component is converted
to scale-independent axis, elongation, and fill features; absolute pixels and
area are never a horizontal gate. A future run reacquires the view with a
locally measured image Jacobian, then compares the normalized silhouette.

The side-view reference proves that the observation geometry is repeatable. It
does not by itself prove strict horizontal level. Strict authorization still
comes from `assess_jaw_level`, so an operator label of `approximately_level`
cannot accidentally become a production pass condition.

```bash
python src/save_jaw_side_view_reference.py \
  --image /path/to/left_wrist.jpg \
  --target-x 471 --target-y 197 \
  --operator-label approximately_level \
  --output src/configs/pasteur_right_jaw_side_view_reference.json
```

Explicitly authorized probe and correction:

```bash
python src/run_rgbd_gripper_level.py \
  --output-dir data/runs/pasteur/rgbd_gripper_level/live \
  --execute
```
