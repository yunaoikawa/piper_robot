# RGB-D physical gripper levelling

The dish transport pipeline must not equate a level production EE frame with
a level physical blue jaw. The removable physical attachment has a measured
pitch offset that the production mesh did not originally encode.

`src/run_rgbd_gripper_level.py` measures the elongated blue jaw in the raw
head RGB-D frame. Record3D gravity and measured horizontal support normals are
checked independently. No accepted camera-to-robot transform is needed. The
blue-curve PCA feature may drive a correction only after a stopped local probe
establishes its sign and gain. It cannot authorize level from one connection.

## Autonomous state machine

The execution path uses production FK only to select a joint direction in the
nullspace of XYZ translation and cross-jaw roll. Every iteration:

1. captures a stopped 15-frame depth burst;
2. performs a fresh 1.2-degree probe;
3. returns to that iteration's exact anchor and verifies joint and visual
   repeatability;
4. applies at most 1.5 degrees of empirically scaled correction;
5. rolls back to the fresh anchor if the physical angle does not improve.

A probe calibration is single-use. If another correction is needed, the code
re-probes at the new pose. This prevents the nonlinear overshoot observed when
a probe measured at a different wrist pose was reused.

Each stopped burst fails closed when its median absolute deviation exceeds
0.6 degrees. A candidate near horizontal must also pass three independent
Record3D connections. Their median must be within 0.25 degrees and their range
within 0.75 degrees. There is no object motion or gripper closure.

Every joint waypoint republishes its endpoint at 30 Hz and requires four
consecutive measured samples within 0.5 degrees before RGB-D capture. The
endpoint loop may integrate steady directional error into at most three
degrees of command bias, then latches the measured stopped state. This handles
controller deadband without relaxing the measured endpoint gate.

The executable prepares MIT mode and tested gains itself and checks right-arm
joint torques against 8 Nm throughout motion. It does not depend on an outer
Codex wrapper having prepared the arm.

Observation only:

```bash
python src/run_rgbd_gripper_level.py \
  --output-dir data/runs/pasteur/rgbd_gripper_level/observe
```

Explicitly start the probe/correction state machine:

```bash
python src/run_rgbd_gripper_level.py \
  --output-dir data/runs/pasteur/rgbd_gripper_level/live \
  --execute
```

## Pasteur calibration saved on 2026-08-10

The confirmed physical-level bursts were `-0.173`, `+0.227`, and `+0.448`
degrees. Their median is `+0.227` degrees and their range is `0.621` degrees,
so the three-burst gate accepted the state. The production mesh reported
`+3.021` degrees at the same measured pose. This gives a physical attachment
pitch offset of `2.793699` degrees, saved as a rotated physical
`approach_axis_ee` in `pasteur_fast_lid_grasp_level.json`.

The exact joint vector and pose are retained only as audit evidence in
`pasteur_right_physical_gripper_level.json`; they must never be blindly
replayed in another pose or installation. Future runs use the calibrated
physical axes for planning and the fresh RGB-D probe loop for correction.

## Side-view repeatability reference

Once an observer wrist has acquired a user-confirmed side view, save its blue
jaw silhouette and both measured EE poses. The clicked component is converted
to scale-independent axis, elongation, and fill features; absolute pixels and
area are never a horizontal gate. A future run reacquires the view with a
locally measured image Jacobian, then compares the normalized silhouette.

The side-view reference proves that observation geometry is repeatable. Strict
authorization requires both calibrated attachment geometry and an accepted
independent RGB-D consensus. An operator label alone cannot become a
production pass condition. The saver also requires the current right-arm pose
to match the pose recorded with that consensus within 2 mm and 0.75 degrees,
so a stale accepted JSON cannot certify a different pose.

```bash
python src/save_jaw_side_view_reference.py \
  --image /path/to/left_wrist.jpg \
  --target-x 471 --target-y 197 \
  --operator-label strictly_level \
  --physical-level-consensus \
    src/configs/pasteur_right_physical_gripper_level.json \
  --output src/configs/pasteur_right_jaw_side_view_reference.json
```
