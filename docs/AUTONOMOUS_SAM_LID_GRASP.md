# Autonomous SAM/RGB-D lid grasp

`src/run_autonomous_sam_lid_grasp.py` is the standalone entry point for the
right-arm lid task. It never homes or automatically returns either arm.

The runtime loop is:

1. Resolve all Record3D cameras by UDID, not enumeration index.
2. Validate an explicitly accepted head-camera-to-robot transform.
3. Run SAM on a fresh head image and estimate the transparent lid edge from
   the locally fitted RGB-D support plane.
4. Segment the right arm as a dynamic layer and build a conservative current
   ESDF. Unknown voxels are not free.
5. Validate every 30 Hz Cartesian waypoint with the production 6-DOF MuJoCo
   model (continuous IK, joint limits and new contacts) plus measured ESDF
   clearance.
6. Stream only 0.5 seconds at a time using the same 30 Hz,
   `preview_time=0.05` target sender as right-arm teleoperation.
7. Reobserve and replan after each chunk for a target shift over 10 mm, a
   position error over 5 mm, an orientation error over 3 degrees, low/unknown
   clearance, a changed target instance or a missing/stale observation.
8. Before closing, require the right-camera blue marker within 8 px of the
   demonstration goal and an estimated gripper-to-lid distance at most 10 mm.
9. Verify the grasp from gripper close residual, right-camera marker
   persistence and lid motion during a short lift.

Every observation, plan, commanded chunk, actual pose, replan cause, SAM
artifact and terminal result is atomically persisted in `run_state.json`.
`--resume` reopens that journal. A normal invocation is fully automatic once
preflight passes:

```bash
python src/run_autonomous_sam_lid_grasp.py \
  --output-dir /tmp/autonomous_lid_001
```

Use live perception and planning without motion:

```bash
python src/run_autonomous_sam_lid_grasp.py \
  --dry-run \
  --output-dir /tmp/autonomous_lid_dry
```

Use an offline scene JSON containing `target_camera_xyz_m` and
`ee_pose_wxyz_xyz`:

```bash
python src/run_autonomous_sam_lid_grasp.py \
  --offline-scene /path/to/scene.json \
  --output-dir /tmp/autonomous_lid_offline
```

The checked-in calibration is intentionally unaccepted. Motion fails closed
until the multi-pose, SAM-labelled robot-CAD fit has populated and accepted
`src/configs/pasteur_head_robot_calibration.json`. To invalidate it:

```bash
python src/run_autonomous_sam_lid_grasp.py --reset-calibration
```

Torque/non-finite samples, IK branch jumps and new MuJoCo contacts abort and
hold immediately. They are never converted into a replan. The historical tiny
joint probes remain diagnostic-only and are not used by this runtime.
