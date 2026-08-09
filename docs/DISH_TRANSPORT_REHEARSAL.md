# Horizontal dish air-transport rehearsal

This pipeline rehearses three independent, dish-sized transport motions without
a dish:

1. physical right arm: incubator to microscope;
2. physical left arm: microscope to bench;
3. physical right arm: bench to incubator.

It intentionally excludes grasp, release, and handoff.  Air success is not
evidence that a real dish can be grasped or retained.

## What is automatic

`src/run_dish_transport_rehearsal.py` compiles every clean successful
open-close-open demonstration for a task, rejects recordings with extra
gripper transitions, resamples route shape by arc length, and selects the
route medoid.  It removes the low grasp/place portions and creates a shortest
path through source, a demonstrated high-clearance waypoint, and destination.
The commanded jaw plane is horizontal for the complete path.

The physical/model arm-name bridges are fixed in one module:

- physical right = production `left_arm_*` = semantic scene `right/*`;
- physical left = production `right_arm_*` = semantic scene `left/*`.

The planner solves a continuous physical joint proxy and audits both that arm
and a horizontal 90 x 14 mm virtual dish in
`robot/pasteur-calibrated-scene/scene.mjcf`.  Contacts already present in the
canonical/model-normalized start pose are recorded as calibration mismatch;
new contacts or materially deeper penetration fail the audit.

Because no verified left-arm station demonstrations exist, the middle segment
uses the right demonstration's motion translated by the measured difference
between physical home EE positions.  This is an operator-reviewed *air-only*
retarget, not a calibrated object transport transform.  At the time of writing
the semantic scene reports a substantial left-arm/microscope collision, so
normal execution remains blocked for that segment.

## Dry run

The default command never connects to the robot:

```bash
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_dish_transport_rehearsal.py
```

It writes `plan.json` and `plan_preview.html` under
`data/runs/pasteur/dish_transport_rehearsal/<timestamp>/`.

## Deliberate execution

Execution uses the same `set_{side}_ee_target` Cartesian RPC path as teleop,
at 30 Hz with a 50 ms preview.  It does not rewrite controller gains or CAN
mode.  Both arms are first sent to canonical home.  The air gripper remains
fully open.

```bash
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_dish_transport_rehearsal.py --execute
```

The command refuses to move if any MuJoCo route has collision warnings.  The
override is intentionally separate and visible:

```bash
... --execute --allow-audit-warnings
```

Do not use that override until the reported physical left/microscope mismatch
has been inspected or recalibrated.

Collision-clean segments can be exercised without overriding a different
segment's warning:

```bash
... --execute --segment incubator_to_microscope
```

At each of the nine stops (source lift, midpoint, arrival for every segment),
the process:

1. holds the measured pose;
2. checks measured jaw level;
3. captures head and active-wrist Record3D frames concurrently;
4. rejects a pair whose host timestamps differ by more than 150 ms;
5. saves raw checkpoint evidence and a side-by-side image;
6. waits on the phone UI at port 8097.

The UI offers continue, abort-and-hold, and abort-and-home.  Continue is
disabled when the measured 3-degree jaw-level gate fails.  Images are held in
memory until the next checkpoint, so they do not disappear after one poll.

Torque telemetry is warning-only, matching the currently accepted loose
teleop behavior.  Invalid tracking, a missed command deadline, camera failure,
or a failed level checkpoint still stops progress and holds the measured pose.

## Artifacts

Every run retains:

- the chosen demo path and SHA-256;
- all commanded Cartesian poses and physical joint proxies;
- MuJoCo contacts and virtual-dish dimensions;
- measured checkpoint pose and jaw-level metrics;
- head/wrist timestamps and skew;
- separate and combined checkpoint JPEGs;
- every operator decision and final home result.

The implementation is task-name agnostic.  New station transfers can be added
as config segments backed by at least three clean successful demonstrations.
Real-object use must add live object localization, pre-grasp, retention, and
release gates; it must not promote this air-rehearsal result directly.
