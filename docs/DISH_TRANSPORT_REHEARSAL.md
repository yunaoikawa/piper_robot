# Horizontal dish air-transport rehearsal

This pipeline rehearses dish-sized station transfers without a dish. It
intentionally excludes grasp, release, and handoff; air success is not evidence
that a real dish can be retained.

## Planning

`src/run_dish_transport_rehearsal.py` filters clean open-close-open recordings.
For recordings that return before opening, it treats the furthest carried pose
as the destination turnaround instead of incorrectly treating release back at
the source as the destination. It selects the outbound-route medoid and uses
its station positions and obstacle-avoiding XY bend. It deliberately discards
the human's high middle Z. Starting 15 mm
above the recorded station poses, it raises the route in 5 mm increments until
MuJoCo reports no new contact and at least 10 mm virtual-dish clearance. The
first passing candidate is the lowest route at that resolution.

The physical/model arm bridge is fixed in one module:

- physical right = production `left_arm_*` = semantic scene `right/*`;
- physical left = production `right_arm_*` = semantic scene `left/*`.

The virtual dish is 90 x 14 mm and its centre is 75 mm along physical
fingertip-outward **EE -X**. Using +X puts it behind the wrist and invalidates
the clearance audit.

The commanded jaw plane stays horizontal. IK is seeded only from the preceding
solution after branch selection; a failed seed cannot silently switch to a
wrist-roll multistart. Consecutive planning samples must differ by at most
0.12 rad. Arm and virtual-dish swept geometry are checked in
the reviewed current scene at
`data/runs/pasteur/current_scene_automatic_20260731/current_scene/scene.mjcf`.
That reconstruction is display-ready but not motion-ready; a passing preview
is therefore visualization evidence only and cannot authorize execution.

## Dry run

The default command does not connect to the robot. `--segment` filters before
planning, so an unrelated failing segment cannot block the requested route:

```bash
PYTHONPATH=. /home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_dish_transport_rehearsal.py \
  --segment incubator_to_microscope
```

It writes `plan.json` and `plan_preview.html` under
`data/runs/pasteur/dish_transport_rehearsal/<timestamp>/`.

## Stopped left observer

For a physical-right transfer, the left wrist camera follows the carrier only
at the five stops: departure, 25%, midpoint, 75%, and arrival. The carrier is
held before any left command; the two arms are never commanded simultaneously.
The observer keeps the operator-verified height and follows only bench-plane
XY, avoiding the earlier mistake of descending into the microscope.

The left image must contain positive blue jaw shape matching the saved side
view. Dark silhouettes and shadows cannot be targets. The coordinated schedule
is audited with both arms and the carried virtual dish. A contact already
present at the operator-verified observer pose is retained only as a calibrated
model mismatch; any new pair or penetration worsening is rejected.

## Deliberate execution

Execution uses the teleop `set_{side}_ee_target` RPC at 30 Hz, 0.03 m/s for the
carrier and 0.02 m/s for the observer. Both arms start at canonical home and the
air gripper remains fully open. Torque is warning-only; communication loss,
tracking failure, IK discontinuity, camera failure, or collision prediction
holds the current measured pose.

```bash
PYTHONPATH=. /home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_dish_transport_rehearsal.py --execute \
  --segment incubator_to_microscope
```

At every stop the process:

1. holds the right carrier;
2. moves the left observer alone;
3. checks right-jaw level (1 degree maximum);
4. captures head RGB-D and both wrist RGB images within 150 ms;
5. checks the positive-blue left side view;
6. displays all three images and metrics on the phone UI at port 8097;
7. waits for continue, abort-and-hold, or abort-and-home.

Departure alone may run bounded level refinement. Later stops measure and hold;
they do not repeatedly nudge the object path. After arrival the left observer
returns home while the right holds, then the right reverses its audited path.

## Artifacts and current gate

Every run retains the demo hash, commands, joint proxies, MuJoCo contacts,
clearance, head depth, three RGB views, side-view decision, timing skew,
operator decisions, and final home result.

The current right `incubator_to_microscope` route passes: the lowest candidate
adds 5 mm, has about 80 mm minimum modelled dish clearance, and a 0.0375 rad
maximum IK step. The stopped left-observer schedule remains blocked because
the semantic scene reports its approach crossing the microscope. Execution
therefore remains prohibited until the microscope/left-arm alignment is fixed
or a collision-clean observer approach is found. `--allow-audit-warnings` is
not a substitute for resolving this two-arm collision.

New station transfers require at least three clean demonstrations. Real-object
use must additionally provide localization, pre-grasp, retention, and release
gates; an air result must not be promoted directly.
