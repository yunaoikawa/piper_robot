# Cylindrical cap transfer contract

## Geometry and ownership

Use this contract for an upright cap removed by a horizontal side pinch. The
container may sit on a surface or be recessed between two supports. Deterministic
code owns camera sampling, normalized perception, IK, collision audit, 30 Hz
streaming, pressure safety, and measured convergence. Codex or a human only
selects a semantic action from fresh evidence.

## State sequence

`observe → approach-open → align-fixed-head → close → lift-probe → verify-removal → transport-hold`

Placement is a separate, unpromoted sequence:

`select-support → plan-clearance → descend-held → open → verify-on-support → retract`

Never append placement automatically merely because transport succeeded.

For a route that MuJoCo falsely marks in contact at an already lifted start,
allow replay only when the waypoint hash exactly matches immutable successful
hardware evidence and the full sampled route stays within joint/step limits.
This exception never authorizes a modified or newly planned path.

## Required gates

| Transition | Required evidence |
| --- | --- |
| Observe | Fresh fixed-head frame and one user-selected cap/rim identity |
| Approach | Current RGB-D scene; container/support relation; MuJoCo-clear path |
| Align | Immutable tap lies between two visible jaw centers; container body remains fixed |
| Close | One continuous close; aperture lies in calibrated non-empty band |
| Lift | Measured vertical progress; jaw moves in head view; source-local cap appearance clears |
| Verify removal | Coloured container/support displacement stays below a scale-normalized limit |
| Transport | Useful EE displacement; non-empty aperture persists; source remains clear |
| Place | Destination support is fresh and separate from the source recess |
| Release | Cap is detected on support after open and does not follow vertical retract |

The fixed-head source ROI radius is expressed in current jaw spans. Support
motion is divided by the observed support diagonal. Do not introduce absolute
pixel positions or largest-white-component selection.

## Failure routing

- If cap identity or both jaws are absent, stop without motion.
- If the bottle/support moves during alignment, backtrack the last free-space
  correction and stop.
- If closure is empty, lift vertically before opening and record failure.
- If source clearance or aperture persistence fails after the lift probe, hold
  or recover vertically; do not begin transport.
- If a transport endpoint cannot verify the held state, keep the gripper
  closed and stop for inspection.
- If placement evidence is missing, mark placement unverified even if the open
  command and retract completed.

## Calibration promotion

Store target appearance and kinematic route separately. A new cap may reuse
the semantic adapter and dimensionless gates. A changed camera, robot mounting,
or target location invalidates stored waypoints. Promote a route only from
immutable before-lift, after-lift, and transported-hold captures plus measured
robot state. Never promote an unverified release.
