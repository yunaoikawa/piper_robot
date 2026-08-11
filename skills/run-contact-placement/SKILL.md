---
name: run-contact-placement
description: Plan, execute through deterministic adapters, resume, and audit generalized pressure-guarded object placement. Use when an object must move between different start and goal poses, a gripper must stay level, support contact must be distinguished from a stalled kinematic branch, every move needs fresh phone-visible evidence, or the workflow must transfer to another laboratory without fixed pixels, tags, or Pasteur-specific coordinates.
---

# Run Contact Placement

Use [references/contact-placement-contract.md](references/contact-placement-contract.md)
as the transition and portability contract. Keep language-model reasoning out
of the real-time loop; use it only to select or diagnose named checkpoints.

## Prepare

1. Confirm one process owns the selected physical arm. Resolve the production
   and semantic model branches with `ArmIdentity`; never infer physical side
   from image left/right or a mesh branch name.
2. Require a motion-ready semantic scene. If it is missing or stale, first use
   `$reconstruct-3d-scene` to refresh SAM/depth objects, static collision
   geometry, exact robot CAD, and the camera-to-robot transform.
3. Copy `src/configs/contact_placement_profile.example.json` into a site or
   run profile. Calibrate tool axes and dimensions. Put site paths and metric
   envelopes in that profile, never in Python control flow.
4. Record the measured start joint state and EE pose. Resolve the goal either
   from a semantic object in robot coordinates or from one confirmed normalized
   RGB-D point. Treat AprilTags as optional metric anchors, not object identity.

## Plan

Run the deterministic planner:

```bash
python src/run_contact_placement_pipeline.py plan \
  --profile <site-profile.json> \
  --start <measured-start.json> \
  --goal <semantic-or-rgbd-goal.json> \
  --output <run-dir>/plan.json
```

Do not execute unless `motion_ready=true`. That requires exact-CAD IK on the
measured branch and an accepted semantic-scene collision audit. A pose-only
plan is preview evidence, not motion authority.

## Advance

After each named primitive, acquire a new synchronized frame set and advance
the persisted state:

```bash
python src/run_contact_placement_pipeline.py advance \
  --profile <site-profile.json> --goal <goal.json> \
  --state <run-dir>/state.json --observation <run-dir>/observation.json \
  --output <run-dir>/transition.json --next-state <run-dir>/state.json
```

Allow a hardware adapter to execute only the returned action. Use the existing
30 Hz Teleop-compatible joint streamer for approach/rebranch/retract. During
descent, keep `RecoveryTorqueGuard` enforced and hold the measured state on a
trip. Never replace a rejected transition with an improvised Cartesian move.

The normal sequence is:

`observe → approach → align → descend → contact → release → verify → retract`

`stalled_without_contact_evidence` routes back through `rebranch`. A stall is
contact only when pressure or support-plane agreement also exists. Release is
allowed only over the verified support, and retract only after the object is
seen remaining on that support.

## Show every checkpoint

Publish the exact frames used by the decision, then keep one phone URL open:

```bash
python src/run_contact_placement_pipeline.py publish \
  --profile <site-profile.json> --evidence <checkpoint-evidence.json> \
  --directory <run-dir>/phone
python src/run_contact_placement_pipeline.py serve \
  --directory <run-dir>/phone --host 0.0.0.0 --port 8774
```

The publisher rejects stale, reused, missing, or overly skewed images and
persists frame IDs, timestamps, hashes, physical arm, stage, and action. Do not
continue when the phone page is unavailable or shows an older revision.

## Generalize and promote

- Derive image coordinates from normalized selections and object/tool scale.
  Never promote a raw pixel, a fixed camera resolution, or largest-component
  selection into the controller.
- Re-measure the start on every run. A new goal position reuses the policy but
  requires a fresh goal estimate and collision plan.
- A new lab must recalibrate camera-to-robot transforms, support normals, tool
  axes, and collision geometry. It may reuse state transitions and dimensionless
  thresholds.
- Keep SAM minimal: use it to label targets/supports/obstacles. Use measured
  joints plus CAD for robot geometry and pressure/support evidence for contact.
- Promote a site runner only after replay tests cover changed starts, goals,
  image resolutions, physical arms, stale cameras, branch stalls, pressure
  contact, release verification, and collision rejection.

## Validate edits

```bash
python -m pytest -q tests/test_contact_placement_pipeline.py
python /home/admin/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/run-contact-placement
```
