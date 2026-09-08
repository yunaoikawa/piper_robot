---
name: run-cylindrical-cap-transfer
description: Audit, execute, resume, and promote reusable vertical cylindrical-cap side-pinch removal and held transfer. Use for culture-media bottle caps or similar upright caps when a user tap identifies the cap, a fixed head camera guides two jaws around it, the container may be recessed between supports, or Codex should hand deterministic perception, 30 Hz motion, lift validation, and replay promotion to repository code.
---

# Run Cylindrical Cap Transfer

Use [references/cap-transfer-contract.md](references/cap-transfer-contract.md)
for gates and failure routing. Treat the user tap as immutable identity in the
fixed head view; derive all image tolerances from jaw/support scale.

## Start

1. Confirm the right arm starts at `physical_home_q("right")`, the gripper is
   open, the head image is fresh, and no other process owns the arm.
2. Confirm the selected object is the cap, not a gripper mount, tag, bottle
   shoulder, or microscope part. Use cap/rim shape plus the supporting bottle;
   AprilTags are localization-only and paper tags are not obstacles.
3. Run the immutable evidence audit before promoting any stored route:

   ```bash
   /home/admin/miniforge3/envs/robot-test/bin/python \
     src/audit_cylindrical_cap_transfer.py \
     --task-profile src/configs/pasteur_culture_media_cap_grasp.json \
     --before <closed-before-lift-capture> \
     --lift <initial-lift-capture> \
     --transported <held-transport-capture>
   ```

## Execute

Use one deterministic entrypoint; do not reproduce intermediate arm commands
in shell snippets:

```bash
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_culture_media_cap_grasp.py \
  --task-profile src/configs/pasteur_culture_media_cap_grasp.json \
  --output-dir data/runs/pasteur/cap_transfer_<timestamp> \
  --stage grasp-transfer --execute
```

The code owns fixed-head tracking, support-motion rejection, Cartesian
correction, closure, 10 mm lift, source-clearance validation, and free-space
egress auditing. Codex chooses whether current evidence authorizes the
entrypoint; it is never part of the 30 Hz loop.

## Promotion boundary

- Promote only `cap_side_pinch_lift_and_held_transport` when all evidence gates
  pass. Persistent non-empty aperture alone is insufficient.
- Treat a MuJoCo contact rejection as blocking for every new route. The sole
  exception is an exact hash match to a previously successful hardware route,
  accepted immutable lift/transport evidence, joint limits, and bounded joint
  steps. Record that narrow model false-positive override in the run report.
- Do not promote placement from an open command. Require a fresh destination
  support, bottom clearance, cap-on-support evidence after opening, and a
  vertical retract in which the cap does not follow.
- Keep site waypoints in JSON and label them hardware-observed. Keep image
  thresholds dimensionless. A new bottle position may reuse the adapter and
  gates but must reacquire its tap and plan a fresh collision-free approach.
- On any low/contact failure, keep the measured aperture, lift vertically,
  then recover. Never move laterally at bottle height.

## Validate edits

```bash
/home/admin/miniforge3/envs/robot-test/bin/python -m pytest -q \
  tests/test_media_cap_target.py \
  tests/test_cylindrical_cap_transfer.py
python /home/admin/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  skills/run-cylindrical-cap-transfer
```
