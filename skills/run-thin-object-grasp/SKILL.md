---
name: run-thin-object-grasp
description: Orchestrate, execute, resume, and audit the repository's thin-object edge-grasp primitives. Use for Petri lids or similar planar objects when Codex should make semantic checkpoint decisions, a deterministic controller should stream real-time right-arm motion, MuJoCo should audit a route, the gripper must remain level, or a failed grasp must be diagnosed from immutable result.json evidence.
---

# Run Thin-Object Grasp

Use reviewed primitives and immutable checkpoint evidence. Never replace a
failed gate with an improvised arm command.

## Start

1. Read `docs/CODEXLESS_THIN_OBJECT_GRASP.md` and
   [references/pipeline-contract.md](references/pipeline-contract.md).
2. Confirm the physical target is present. Treat old images and caches only as
   seeds, never current evidence.
3. Confirm `physical_arm=right`, `execution.left_arm_commands=0`, torque is
   observe-only, pressure safety remains enabled, and no other controller owns
   `/tmp/piper_robot_right_arm_controller.lock`.
4. Create a unique run directory and run the I/O-free audit:

   ```bash
   /home/admin/miniforge3/envs/robot-test/bin/python \
     src/run_thin_object_grasp_primitive.py \
     --run-dir data/runs/pasteur/<run>/checkpoints audit
   ```

## Orchestrate

Advance only through these named primitives:

`observe → plan-hover → move-hover → align-hover → descend-bottom → seat-2mm → close → verify-lift → home`

Use `recover` or `stop` whenever the current result does not authorize its
nominal successor. Inspect each `result.json`, its evidence images, measured
state, and `allowed_next_actions`. Bind a decision to that exact result:

```bash
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_thin_object_grasp_primitive.py \
  --run-dir data/runs/pasteur/<run>/checkpoints decide \
  --result <result.json> --action <allowed-action> \
  --reason '<evidence-based reason>'
```

Do not edit `result.json` or `decision.json`. Do not auto-retry physical
motion. A changed profile, target, camera pose, or robot state invalidates the
old decision chain.

## Control boundary

- Let deterministic code own camera polling, SAM/shape extraction, IK,
  MuJoCo collision checks, 30 Hz Teleop-compatible streaming, pressure safety,
  and measured joint convergence.
- Let Codex choose only among `allowed_next_actions` at semantic checkpoints.
- Keep the left arm measured-but-fixed and never send a left-arm command.
- Use the selected target's circular/rim shape as primary semantic identity.
  Treat the blue cross and AprilTag as optional continuity or metric anchors;
  neither may replace an inconsistent shape observation.
- Check jaw level immediately before normal descent and after the single final
  seating command. Do not add continuous 30 Hz vision reasoning.
- Complete normal descent first. Then issue exactly one additional 2 mm down
  command with XY and orientation fixed, recapture, recheck level/shape, and
  only then allow one continuous close.
- Lift vertically and require both calibrated obstruction and target-follow
  evidence. An attractive image, a hover, or obstruction alone is not success.

## Recovery and learning

On a low-pose rejection, lift vertically before opening or joint-space
retreat. Preserve the rejected result and evidence. Promote calibration only
after an end-to-end verified lift; never promote from alignment alone.

Once the primitive sequence is stable across replays, use the same contracts
and transitions in a Codexless state machine. Do not maintain a second policy.

## Editing

- Put contracts, hashes, locks, and transition rules in
  `rollout/grasp_orchestration.py`.
- Put deterministic perception/evidence in `rollout/thin_object_grasp.py`.
- Put real-time RPC execution in `src/run_codexless_thin_object_grasp.py`.
- Put site calibration in JSON, not unexplained pixels in control flow.
- Add replay/unit tests before another physical run.

Validate changes with:

```bash
/home/admin/miniforge3/envs/robot-test/bin/python -m pytest -q \
  tests/test_grasp_orchestration.py \
  tests/test_codexless_thin_object_grasp.py \
  tests/test_grasp_window.py \
  tests/test_teleop_trajectory_stream.py \
  tests/test_fast_lid_grasp.py \
  tests/test_gripper_level.py
```
