# Thin-object edge grasp: checkpoint orchestration and Codexless execution

The physical controller is deterministic and Codex is never in the 30 Hz
control loop. During development, Codex or an operator may choose only among
actions authorized by immutable semantic checkpoints. Once repeated replays
are stable, the same transition contract can run Codexless without a second
policy.

The sequence is:

`observe → plan-hover → move-hover → align-hover → descend-bottom → seat-2mm → close → verify-lift → home`

Every physical failure routes through `recover` or `stop`. Run the I/O-free
contract and MuJoCo audit with:

```bash
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_thin_object_grasp_primitive.py \
  --run-dir data/runs/pasteur/<run>/checkpoints audit
```

## Run it

Use the `robot-test` environment from the repository root. A command without `--execute` performs model/profile audit only and sends no robot command.

```bash
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_codexless_thin_object_grasp.py \
  --output-dir data/runs/pasteur/codexless_thin_object_dry_run
```

After confirming the robot server, camera mapping, lighting, clear workspace, and absence of another controller session, use a unique output directory:

```bash
/home/admin/miniforge3/envs/robot-test/bin/python \
  src/run_codexless_thin_object_grasp.py \
  --output-dir data/runs/pasteur/codexless_thin_object_<timestamp> \
  --cycles 3 --max-attempts 8 --execute
```

`--cycles 3` requires three consecutive complete grasps. A failed attempt resets the consecutive count. `run.json.complete` is the run-level result.

## State machine

The execution order is fixed:

1. Reacquire the tapped target with the fixed head camera and support plane.
2. Reject target motion outside the locally calibrated envelope.
3. Audit the right trajectory in MuJoCo while holding the measured left arm fixed.
4. Stream a right-arm transit to a level hover using the Teleop-compatible 30 Hz RPC path.
5. Align in tool-relative coordinates with bounded metric XY corrections.
6. Check measured jaw level once, then descend vertically with the measured hover quaternion fixed.
7. Require a fresh nominal low-pose target/tool observation and a second measured level check. Correct measured fingertip mismatch with the jaws open even when XY is not yet accepted. If the open jaws settle slightly high, move straight down to the audited support height and recapture before making any XY decision.

The support-contact height is independent from the visual preclose seed. The Pasteur profile uses the operator-verified `20260730T084719...verified_true_preclose` pose (`z=0.837372 m`), which subsequently produced a mechanically nonempty close and a 12.3 mm target-following lift. Do not replace it with the height of a merely camera-aligned frame. Normal height correction must finish first. Then `seat-2mm` sends exactly one additional 2 mm downward request with XY/orientation fixed and jaws open. It is not part of the correction loop and cannot repeat in the same attempt. The executor recaptures and checks jaw level and target alignment again before closure.
8. Close once continuously; compare the measured aperture with calibrated empty/non-empty populations.
9. Lift straight up by an initial 2 mm and require obstruction to persist; table friction or a shallow rim contact that disappears here is a failed grasp. Continue the remaining lift only after this check, and require the same selected target to follow.
10. Return to support, open, and retreat. Record one end-to-end success.

Every low-pose rejection recovers vertically before opening or joint-space retreat. The executor never sends a left-arm command. Torque readings are recorded as warnings and do not stop motion.

## Perception and generalization

Control flow contains no absolute target pixel threshold. The target's circular
rim/SAM mask is the primary semantic identity for general operation. The blue
cross may provide local continuity and AprilTags may provide metric camera
registration, but both are auxiliary and cannot override inconsistent target
shape. Cyan gripper material plus the light finger pad supplies a per-frame
tool coordinate system. Areas are normalized by tool scale.

SAM, RGB-D, and MuJoCo are useful upstream for object/scene calibration and collision geometry. They do not replace the low-pose closure gate. The legacy monolithic executable still validates its configured feature adapter; add a replay-tested semantic adapter before switching that backend rather than silently treating an unmarked target as blue.

Normal light-pad detection uses the calibrated material threshold. If a darker image requires the configured fallback, that observation may only prove a regression and return to a previously observed best pose. It cannot close or create a new exploratory probe.

## Resume semantics

Two atomic sidecars preserve learning across processes:

- `data/runs/pasteur/codexless_thin_object_runtime_alignment.json`: stable local hover seed.
- `data/runs/pasteur/codexless_thin_object_pending_alignment.json`: fresh low-pose proposal or regression backtrack that has not completed a grasp.

Both are tied to the fixed-head target scene point and marker position. A pending seed always carries `closure_authorized=false` and `fresh_preclose_required=true`. It is promoted only after a complete grasp and target-follow verification. Moving the target outside the envelope invalidates replay and sends no arm command unless a global head-to-arm relocation transform has been separately calibrated.

For a revalidated pending low-pose correction, the executor may skip the redundant high wrist-camera identity frame. Eye-in-hand parallax can put the target outside that vertically offset view. This does not skip perception for closure: the gripper remains open, the executor visits the proposed low pose, and a newly captured low image plus measured jaw-level gate must pass before the first close command.

## Artifacts and audit

Each run writes `run.json`; each attempt writes `attempt.json`, hover images, `preclose.png`, and verification images when reached. Useful fields include:

- `left_arm_commands` and `left_arm_max_abs_delta_rad`
- `hover.normalized_center_error` and `hover.level`
- `preclose.observation.grasp_window`, `preclose.level`, and `allowed_to_close`
- `visual_replan`, `pending_alignment`, and recovery method
- `closure_before_lift`, target-follow evidence, placement, and ledger

Do not call a run successful from a single image or obstruction reading.

## Regression suite

```bash
/home/admin/miniforge3/envs/robot-test/bin/python -m pytest -q \
  tests/test_grasp_orchestration.py \
  tests/test_codexless_thin_object_grasp.py \
  tests/test_grasp_window.py \
  tests/test_teleop_trajectory_stream.py \
  tests/test_fast_lid_grasp.py \
  tests/test_gripper_level.py
```

The 2026-08-05 physical replay improved normalized preclose center error from `0.694` to `0.393` while maintaining sub-degree measured level, then rejected a darker third pose before closure. The saved third image now exercises the dark-scene regression path in the test suite. This is evidence that alignment learned and fail-closed behavior worked; it is not yet an end-to-end grasp success.
