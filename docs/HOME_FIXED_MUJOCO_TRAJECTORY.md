# Home-fixed MuJoCo trajectory

This offline workflow treats both physical arms as being at the hardware
homing pose at the beginning of every task. Only the dish/lid object state is
updated from perception. The physical left arm remains fixed at home while a
collision-checked physical-right trajectory is visualized.

## Coordinate and state contract

- Hardware homing values are defined once in `robot/arm/home.py`.
- Physical right maps to the MuJoCo `left_arm_*` branch.
- Physical left maps to the MuJoCo `right_arm_*` branch.
- The accepted SAM reconstruction uses natural `right/*` and `left/*`
  branches and the NYU gripper.
- Its model-specific `home` keyframe stays unchanged. Physical trajectories
  are transferred as deltas from the hardware home to that model home, so the
  correctly upright model is not accidentally replaced by the obsolete
  Cone-E scene.
- The planner and renderer never import a live robot client and never send
  commands.

The current sample target comes from an operator-verified successful grasp and
lift. It is intentionally `display_only`: the container pose is unknown and
the historical camera-to-robot transform has not been accepted. These missing
authorities remain explicit in the plan JSON and prevent it from being
mistaken for an executable trajectory.

## Reproduce

```bash
conda run -n robot-test python src/plan_home_lid_trajectory.py \
  --objects src/configs/pasteur_verified_lid_grasp_scene_20260730.json \
  --output /tmp/pasteur_home_lid_trajectory/plan.json

conda run -n robot-test python src/render_home_lid_trajectory.py \
  --plan /tmp/pasteur_home_lid_trajectory/plan.json \
  --output /tmp/pasteur_home_lid_trajectory/home_to_lid.mp4 \
  --report /tmp/pasteur_home_lid_trajectory/render_report.json \
  --width 640 --height 480 --fps 30
```

The plan uses minimum-jerk joint interpolation at 30 Hz. Every waypoint is
checked against joint limits, maximum adjacent joint change, new MuJoCo
contacts, and confirmed analytic object proxies. Three deterministic corridor
candidates are evaluated and the highest-clearance valid candidate is chosen.
The render report proves the fixed arm stayed at home by recording its maximum
drift.

The default model is the tracked, portable snapshot
`robot/pasteur-calibrated-scene/scene.mjcf`. It preserves the final
volume-optimized angled incubator placement, independently upright Piper
bases, NYU grippers, and collision geometry from
`automated_replay_20260730_v1`.
The obsolete Cone-E lab scene was removed because its arm bases render in the
previously rejected orientation. Tools must receive an explicit model or use
the accepted reconstruction above; there is no fallback to that scene.

Before this becomes motion-authoritative, perception must confirm both dish
body and lid in robot coordinates, camera calibration must be accepted, and
the resulting dynamic-object scene must be rerun through the same checks.
