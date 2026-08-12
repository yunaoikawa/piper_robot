# ACT agent correction data collection

This workflow is intentionally isolated on branch `vla-agent-data-collection`.
It shares the physical robot services and cameras, but it does not modify the
ordinary LLM-control recorder or its datasets.

## Invariants

- One process owns the right arm through
  `/tmp/piper_robot_right_arm_controller.lock`.
- ACT actions are absolute, served at 30 Hz, horizon 40, replanned when 12
  actions remain. Delta ACT is rejected.
- Only the right arm is commanded. The left arm remains available as a camera.
- Right bias starts at `[0, 0, -0.03]` m on every episode and is applied exactly
  once in the client, in robot coordinates.
- Pause holds the measured EE pose and gripper, then clears both server queues.
  Resume clears again and rejects actions generated from pre-resume images.
- Pause time is omitted from the training timeline.
- Human-confirmed successes are atomically promoted to `success/<task>`.
  Failures are retained under `failures/<task>` with a reason and are never
  imported when `--require-success-manifest` is used.

## Start

On the inference GPU:

```bash
python cloud_inference_clean-main/hpc_inference_act.py \
  --checkpoint outputs/lab/act/horizon/lid_open/ep100/checkpoints/last/pretrained_model \
  --pred_horizon 40 --replan-at 12 --active-arm right --hz 30
```

On Pasteur, from this worktree:

```bash
python cloud_inference_control_collect_v2.py \
  --host <INFERENCE_IP> --agent-collection --agent-task lid_open \
  --agent-root data/vla_agent/lid_bias --agent-ui-port 8780
```

Open `http://<PASTEUR_TAILSCALE_IP>:8780/`. Tap the lid in the head image,
start, and use `PAUSE → X/Y/Z ± 1/2/5 mm → RESUME`. Move the physical lid by
hand between episodes. Confirm SUCCESS or select a failure reason. Repeat until
10 successes, then repeat with `lid_close` and its checkpoint.

## Audit and training conversion

```bash
python src/replay_agent_interventions.py \
  data/vla_agent/lid_bias/success/lid_open/<episode>

python src/convert_to_lerobot.py \
  --data_dirs data/vla_agent/lid_bias/success/lid_open \
  --task_names "open the petri dish lid" \
  --output_dir data/train/lid_open_agent --repo_id yunaoikawa/lid_open_agent \
  --fps 30 --require-success-manifest --intervention-slice all
```

`all` trains from the complete corrected measured trajectory. For an ablation,
`post-intervention` starts at the first nudge. In either case the action label is
the next measured EE state, not the uncorrected policy proposal.
