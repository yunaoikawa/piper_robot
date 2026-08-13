#!/usr/bin/env bash
# Exact inference-side reproduction of the ACT used after the 2026-06-25
# microscope->bench training run. Run inside a Slurm GPU allocation.
set -euo pipefail

cd /home/yoikawa/src/robot-vla-data
export PYTHONPATH="/home/yoikawa/src/robot-vla-data/.runtime_deps${PYTHONPATH:+:${PYTHONPATH}}"
exec /home/yoikawa/miniconda3/envs/lerobot/bin/python \
  cloud_inference_clean-main/hpc_inference_act.py \
  --checkpoint /home/yoikawa/src/robot/outputs/lab/act/after0610/petri/microscope/checkpoints/last/pretrained_model \
  --obs-port "${OBS_PORT:-5755}" \
  --action-port "${ACTION_PORT:-5756}" \
  --device cuda \
  --pred_horizon 50 \
  --replan-at 0 \
  --active-arm both \
  --legacy-full-chunk \
  --hz 30
