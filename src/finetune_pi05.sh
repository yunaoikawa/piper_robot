#!/bin/bash
#SBATCH --job-name=pi05_finetune
#SBATCH --output=logs/pi05_finetune_%j.out
#SBATCH --error=logs/pi05_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NOTE: this node does NOT schedule GPUs via Slurm gres (AllocTRES has no gpu);
# every job sees all 8 A100s and the nvidia-smi scan below self-selects a free
# one (>= MIN_FREE_MIB). So we only need to request CPU/mem here. `--gpus` is
# kept for portability but is effectively a no-op on this cluster.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#
# Usage:
#   bash src/train.sh
#
# 4-GPU DDP: batch_size=32 per GPU × 4 GPUs = 128 effective batch size
#
# Latest checkpoint always at: outputs/main/checkpoints/last/pretrained_model

# ── Setup ─────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs

cd /home/yoikawa/src/robot
source .venv/bin/activate

# ── Configure ─────────────────────────────────────────────────────────────────
DATASET_REPO_ID="yoikawa/lab"
# Optional positional args: $1=DATASET_ROOT  $2=OUTPUT_DIR (defaults below)
DATASET_ROOT="${1:-data/train/new_fps23/}"
OUTPUT_DIR="${2:-outputs/lab_act/}"
JOB_NAME="pi05_finetune"

STEPS=100000

BATCH_SIZE=8          # per GPU (×4 GPUs = 128 effective)
NUM_WORKERS=4          # per GPU
SAVE_FREQ=500
NUM_GPUS=1
MIN_FREE_MIB=40000
# ──────────────────────────────────────────────────────────────────────────────

# ── Wait for N free GPUs ──────────────────────────────────────────────────────
# This node doesn't schedule GPUs via Slurm, so a job can land while all GPUs are
# busy. Rather than fail instantly, poll until NUM_GPUS have >= MIN_FREE_MIB free
# (or give up after GPU_WAIT_SECS). The job only holds its cheap CPU/mem slot
# while waiting.
GPU_WAIT_SECS=21600     # max time to wait for free GPU(s) (6h)
GPU_POLL_SECS=60
WAITED=0
while :; do
    # `|| FREE_GPUS=""` keeps a transient nvidia-smi failure from tripping `set -e`.
    FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F',' -v min="$MIN_FREE_MIB" '{gsub(/ /,"",$2); if ($2+0 >= min) print $2, $1}' \
        | sort -rn \
        | head -${NUM_GPUS} \
        | awk '{print $2}' \
        | sort -n \
        | paste -sd,) || FREE_GPUS=""

    if [[ -n "$FREE_GPUS" ]]; then
        N_FOUND=$(echo "$FREE_GPUS" | tr ',' '\n' | grep -c .)
    else
        N_FOUND=0
    fi

    [[ "$N_FOUND" -ge "$NUM_GPUS" ]] && break

    if [[ "$WAITED" -ge "$GPU_WAIT_SECS" ]]; then
        echo "ERROR: Timed out after ${WAITED}s waiting for ${NUM_GPUS} GPU(s) with >= ${MIN_FREE_MIB} MiB free. Found ${N_FOUND}."
        nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu \
            --format=csv,noheader
        exit 1
    fi

    echo "[gpu-wait] Need ${NUM_GPUS} GPU(s) >= ${MIN_FREE_MIB} MiB free; found ${N_FOUND}. Sleeping ${GPU_POLL_SECS}s (waited ${WAITED}s)."
    sleep ${GPU_POLL_SECS}
    WAITED=$((WAITED + GPU_POLL_SECS))
done

export CUDA_VISIBLE_DEVICES=$FREE_GPUS

# ── Handle existing checkpoint ────────────────────────────────────────────────
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
PRETRAINED_PATH="lerobot/pi05_base"

if [[ -f "${CHECKPOINT_DIR}/config.json" ]]; then
    BACKUP_DIR="outputs/main_prev_$(date +%Y%m%d_%H%M%S)"
    echo "Found existing checkpoint. Backing up to ${BACKUP_DIR}"
    mv "${OUTPUT_DIR}" "${BACKUP_DIR}"

    PRETRAINED_PATH="${BACKUP_DIR}/checkpoints/last/pretrained_model"

    sed -i 's/"compile_model": true/"compile_model": false/' "${PRETRAINED_PATH}/config.json" 2>/dev/null
    sed -i 's/"compile_model": true/"compile_model": false/' "${PRETRAINED_PATH}/train_config.json" 2>/dev/null

    echo "RESUMING from: ${PRETRAINED_PATH}"
else
    echo "No existing checkpoint. Starting FRESH from lerobot/pi05_base"
fi

echo "============================================="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "GPUs:         ${FREE_GPUS} (${NUM_GPUS} GPUs)"
echo "Batch size:   ${BATCH_SIZE}/GPU × ${NUM_GPUS} = $((BATCH_SIZE * NUM_GPUS)) effective"
echo "Output dir:   $OUTPUT_DIR"
echo "Dataset:      $DATASET_REPO_ID  @ ${DATASET_ROOT}"
echo "Pretrained:   $PRETRAINED_PATH"
echo "Steps:        $STEPS"
echo "============================================="

#
lerobot-train \
    --dataset.repo_id=${DATASET_REPO_ID} \
    --dataset.root=${DATASET_ROOT} \
    --policy.type=pi05 \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${JOB_NAME} \
    --policy.pretrained_path=${PRETRAINED_PATH} \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    --steps=${STEPS} \
    --policy.device=cuda \
    --batch_size=${BATCH_SIZE} \
    --num_workers=${NUM_WORKERS} \
    --save_freq=${SAVE_FREQ} \
    --wandb.enable=true \
    --policy.push_to_hub=false \
    --dataset.video_backend=pyav \
    --optimizer.lr=2.5e-5 \
    --scheduler.type=cosine_decay_with_warmup \
    --scheduler.num_warmup_steps=500 \
    --scheduler.peak_lr=2.5e-5 \
    --scheduler.num_decay_steps=100000 \
    --scheduler.decay_lr=1e-6 \
    --use_policy_training_preset=false \
    --tolerance_s=0.04

echo ""
echo "Training complete at $(date)"
echo "Output: $OUTPUT_DIR"
echo "Checkpoint: ${OUTPUT_DIR}/checkpoints/last/pretrained_model"