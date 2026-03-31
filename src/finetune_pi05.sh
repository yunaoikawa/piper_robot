#!/bin/bash
#SBATCH --job-name=pi05_finetune
#SBATCH --output=logs/pi05_finetune_%j.out
#SBATCH --error=logs/pi05_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#
# Usage - always the same command:
#   bash src/finetune_pi05.sbatch
#
# First run:  trains from lerobot/pi05_base → saves to outputs/main
# Next runs:  backs up outputs/main → trains from backup → saves to outputs/main
#
# Latest checkpoint always at: outputs/main/checkpoints/last/pretrained_model

# ── Setup ─────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs

cd /home/yoikawa/src/robot
source .venv/bin/activate

# ── Configure ─────────────────────────────────────────────────────────────────
DATASET_REPO_ID="yoikawa/flask_tasks"
DATASET_ROOT="data/train/v3"
OUTPUT_DIR="outputs/main"
JOB_NAME="pi05_finetune"

STEPS=100000
BATCH_SIZE=8
NUM_WORKERS=2
SAVE_FREQ=500
MIN_FREE_MIB=50000
# ──────────────────────────────────────────────────────────────────────────────

# ── Auto-select free GPU ──────────────────────────────────────────────────────
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v min="$MIN_FREE_MIB" '{gsub(/ /,"",$2); if ($2+0 >= min) print $2, $1}' \
    | sort -rn \
    | head -1 \
    | awk '{print $2}')

if [[ -z "$FREE_GPU" ]]; then
    echo "ERROR: No GPU with >= ${MIN_FREE_MIB} MiB free. Current state:"
    nvidia-smi --query-gpu=index,memory.free,memory.used,utilization.gpu \
        --format=csv,noheader
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$FREE_GPU
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $FREE_GPU | tr -d ' ')

# ── Handle existing checkpoint ────────────────────────────────────────────────
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
PRETRAINED_PATH="lerobot/pi05_base"

if [[ -f "${CHECKPOINT_DIR}/config.json" ]]; then
    # Back up existing run, keep only the checkpoint we need
    BACKUP_DIR="outputs/main_prev_$(date +%Y%m%d_%H%M%S)"
    echo "Found existing checkpoint. Backing up to ${BACKUP_DIR}"
    mv "${OUTPUT_DIR}" "${BACKUP_DIR}"

    PRETRAINED_PATH="${BACKUP_DIR}/checkpoints/last/pretrained_model"

    # Ensure compile_model is false
    sed -i 's/"compile_model": true/"compile_model": false/' "${PRETRAINED_PATH}/config.json" 2>/dev/null
    sed -i 's/"compile_model": true/"compile_model": false/' "${PRETRAINED_PATH}/train_config.json" 2>/dev/null

    echo "RESUMING from: ${PRETRAINED_PATH}"
else
    echo "No existing checkpoint. Starting FRESH from lerobot/pi05_base"
fi

echo "============================================="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "GPU:          $FREE_GPU  (${FREE_MIB} MiB free)"
echo "Output dir:   $OUTPUT_DIR"
echo "Dataset:      $DATASET_REPO_ID  @ ${DATASET_ROOT}"
echo "Pretrained:   $PRETRAINED_PATH"
echo "Steps:        $STEPS"
echo "Batch size:   $BATCH_SIZE"
echo "============================================="

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
    --wandb.enable=false \
    --policy.push_to_hub=false \
    --dataset.video_backend=pyav \
    --tolerance_s=0.04

echo ""
echo "Training complete at $(date)"
echo "Output: $OUTPUT_DIR"
echo "Checkpoint: ${OUTPUT_DIR}/checkpoints/last/pretrained_model"