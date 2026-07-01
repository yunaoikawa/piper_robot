#!/bin/bash
#SBATCH --job-name=act_finetune
#SBATCH --output=logs/act_finetune_%j.out
#SBATCH --error=logs/act_finetune_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
# NOTE: this node does NOT schedule GPUs via Slurm gres (AllocTRES has no gpu);
# every job sees all 8 A100s and the nvidia-smi scan below self-selects a free
# one (>= MIN_FREE_MIB). So we only need to request CPU/mem here. `--gpus` is
# kept for portability but is effectively a no-op on this cluster.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
# Usage:
#   bash src/train_act.sh
#
# ACT (Action Chunking Transformer).
# Latest checkpoint always at: outputs/lab/checkpoints/last/pretrained_model

# ── Setup ─────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs

cd /home/yoikawa/src/robot
source .venv/bin/activate

# ── Mode toggle ───────────────────────────────────────────────────────────────
# "scratch" : always start from scratch (back up any existing output dir first)
# "resume"  : back up existing checkpoint and resume from it
TRAIN_MODE="scratch"          # "scratch" | "resume"

# ── Configure ─────────────────────────────────────────────────────────────────
DATASET_REPO_ID="yoikawa/lab_act"
# Optional positional args: $1=DATASET_ROOT  $2=OUTPUT_DIR (defaults below)
DATASET_ROOT="${1:-data/train/after0610/liquid}"
OUTPUT_DIR="${2:-outputs/lab/act/after0610/liquid}"
JOB_NAME=${OUTPUT_DIR}

STEPS=100000

BATCH_SIZE=16          # per GPU
NUM_WORKERS=4
SAVE_FREQ=10000
NUM_GPUS=1
MIN_FREE_MIB=20000

# ACT-specific
CHUNK_SIZE=100
N_ACTION_STEPS=100
KL_WEIGHT=10.0
LR=1e-5
LR_BACKBONE=1e-5
WEIGHT_DECAY=1e-4
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
export HF_DATASETS_OFFLINE=1

# ── Handle existing output dir / checkpoint ───────────────────────────────────
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
RESUME_PATH=""

case "$TRAIN_MODE" in
    scratch)
        if [[ -d "${OUTPUT_DIR}" ]]; then
            BACKUP_DIR="outputs/lab_prev_$(date +%Y%m%d_%H%M%S)"
            echo "[scratch] Output dir exists. Backing up to ${BACKUP_DIR}"
            mv "${OUTPUT_DIR}" "${BACKUP_DIR}"
        fi
        echo "[scratch] Training ACT from SCRATCH."
        ;;
    resume)
        if [[ -f "${CHECKPOINT_DIR}/config.json" ]]; then
            BACKUP_DIR="outputs/lab_prev_$(date +%Y%m%d_%H%M%S)"
            echo "[resume] Found existing checkpoint. Backing up to ${BACKUP_DIR}"
            mv "${OUTPUT_DIR}" "${BACKUP_DIR}"
            RESUME_PATH="${BACKUP_DIR}/checkpoints/last/pretrained_model"
            echo "[resume] RESUMING from: ${RESUME_PATH}"
        else
            echo "[resume] No existing checkpoint. Training ACT from SCRATCH."
        fi
        ;;
    *)
        echo "ERROR: Unknown TRAIN_MODE='${TRAIN_MODE}' (expected 'scratch' or 'resume')."
        exit 1
        ;;
esac

echo "============================================="
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Node:         $(hostname)"
echo "Mode:         ${TRAIN_MODE}"
echo "GPUs:         ${FREE_GPUS} (${NUM_GPUS} GPUs)"
echo "Batch size:   ${BATCH_SIZE}/GPU × ${NUM_GPUS} = $((BATCH_SIZE * NUM_GPUS)) effective"
echo "Output dir:   $OUTPUT_DIR"
echo "Dataset:      $DATASET_REPO_ID  @ ${DATASET_ROOT}"
echo "Steps:        $STEPS"
echo "Chunk size:   $CHUNK_SIZE"
echo "============================================="

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(lerobot-train
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --policy.type=act
    --output_dir="${OUTPUT_DIR}"
    --job_name="${JOB_NAME}"
    --policy.chunk_size=${CHUNK_SIZE}
    --policy.n_action_steps=${N_ACTION_STEPS}
    --policy.use_vae=true
    --policy.kl_weight=${KL_WEIGHT}
    --policy.dim_model=512
    --policy.n_heads=8
    --policy.dim_feedforward=3200
    --policy.n_encoder_layers=4
    --policy.n_decoder_layers=1
    --policy.vision_backbone=resnet18
    --policy.replace_final_stride_with_dilation=false
    --policy.pre_norm=false
    --policy.dropout=0.1
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "MEAN_STD"}'
    --policy.device=cuda
    --steps=${STEPS}
    --batch_size=${BATCH_SIZE}
    --num_workers=${NUM_WORKERS}
    --save_freq=${SAVE_FREQ}
    --wandb.enable=true
    --policy.push_to_hub=false
    --dataset.video_backend=pyav
    --use_policy_training_preset=false
    --optimizer.type=adamw
    --optimizer.lr=${LR}
    --optimizer.weight_decay=${WEIGHT_DECAY}
    --optimizer.grad_clip_norm=10.0
    --policy.optimizer_lr_backbone=${LR_BACKBONE}
    --scheduler.type=cosine_decay_with_warmup
    --scheduler.num_warmup_steps=500
    --scheduler.peak_lr=${LR}
    --scheduler.num_decay_steps=${STEPS}
    --scheduler.decay_lr=1e-7
    --tolerance_s=0.04
)

# Only resume mode injects a pretrained checkpoint.
[[ -n "$RESUME_PATH" ]] && CMD+=(--policy.pretrained_path="${RESUME_PATH}")

"${CMD[@]}"

echo ""
echo "Training complete at $(date)"
echo "Output: $OUTPUT_DIR"
echo "Checkpoint: ${OUTPUT_DIR}/checkpoints/last/pretrained_model"