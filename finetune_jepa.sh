#!/usr/bin/env bash

set -euo pipefail

# Apples-to-apples stage-2 entry point for the completed JEPA run.  All task,
# optimizer, scheduler, augmentation, and split defaults are inherited from
# finetune.sh; the model is pinned to the checkpoint's tiny encoder.
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The v7 NumPy trees are staged beside the repository on the faster /scratch
# filesystem.  Explicit environment overrides remain available for relocations.
export DATASET_PATH="${DATASET_PATH:-${repo_dir}/events_v7.0*}"
export METADATA_PATH="${METADATA_PATH:-${repo_dir}/events_v7.0_500_npz/metadata_stats.pkl}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-${repo_dir}/checkpoints_final/pretrain_jepa_latent/loss_total_val/last.ckpt}"
export PRETRAINED_ENCODER_SOURCE="${PRETRAINED_ENCODER_SOURCE:-target}"
export MODEL="${MODEL:-tiny}"
export NAME="${NAME:-finetune_jepa_target}"
export CHECKPOINT_NAME="${CHECKPOINT_NAME:-${NAME}}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints_final}"
export GPUS="${GPUS:-0 1}"
# finetune.sh uses 1024 samples on one GPU.  With two GPUs, 512 per device
# preserves the same global batch, LR scaling, and optimizer-step schedule.
export BATCH_SIZE="${BATCH_SIZE:-512}"

if [[ ! -f "$LOAD_CHECKPOINT" ]]; then
    echo "JEPA checkpoint not found: $LOAD_CHECKPOINT" >&2
    exit 2
fi

exec bash "${repo_dir}/finetune.sh"
