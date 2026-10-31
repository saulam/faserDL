#!/bin/bash
# One job for a fine-tune and its test-set CSV: clariden/finetune_flash.sbatch,
# then clariden/evaluate_flash.sbatch on the same node, saving the queue wait
# between the two.  The published 20-epoch recipe takes ~41 min on 1 x 4 GH200
# (fine-tune ~35, CSV ~6), so the job also fits the debug partition (at most
# 90 node-minutes per job), which starts within minutes when the normal
# partition is full.
#
# Fine-tuning settings as for finetune_flash.sbatch (LOAD_CHECKPOINT, NAME,
# SEED, LAYER_DECAY, ...).  Submit from the repository root, e.g.
#   LOAD_CHECKPOINT=/capstor/scratch/cscs/salonso/faser/checkpoint/pretrain_x/snapshots/epoch=199.ckpt \
#   NAME=finetune_x_ld085 SEED=7 LAYER_DECAY=0.85 sbatch --partition=debug clariden/finetune_evaluate_flash.sbatch
# Refuses to start if the fine-tuning run or its CSV already exists.
#
#SBATCH --job-name=finetune_evaluate_flash
#SBATCH --account=a0178
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/salonso/faser/slurm/%x_%j.out

set -euo pipefail
cd /users/salonso/faserDL
: "${NAME:?Set NAME to the fine-tuning run name.}"
: "${LOAD_CHECKPOINT?Set LOAD_CHECKPOINT to the pretraining checkpoint (or to an empty string for scratch training).}"
checkpoints="${CHECKPOINT_PATH:-/capstor/scratch/cscs/salonso/faser/checkpoint}"
for path in "$checkpoints/$NAME" "results/results_v7_test_$NAME.csv"; do
    [[ -e "$path" ]] && { echo "$path exists; not overwriting" >&2; exit 2; }
done

echo "$(date '+%F %T') fine-tuning ${LOAD_CHECKPOINT:-from scratch} as $NAME"
bash clariden/finetune_flash.sbatch
echo "$(date '+%F %T') evaluating $NAME"
[[ -n "$LOAD_CHECKPOINT" ]] || export ALLOW_SCRATCH=1
CHECKPOINT_DIR="$checkpoints/$NAME" bash clariden/evaluate_flash.sbatch
echo "$(date '+%F %T') done"
