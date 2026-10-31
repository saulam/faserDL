#!/bin/bash
# One job for a whole pretraining run: flash-JEPA pretraining on 2 nodes
# (clariden/pretrain_flash.sbatch), then the published 20-epoch fine-tune of
# its last.ckpt on the first node (clariden/finetune_flash.sbatch: 1 x 4 GH200,
# global batch 1,024, as the separate fine-tuning jobs) and the test-set CSV
# (clariden/evaluate_flash.sbatch).  This saves the 1-4 h queue waits between
# the three steps; the second node idles during the ~40 min of the last two.
#
# Pretraining settings as for pretrain_flash.sbatch (NAME, SEED, TARGET_LAYERS,
# JEPA_EMA_DECAY, ...).  None of them reach the fine-tune, which takes only
# FT_NAME (required), FT_SEED (default 7) and FT_LAYER_DECAY (default 0.85);
# everything else is finetune_flash.sh's default recipe.  Submit from the
# repository root, e.g.
#   NAME=pretrain_x SEED=9 TARGET_LAYERS=16 FT_NAME=finetune_x_ld085 sbatch clariden/pretrain_finetune_flash.sbatch
# Refuses to start if the fine-tuning run or its CSV already exists.  If the
# job is cut short, resubmitting it resumes the pretraining from last.ckpt.
#
#SBATCH --job-name=pretrain_finetune_flash
#SBATCH --account=a0178
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=06:30:00
#SBATCH --output=/capstor/scratch/cscs/salonso/faser/slurm/%x_%j.out

set -euo pipefail
cd /users/salonso/faserDL
: "${NAME:?Set NAME to the pretraining run name.}"
: "${FT_NAME:?Set FT_NAME to the fine-tuning run name.}"
checkpoints="${CHECKPOINT_PATH:-/capstor/scratch/cscs/salonso/faser/checkpoint}"
for path in "$checkpoints/$FT_NAME" "results/results_v7_test_$FT_NAME.csv"; do
    [[ -e "$path" ]] && { echo "$path exists; not overwriting" >&2; exit 2; }
done

echo "$(date '+%F %T') pretraining $NAME"
bash clariden/pretrain_flash.sbatch
pretrained="$checkpoints/${CHECKPOINT_NAME:-$NAME}/loss_total_val/last.ckpt"
[[ -f "$pretrained" ]] || { echo "no $pretrained after pretraining" >&2; exit 3; }

# The variables both launchers read, so that no pretraining setting
# (SEED, BLR, EPOCHS, EFFECTIVE_BATCH, ...) changes the fine-tuning recipe.
shared=(ACCUM_GRAD_BATCHES AHCAL_PATCH_SIZE BATCH_SIZE BETA1 BETA2 BLR CHECKPOINT_NAME
        COSINE_ANNEALING_EPOCHS DROPOUT DROP_PATH_RATE EARLY_STOP_PATIENCE EFFECTIVE_BATCH EPOCHS
        EPS FCAL_PATCH_SIZE GPUS LOAD_CHECKPOINT LOG_EVERY_N_STEPS MODEL NAME NB_NODES NUM_WORKERS
        PREPROCESSING_INPUT PREPROCESSING_OUTPUT RESUME_CHECKPOINT SAVE_TOP_K SEED WARMUP_EPOCHS
        WEIGHT_DECAY)
echo "$(date '+%F %T') fine-tuning $pretrained as $FT_NAME"
(
    unset "${shared[@]}"
    export NAME="$FT_NAME" SEED="${FT_SEED:-7}" LAYER_DECAY="${FT_LAYER_DECAY:-0.85}"
    export LOAD_CHECKPOINT="$pretrained" FT_NODES=1
    if [[ -n "${FT_EPOCHS:-}" ]]; then  # shorter fine-tune, for tests of this script only
        export EPOCHS="$FT_EPOCHS" WARMUP_EPOCHS=0 COSINE_ANNEALING_EPOCHS="$FT_EPOCHS"
    fi
    bash clariden/finetune_flash.sbatch
)
echo "$(date '+%F %T') evaluating $FT_NAME"
(
    unset "${shared[@]}"
    export CHECKPOINT_DIR="$checkpoints/$FT_NAME"
    bash clariden/evaluate_flash.sbatch
)
echo "$(date '+%F %T') done"
