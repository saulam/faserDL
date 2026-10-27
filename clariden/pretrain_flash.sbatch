#!/bin/bash
# Multi-node flash-attention JEPA pretraining on Clariden (GH200, 4 GPUs/node).
#
# Submit from the repository root, e.g.
#   sbatch clariden/pretrain_flash.sbatch
#   sbatch --nodes=4 clariden/pretrain_flash.sbatch
#   NAME=my_run EPOCHS=100 WARMUP_EPOCHS=10 sbatch clariden/pretrain_flash.sbatch
#   NAME=pretrain_flash_jepa_base_d2v16 TARGET_LAYERS=16 sbatch clariden/pretrain_flash.sbatch
#   NAME=pretrain_flash_jepa_base_d2v16_bn TARGET_LAYERS=16 TARGET_NORM=batch sbatch clariden/pretrain_flash.sbatch
# Any pretrain_flash.sh variable can be set in the environment like this;
# sbatch options (nodes, time, partition, account) on the sbatch command line.
#
# The effective batch stays EFFECTIVE_BATCH (default 4,096, as on dlnu and in
# the published MAE): the per-GPU batch is EFFECTIVE_BATCH / (4 x nodes), so
# 2 nodes give 512 per GPU and 4 nodes 256.  The LR follows by linear scaling.
# The job resumes from <checkpoint>/<name>/loss_total_val/last.ckpt when it
# exists (time limit, node failure, `sbatch --array=1-3%1 ...` chains), and
# refuses to start unless the train/val/test split matches dlnu's.
#
#SBATCH --job-name=pretrain_flash
#SBATCH --account=a0178
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/salonso/faser/slurm/%x_%j.out

set -euo pipefail
cd /users/salonso/faserDL
# The uenv is loaded per job step (srun --uenv): loading its view in the batch
# shell breaks srun itself (its OpenSSL shadows the system one).
export UENV="${UENV:-pytorch/v2.9.1:v2}"
export VENV="${VENV:-/users/salonso/venvs/neutrino-jepa-pt2.9}"

export DATASET_PATH="${DATASET_PATH:-/iopsstor/scratch/cscs/salonso/faser_new/events_v7.0*}"
export METADATA_PATH="${METADATA_PATH:-/iopsstor/scratch/cscs/salonso/faser_new/events_v7.0_500_npz/metadata_stats.pkl}"
# Split of the 1,226,375 dlnu events (python -m train.split_fingerprint); set to
# an empty string to skip the check, e.g. for tests on a partial copy.
export EXPECTED_SPLIT_SHA256="${EXPECTED_SPLIT_SHA256-02078f72b5ec271f81e14f3a24b769e665a9a928908b141bd1fb2134c730fd7c}"
# Stage the slim dataset (python -m dataset.slim_shards; only the arrays used
# here, same event names) into each node's memory: the job then reads RAM
# instead of ~1.2M small Lustre files per epoch, which limited 2 nodes to
# ~11k events/s.  SLIM_SHARDS="" reads DATASET_PATH directly.
export SLIM_SHARDS="${SLIM_SHARDS-/iopsstor/scratch/cscs/salonso/faser_slim_v7.0}"
if [[ -n "$SLIM_SHARDS" ]]; then
    staged="/dev/shm/$USER/faser_slim"
    trap 'srun --ntasks-per-node=1 --ntasks="$SLURM_NNODES" rm -rf "$staged" || true' EXIT
    srun --ntasks-per-node=1 --ntasks="$SLURM_NNODES" --cpus-per-task="$SLURM_CPUS_ON_NODE" \
        bash clariden/stage_slim_dataset.sh "$SLIM_SHARDS" "$staged"
    export DATASET_PATH="$staged/events_v7.0*"
fi
echo "dataset: $DATASET_PATH${SLIM_SHARDS:+ (staged from $SLIM_SHARDS)}"
export NAME="${NAME:-pretrain_flash_jepa_base}"
export CHECKPOINT_NAME="${CHECKPOINT_NAME:-$NAME}"
export SAVE_DIR="${SAVE_DIR:-/capstor/scratch/cscs/salonso/faser/logs}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/capstor/scratch/cscs/salonso/faser/checkpoint}"
export NUM_WORKERS="${NUM_WORKERS:-48}"           # per GPU (its task has 72 Grace cores)
export SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-25}"
export GPUS="0 1 2 3"
export NB_NODES="$SLURM_NNODES"

gpus_total=$((SLURM_NNODES * 4))
export ACCUM_GRAD_BATCHES="${ACCUM_GRAD_BATCHES:-1}"
effective_batch="${EFFECTIVE_BATCH:-4096}"
if [[ -z "${BATCH_SIZE:-}" ]]; then
    if (( effective_batch % (gpus_total * ACCUM_GRAD_BATCHES) )); then
        echo "EFFECTIVE_BATCH=$effective_batch is not divisible by $gpus_total GPUs x $ACCUM_GRAD_BATCHES" >&2
        exit 2
    fi
    export BATCH_SIZE=$((effective_batch / (gpus_total * ACCUM_GRAD_BATCHES)))
fi
echo "nodes=$SLURM_NNODES gpus=$gpus_total batch/GPU=$BATCH_SIZE accumulation=$ACCUM_GRAD_BATCHES" \
     "effective batch=$((gpus_total * BATCH_SIZE * ACCUM_GRAD_BATCHES)) workers/GPU=$NUM_WORKERS"

last="$CHECKPOINT_PATH/$CHECKPOINT_NAME/loss_total_val/last.ckpt"
if [[ -z "${RESUME_CHECKPOINT:-}" && "${AUTO_RESUME:-1}" == "1" && -f "$last" ]]; then
    export RESUME_CHECKPOINT="$last"
    echo "resuming from $RESUME_CHECKPOINT"
fi

export OMP_NUM_THREADS=1            # loader workers are processes; avoid thread oversubscription
export PYTHONUNBUFFERED=1
srun --uenv="$UENV" --view=default --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash -c 'source "$VENV/bin/activate" && exec bash pretrain_flash.sh'
