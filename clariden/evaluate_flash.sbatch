#!/bin/bash
# Test-set CSV of a fine-tuning run on Clariden (one GPU), with the same split,
# per-task best-validation-loss checkpoints and schema as on dlnu.
#
#   CHECKPOINT_DIR=/capstor/scratch/cscs/salonso/faser/checkpoint/finetune_flash_jepa_base \
#   sbatch clariden/evaluate_flash.sbatch
#
# Writes results/results_v7_test_<run>.csv (+ manifest and metadata) in the
# repository unless OUT is set; MAX_BATCHES=N gives a quick partial CSV.
#
#SBATCH --job-name=evaluate_flash
#SBATCH --account=a0178
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/salonso/faser/slurm/%x_%j.out

set -euo pipefail
cd /users/salonso/faserDL
# The uenv is loaded per job step (srun --uenv): loading its view in the batch
# shell breaks srun itself (its OpenSSL shadows the system one).
export UENV="${UENV:-pytorch/v2.9.1:v2}"
export VENV="${VENV:-/users/salonso/venvs/neutrino-jepa-pt2.9}"

export DATASET_PATH="${DATASET_PATH:-/iopsstor/scratch/cscs/salonso/faser_new/events_v7.0*}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to <checkpoint path>/<fine-tuning run name>.}"
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

export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
srun --uenv="$UENV" --view=default --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash -c 'source "$VENV/bin/activate" && exec python -m evaluation.create_csv_v7_ft "$@"' _ \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --log_root "${LOG_ROOT:-/capstor/scratch/cscs/salonso/faser/logs/logs}" \
    --dataset_path "$DATASET_PATH" \
    --num_workers "${NUM_WORKERS:-48}" \
    --batch_size "${EVAL_BATCH_SIZE:-256}" \
    ${OUT:+--out "$OUT"} \
    ${MAX_BATCHES:+--max_batches "$MAX_BATCHES"}
