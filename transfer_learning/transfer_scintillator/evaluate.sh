#!/usr/bin/env bash

set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the scintillator dataset root.}"
: "${CHECKPOINT:?Set CHECKPOINT to a checkpoint file or directory.}"

charge_metadata_path="${CHARGE_METADATA_PATH:-transfer_learning/transfer_scintillator/charge_metadata.pkl}"
model="${MODEL:-base}"
spatial_shape="${SPATIAL_SHAPE:-120 120 120}"
patch_size="${PATCH_SIZE:-12 12 10}"
window_size="${WINDOW_SIZE:-2 2 2}"
num_cls="${NUM_CLS:-2}"
batch_size="${BATCH_SIZE:-1024}"
num_workers="${NUM_WORKERS:-16}"
gpu="${GPU:-1}"
split="${SPLIT:-testing}"
output="${OUTPUT:-transfer_learning/transfer_scintillator/evaluation_results.csv}"

python -m transfer_learning.transfer_scintillator.evaluate \
    --data_dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --charge_metadata_path "$charge_metadata_path" \
    --model $model \
    --spatial_shape $spatial_shape \
    --patch_size $patch_size \
    --window_size $window_size \
    --num_cls $num_cls \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --gpu $gpu \
    --split "$split" \
    --global_pool \
    --output "$output"
