#!/usr/bin/env bash

set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the scintillator dataset root.}"

split="${SPLIT:-training}"
out="${OUT:-transfer_learning/transfer_scintillator/charge_metadata.pkl}"
batch_size="${BATCH_SIZE:-512}"
num_workers="${NUM_WORKERS:-16}"
spatial_shape="${SPATIAL_SHAPE:-120 120 120}"

python -m transfer_learning.build_transfer_charge_metadata \
    --dataset scintillator \
    --data_dir "$DATA_DIR" \
    --split "$split" \
    --out "$out" \
    --scintillator_spatial_shape $spatial_shape \
    --batch_size $batch_size \
    --num_workers $num_workers
