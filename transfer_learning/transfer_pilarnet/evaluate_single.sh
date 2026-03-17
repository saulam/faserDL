#!/usr/bin/env bash

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to a checkpoint file or directory.}"

manifest="${MANIFEST:-transfer_learning/transfer_pilarnet/manifest_768px_test.npz}"
charge_metadata_path="${CHARGE_METADATA_PATH:-transfer_learning/transfer_pilarnet/charge_metadata.pkl}"
model="${MODEL:-base}"
spatial_shape="${SPATIAL_SHAPE:-168 168 180}"
patch_size="${PATCH_SIZE:-12 12 10}"
window_size="${WINDOW_SIZE:-2 2 3}"
num_cls="${NUM_CLS:-2}"
batch_size="${BATCH_SIZE:-256}"
num_workers="${NUM_WORKERS:-8}"
meta_dropout="${META_DROPOUT:-0.1}"
gpu="${GPU:-0}"

python -m transfer_learning.transfer_pilarnet.evaluate \
    --manifest "$manifest" \
    --checkpoint "$CHECKPOINT" \
    --charge_metadata_path "$charge_metadata_path" \
    --model $model \
    --spatial_shape $spatial_shape \
    --patch_size $patch_size \
    --window_size $window_size \
    --num_cls $num_cls \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --meta_dropout $meta_dropout \
    --single_particle_meta \
    --gpu $gpu \
    --global_pool
