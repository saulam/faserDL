#!/usr/bin/env bash

set -euo pipefail

manifest="${MANIFEST:-transfer_learning/transfer_pilarnet/manifest_768px_train.npz}"
out="${OUT:-transfer_learning/transfer_pilarnet/charge_metadata.pkl}"
batch_size="${BATCH_SIZE:-64}"
num_workers="${NUM_WORKERS:-8}"
charge_bin_width="${CHARGE_BIN_WIDTH:-0.1}"

python -m transfer_learning.build_transfer_charge_metadata \
    --dataset pilarnet \
    --manifest "$manifest" \
    --out "$out" \
    --batch_size $batch_size \
    --num_workers $num_workers \
    --charge_bin_width $charge_bin_width
