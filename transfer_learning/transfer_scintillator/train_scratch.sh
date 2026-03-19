#!/usr/bin/env bash

set -euo pipefail

: "${DATA_DIR:?Set DATA_DIR to the scintillator dataset root.}"

charge_metadata_path="${CHARGE_METADATA_PATH:-transfer_learning/transfer_scintillator/charge_metadata.pkl}"
model="${MODEL:-base}"
spatial_shape="${SPATIAL_SHAPE:-120 120 120}"
patch_size="${PATCH_SIZE:-12 12 10}"
window_size="${WINDOW_SIZE:-2 2 2}"
num_cls="${NUM_CLS:-2}"
batch_size="${BATCH_SIZE:-1024}"
epochs="${EPOCHS:-30}"
blr="${BLR:-5e-4}"
warmup_epochs="${WARMUP_EPOCHS:-5}"
cosine_annealing_epochs="${COSINE_ANNEALING_EPOCHS:-25}"
weight_decay="${WEIGHT_DECAY:-0.05}"
layer_decay="${LAYER_DECAY:-1.0}"
drop_path_rate="${DROP_PATH_RATE:-0.2}"
ema_decay="${EMA_DECAY:-0.9999}"
label_smoothing="${LABEL_SMOOTHING:-0.02}"
save_dir="${SAVE_DIR:-transfer_learning/transfer_scintillator/logs_scintillator}"
name="${NAME:-scintillator_pid_base_scratch}"
checkpoint_path="${CHECKPOINT_PATH:-transfer_learning/transfer_scintillator/checkpoints_scintillator}"
checkpoint_name="${CHECKPOINT_NAME:-$name}"
num_workers="${NUM_WORKERS:-16}"
nb_nodes="${NB_NODES:-1}"
gpus="${GPUS:-1}"

python -m transfer_learning.transfer_scintillator.train \
    --data_dir "$DATA_DIR" \
    --charge_metadata_path "$charge_metadata_path" \
    --model $model \
    --spatial_shape $spatial_shape \
    --patch_size $patch_size \
    --window_size $window_size \
    --num_cls $num_cls \
    --batch_size $batch_size \
    --epochs $epochs \
    --blr $blr \
    --warmup_epochs $warmup_epochs \
    --cosine_annealing_epochs $cosine_annealing_epochs \
    --weight_decay $weight_decay \
    --layer_decay $layer_decay \
    --drop_path_rate $drop_path_rate \
    --ema_decay $ema_decay \
    --label_smoothing $label_smoothing \
    --save_dir "$save_dir" \
    --name "$name" \
    --checkpoint_path "$checkpoint_path" \
    --checkpoint_name "$checkpoint_name" \
    --num_workers $num_workers \
    --nb_nodes $nb_nodes \
    --gpus $gpus \
    --augment \
    --global_pool
