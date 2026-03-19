#!/usr/bin/env bash

set -euo pipefail

manifest="${MANIFEST:-transfer_learning/transfer_pilarnet/manifest_768px_train.npz}"
val_manifest="${VAL_MANIFEST:-transfer_learning/transfer_pilarnet/manifest_768px_val.npz}"
charge_metadata_path="${CHARGE_METADATA_PATH:-transfer_learning/transfer_pilarnet/charge_metadata.pkl}"
model="${MODEL:-base}"
spatial_shape="${SPATIAL_SHAPE:-168 168 180}"
patch_size="${PATCH_SIZE:-12 12 10}"
window_size="${WINDOW_SIZE:-2 2 3}"
num_cls="${NUM_CLS:-2}"
batch_size="${BATCH_SIZE:-64}"
epochs="${EPOCHS:-30}"
blr="${BLR:-1e-2}"
warmup_epochs="${WARMUP_EPOCHS:-10}"
cosine_annealing_epochs="${COSINE_ANNEALING_EPOCHS:-20}"
weight_decay="${WEIGHT_DECAY:-0.05}"
layer_decay="${LAYER_DECAY:-1.0}"
drop_path_rate="${DROP_PATH_RATE:-0.2}"
ema_decay="${EMA_DECAY:-0.9999}"
label_smoothing="${LABEL_SMOOTHING:-0.02}"
meta_dropout="${META_DROPOUT:-0.1}"
context_layers="${CONTEXT_LAYERS:-3}"
context_heads="${CONTEXT_HEADS:-12}"
context_dropout="${CONTEXT_DROPOUT:-0.1}"
min_particles_per_event="${MIN_PARTICLES_PER_EVENT:-1}"
save_dir="${SAVE_DIR:-transfer_learning/transfer_pilarnet/logs_pilarnet}"
name="${NAME:-pilarnet_pid_multi_base_scratch}"
checkpoint_path="${CHECKPOINT_PATH:-transfer_learning/transfer_pilarnet/checkpoints_pilarnet}"
checkpoint_name="${CHECKPOINT_NAME:-$name}"
num_workers="${NUM_WORKERS:-8}"
nb_nodes="${NB_NODES:-1}"
gpus="${GPUS:-0}"

python -m transfer_learning.transfer_pilarnet.train \
    --task_mode multi \
    --manifest "$manifest" \
    --val_manifest "$val_manifest" \
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
    --meta_dropout $meta_dropout \
    --context_layers $context_layers \
    --context_heads $context_heads \
    --context_dropout $context_dropout \
    --min_particles_per_event $min_particles_per_event \
    --save_dir "$save_dir" \
    --name "$name" \
    --checkpoint_path "$checkpoint_path" \
    --checkpoint_name "$checkpoint_name" \
    --num_workers $num_workers \
    --nb_nodes $nb_nodes \
    --gpus $gpus \
    --augment \
    --global_pool
