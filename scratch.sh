#!/usr/bin/env bash

set -euo pipefail

: "${DATASET_PATH:?Set DATASET_PATH to the training dataset path or glob.}"
: "${METADATA_PATH:?Set METADATA_PATH to the metadata statistics pickle.}"

shardshuffle="${SHARDSHUFFLE:-200}"
shuffle="${SHUFFLE:-2000}"
model="${MODEL:-base}"
eps="${EPS:-1e-8}"
batch_size="${BATCH_SIZE:-1024}"
mixup_alpha="${MIXUP_ALPHA:-0.0}"
preprocessing_input="${PREPROCESSING_INPUT:-log}"
preprocessing_output="${PREPROCESSING_OUTPUT:-log}"
label_smoothing="${LABEL_SMOOTHING:-0.02}"
dropout="${DROPOUT:-0.0}"
attn_dropout="${ATTN_DROPOUT:-0.0}"
drop_path_rate="${DROP_PATH_RATE:-0.2}"
kendall_w_max="${KENDALL_W_MAX:-5.0}"
cls_lr_scale="${CLS_LR_SCALE:-1.0}"
head_dropout_cls="${HEAD_DROPOUT_CLS:-0.0}"
head_dropout_reg="${HEAD_DROPOUT_REG:-0.0}"
epochs="${EPOCHS:-40}"
num_workers="${NUM_WORKERS:-16}"
blr="${BLR:-1e-3}"
layer_decay="${LAYER_DECAY:-1.0}"
accum_grad_batches="${ACCUM_GRAD_BATCHES:-1}"
warmup_epochs="${WARMUP_EPOCHS:-5}"
cosine_annealing_epochs="${COSINE_ANNEALING_EPOCHS:-35}"
weight_decay="${WEIGHT_DECAY:-0.05}"
beta1="${BETA1:-0.9}"
beta2="${BETA2:-0.999}"
ema_decay="${EMA_DECAY:-0.9999}"
head_init="${HEAD_INIT:-2e-5}"
save_dir="${SAVE_DIR:-logs_final}"
name="${NAME:-finetune_v7.0_scratch_paper}"
log_every_n_steps="${LOG_EVERY_N_STEPS:-10}"
save_top_k="${SAVE_TOP_K:-10}"
checkpoint_path="${CHECKPOINT_PATH:-checkpoints}"
checkpoint_name="${CHECKPOINT_NAME:-$name}"
early_stop_patience="${EARLY_STOP_PATIENCE:-15}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
nb_nodes="${NB_NODES:-1}"
read -r -a gpus <<< "${GPUS:-0}"

python -m train.finetune \
    --train \
    --stage2 \
    --augmentations_enabled \
    --dataset_path "$DATASET_PATH" \
    --metadata_path "$METADATA_PATH" \
    --model $model \
    --eps $eps \
    --mixup_alpha $mixup_alpha \
    --batch_size $batch_size \
    --preprocessing_input $preprocessing_input \
    --preprocessing_output $preprocessing_output \
    --label_smoothing $label_smoothing \
    --dropout $dropout \
    --drop_path_rate $drop_path_rate \
    --kendall_w_max $kendall_w_max \
    --cls_lr_scale $cls_lr_scale \
    --head_dropout_cls $head_dropout_cls \
    --head_dropout_reg $head_dropout_reg \
    --epochs $epochs \
    --num_workers $num_workers \
    --blr $blr \
    --layer_decay $layer_decay \
    --accum_grad_batches $accum_grad_batches \
    --warmup_epochs $warmup_epochs \
    --cosine_annealing_epochs $cosine_annealing_epochs \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --ema_decay $ema_decay \
    --head_init $head_init \
    --save_dir $save_dir \
    --name $name \
    --log_every_n_steps $log_every_n_steps \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --early_stop_patience $early_stop_patience \
    --nb_nodes $nb_nodes \
    ${resume_checkpoint:+--resume_checkpoint "$resume_checkpoint"} \
    --gpus "${gpus[@]}"
