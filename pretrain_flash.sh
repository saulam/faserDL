#!/usr/bin/env bash
# I-JEPA-style pretraining of the spconv-free flash-attention encoder
# (train/pretrain_flash.py).  Run inside the neutrino-jepa environment.
# Defaults reproduce the controlled comparison of the published MAE and the
# patch-level JEPA runs: base-size encoder, effective batch 4,096, blr 1e-4,
# 400 epochs (40 warm-up).  Without spconv's voxel limit the 4,096 fit as
# 2 GPUs x 2,048 without accumulation, the fastest measured configuration
# (~5% faster than 1,024 x 2; ~68 GB of an 80 GB A100).  24 loader workers per
# GPU saturate the CPUs; more workers were slower.

set -euo pipefail

: "${DATASET_PATH:?Set DATASET_PATH to the training dataset path.}"
: "${METADATA_PATH:?Set METADATA_PATH to the metadata statistics pickle.}"

model="${MODEL:-base}"
eps="${EPS:-1e-8}"
batch_size="${BATCH_SIZE:-2048}"
preprocessing_input="${PREPROCESSING_INPUT:-log}"
preprocessing_output="${PREPROCESSING_OUTPUT:-log}"
dropout="${DROPOUT:-0.0}"
drop_path_rate="${DROP_PATH_RATE:-0.0}"
num_workers="${NUM_WORKERS:-24}"
blr="${BLR:-1e-4}"
accum_grad_batches="${ACCUM_GRAD_BATCHES:-1}"
epochs="${EPOCHS:-400}"
warmup_epochs="${WARMUP_EPOCHS:-40}"
cosine_annealing_epochs="${COSINE_ANNEALING_EPOCHS:-$((epochs - warmup_epochs))}"
weight_decay="${WEIGHT_DECAY:-0.05}"
beta1="${BETA1:-0.9}"
beta2="${BETA2:-0.95}"
jepa_ema_decay="${JEPA_EMA_DECAY:-0.996}"
jepa_ema_end_decay="${JEPA_EMA_END_DECAY:-1.0}"
fcal_patch_size="${FCAL_PATCH_SIZE:-12 12 10}"
ahcal_patch_size="${AHCAL_PATCH_SIZE:-6 6 5}"
predictor_embed_dim="${PREDICTOR_EMBED_DIM:-256}"
predictor_depth="${PREDICTOR_DEPTH:-6}"
predictor_num_heads="${PREDICTOR_NUM_HEADS:-4}"
mask_num_blocks="${MASK_NUM_BLOCKS:-4}"
mask_block_scale="${MASK_BLOCK_SCALE:-0.15 0.2}"
mask_block_aspect="${MASK_BLOCK_ASPECT:-0.5 2.0}"
mask_global_prob="${MASK_GLOBAL_PROB:-0.25}"
save_dir="${SAVE_DIR:-logs_final}"
name="${NAME:-pretrain_flash_jepa_base}"
log_every_n_steps="${LOG_EVERY_N_STEPS:-50}"
# The EMA target moves, so validation losses of different epochs are not
# comparable: keep only last.ckpt and never stop early by default.
save_top_k="${SAVE_TOP_K:-0}"
early_stop_patience="${EARLY_STOP_PATIENCE:-0}"
checkpoint_path="${CHECKPOINT_PATH:-checkpoints_final}"
checkpoint_name="${CHECKPOINT_NAME:-$name}"
load_checkpoint="${LOAD_CHECKPOINT:-}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
seed="${SEED:-7}"
nb_nodes="${NB_NODES:-1}"
read -r -a gpus <<< "${GPUS:-0 1}"
read -r -a fcal_patch <<< "$fcal_patch_size"
read -r -a ahcal_patch <<< "$ahcal_patch_size"
read -r -a block_scale <<< "$mask_block_scale"
read -r -a block_aspect <<< "$mask_block_aspect"
fast_dev_run="${FAST_DEV_RUN:-0}"

if (( warmup_epochs < 0 || cosine_annealing_epochs < 0 )); then
    echo "Warm-up and cosine-annealing epochs must be non-negative." >&2
    exit 2
fi

fast_dev_run_args=()
if [[ "$fast_dev_run" == "1" || "$fast_dev_run" == "true" ]]; then
    fast_dev_run_args+=(--fast_dev_run)
fi

python -m train.pretrain_flash \
    --train \
    --stage2 \
    --augmentations_enabled \
    --dataset_path "$DATASET_PATH" \
    --metadata_path "$METADATA_PATH" \
    --model $model \
    --eps $eps \
    --batch_size $batch_size \
    --preprocessing_input $preprocessing_input \
    --preprocessing_output $preprocessing_output \
    --dropout $dropout \
    --drop_path_rate $drop_path_rate \
    --epochs $epochs \
    --num_workers $num_workers \
    --blr $blr \
    --accum_grad_batches $accum_grad_batches \
    --warmup_epochs $warmup_epochs \
    --cosine_annealing_epochs $cosine_annealing_epochs \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --jepa_ema_decay $jepa_ema_decay \
    --jepa_ema_end_decay $jepa_ema_end_decay \
    --flash_fcal_patch_size "${fcal_patch[@]}" \
    --flash_ahcal_patch_size "${ahcal_patch[@]}" \
    --flash_predictor_embed_dim $predictor_embed_dim \
    --flash_predictor_depth $predictor_depth \
    --flash_predictor_num_heads $predictor_num_heads \
    --flash_mask_num_blocks $mask_num_blocks \
    --flash_mask_block_scale "${block_scale[@]}" \
    --flash_mask_block_aspect "${block_aspect[@]}" \
    --flash_mask_global_prob $mask_global_prob \
    --save_dir $save_dir \
    --name $name \
    --log_every_n_steps $log_every_n_steps \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --early_stop_patience $early_stop_patience \
    --seed $seed \
    --nb_nodes $nb_nodes \
    --gpus "${gpus[@]}" \
    ${load_checkpoint:+--load_checkpoint "$load_checkpoint"} \
    ${resume_checkpoint:+--resume_checkpoint "$resume_checkpoint"} \
    "${fast_dev_run_args[@]}"
