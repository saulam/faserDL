#!/usr/bin/env bash

set -euo pipefail

: "${DATASET_PATH:?Set DATASET_PATH to the training dataset path or glob.}"
: "${METADATA_PATH:?Set METADATA_PATH to the metadata statistics pickle.}"

shardshuffle="${SHARDSHUFFLE:-200}"
shuffle="${SHUFFLE:-4000}"
model="${MODEL:-tiny}"
eps="${EPS:-1e-8}"
batch_size="${BATCH_SIZE:-512}"
preprocessing_input="${PREPROCESSING_INPUT:-log}"
label_smoothing="${LABEL_SMOOTHING:-0.02}"
dropout="${DROPOUT:-0.0}"
attn_dropout="${ATTN_DROPOUT:-0.0}"
drop_path_rate="${DROP_PATH_RATE:-0.0}"
dropout_dec="${DROPOUT_DEC:-0.0}"
attn_dropout_dec="${ATTN_DROPOUT_DEC:-0.0}"
drop_path_rate_dec="${DROP_PATH_RATE_DEC:-0.0}"
mask_ratio="${MASK_RATIO:-0.75}"
epochs="${EPOCHS:-400}"
num_workers="${NUM_WORKERS:-16}"
blr="${BLR:-1.5e-4}"
accum_grad_batches="${ACCUM_GRAD_BATCHES:-2}"
warmup_epochs="${WARMUP_EPOCHS:-40}"
cosine_annealing_epochs="${COSINE_ANNEALING_EPOCHS:-360}"
weight_decay="${WEIGHT_DECAY:-0.05}"
beta1="${BETA1:-0.9}"
beta2="${BETA2:-0.95}"
save_dir="${SAVE_DIR:-logs_final}"
name="${NAME:-pretrain_v7.0_distance_v1}"
log_every_n_steps="${LOG_EVERY_N_STEPS:-50}"
save_top_k="${SAVE_TOP_K:-1}"
checkpoint_path="${CHECKPOINT_PATH:-checkpoints_final}"
checkpoint_name="${CHECKPOINT_NAME:-$name}"
early_stop_patience="${EARLY_STOP_PATIENCE:-200}"
load_checkpoint="${LOAD_CHECKPOINT:-}"
resume_checkpoint="${RESUME_CHECKPOINT:-}"
nb_nodes="${NB_NODES:-1}"
read -r -a gpus <<< "${GPUS:-0 1 2 3}"
reconstruction_loss_mode="${RECONSTRUCTION_LOSS_MODE:-hybrid}"
reconstruction_chamfer_weight="${RECONSTRUCTION_CHAMFER_WEIGHT:-0.3}"
reconstruction_distance_reg_weight="${RECONSTRUCTION_DISTANCE_REG_WEIGHT:-0.3}"
reconstruction_max_distance_fcal="${RECONSTRUCTION_MAX_DISTANCE_FCAL:-5.0}"
reconstruction_max_distance_ahcal="${RECONSTRUCTION_MAX_DISTANCE_AHCAL:-3.0}"
reconstruction_gamma_distance="${RECONSTRUCTION_GAMMA_DISTANCE:-2.0}"
relational_loss_mode="${RELATIONAL_LOSS_MODE:-standard}"
relational_distance_weight="${RELATIONAL_DISTANCE_WEIGHT:-0.3}"
relational_max_distance="${RELATIONAL_MAX_DISTANCE:-3.0}"
relational_gamma_distance="${RELATIONAL_GAMMA_DISTANCE:-2.0}"
relational_pass_prob="${RELATIONAL_PASS_PROB:-0.5}"
relational_mask_ratio="${RELATIONAL_MASK_RATIO:-0.25}"
relational_voxel_keep_prob="${RELATIONAL_VOXEL_KEEP_PROB:-0.5}"
relational_pass_seed="${RELATIONAL_PASS_SEED:-42}"
sparse_ecal="${SPARSE_ECAL:-auto}"

sparse_ecal_flag=""
if [[ "$sparse_ecal" == "1" || "$sparse_ecal" == "true" || "$sparse_ecal" == "TRUE" ]]; then
    sparse_ecal_flag="--sparse_ecal"
elif [[ "$sparse_ecal" == "auto" ]]; then
    case "$DATASET_PATH" in
        *events_v8.*) sparse_ecal_flag="--sparse_ecal" ;;
    esac
fi

python -m train.pretrain \
    --train \
    --stage1 \
    ${sparse_ecal_flag:+$sparse_ecal_flag} \
    --augmentations_enabled \
    --dataset_path "$DATASET_PATH" \
    --metadata_path "$METADATA_PATH" \
    --model $model \
    --eps $eps \
    --batch_size $batch_size \
    --preprocessing_input $preprocessing_input \
    --label_smoothing $label_smoothing \
    --dropout $dropout \
    --attn_dropout $attn_dropout \
    --drop_path_rate $drop_path_rate \
    --dropout_dec $dropout_dec \
    --attn_dropout_dec $attn_dropout_dec \
    --drop_path_rate_dec $drop_path_rate_dec \
    --mask_ratio $mask_ratio \
    --epochs $epochs \
    --num_workers $num_workers \
    --blr $blr \
    --accum_grad_batches $accum_grad_batches \
    --warmup_epochs $warmup_epochs \
    --cosine_annealing_epochs $cosine_annealing_epochs \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --save_dir $save_dir \
    --name $name \
    --log_every_n_steps $log_every_n_steps \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --early_stop_patience $early_stop_patience \
    --nb_nodes $nb_nodes \
    --gpus "${gpus[@]}" \
    --reconstruction_loss_mode $reconstruction_loss_mode \
    --reconstruction_chamfer_weight $reconstruction_chamfer_weight \
    --reconstruction_distance_reg_weight $reconstruction_distance_reg_weight \
    --reconstruction_max_distance_fcal $reconstruction_max_distance_fcal \
    --reconstruction_max_distance_ahcal $reconstruction_max_distance_ahcal \
    --reconstruction_gamma_distance $reconstruction_gamma_distance \
    --relational_loss_mode $relational_loss_mode \
    --relational_distance_weight $relational_distance_weight \
    --relational_max_distance $relational_max_distance \
    --relational_gamma_distance $relational_gamma_distance \
    --relational_pass_prob $relational_pass_prob \
    --relational_mask_ratio $relational_mask_ratio \
    --relational_voxel_keep_prob $relational_voxel_keep_prob \
    --relational_pass_seed $relational_pass_seed \
    ${load_checkpoint:+--load_checkpoint "$load_checkpoint"} \
    ${resume_checkpoint:+--resume_checkpoint "$resume_checkpoint"}
