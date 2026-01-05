#!/bin/bash

# Distance-Aware MAE Pretraining Script
# Based on pretrain.sh with distance-aware loss parameters added

# Default arguments (same as original)
dataset_path="/scratch/salonso/sparse-nns/faser/events_v7.0*"
metadata_path="/scratch/salonso/sparse-nns/faser/events_v7.0_500_npz/metadata_stats.pkl"
shardshuffle=200
shuffle=4000
model="tiny"
eps=1e-8
batch_size=256
preprocessing_input="log"
label_smoothing=0.02
dropout=0.0
attn_dropout=0.0
drop_path_rate=0.0
dropout_dec=0.0
attn_dropout_dec=0.0
drop_path_rate_dec=0.0
mask_ratio=0.75
epochs=400
num_workers=16
blr=1.5e-4
accum_grad_batches=2
warmup_epochs=40
cosine_annealing_epochs=360
weight_decay=0.05
beta1=0.9
beta2=0.95
save_dir="logs_final"
name="pretrain_v7.0_distance_v1"
log_every_n_steps=50
save_top_k=1
checkpoint_path="checkpoints_final"
checkpoint_name="pretrain_v7.0_distance_v1"
early_stop_patience=200
load_checkpoint=""  # Set to existing checkpoint if resuming
gpus=(0 1)

# ============================================================
# DISTANCE-AWARE LOSS PARAMETERS (NEW)
# ============================================================

# Loss mode: 'hybrid', 'distance_only', or 'focal_dt'
# Start with 'hybrid' for smooth transition from standard training
distance_loss_mode="hybrid"

# Maximum spatial distance (in voxels) to consider for partial credit
# Typical: 3.0-7.0 depending on detector resolution and typical errors
max_distance=5.0

# Distance decay exponent (controls penalty steepness)
# 1.0 = linear, 2.0 = quadratic (recommended), 3.0 = cubic
gamma_distance=2.0

# Weight for soft chamfer occupancy component in hybrid mode
# Start with 0.2-0.3, can increase to 0.5 after initial training
chamfer_weight=0.3

# Weight for distance-weighted regression component
# Start with 0.2-0.3, can increase to 0.5 after initial training
distance_reg_weight=0.3

# Temperature for soft chamfer matching (1.0 = balanced)
temperature_chamfer=1.0

# Use focal distance transform for occupancy (optional)
use_focal_dt=false

# ============================================================
# SEMANTIC SEGMENTATION DISTANCE-AWARE PARAMETERS (NEW)
# ============================================================

# Enable distance-aware losses for semantic segmentation (hie, dec, pid)
# These tasks predict primary/secondary particle and PDG on kept patches
use_distance_semantic=true

# Weight for distance-aware semantic component
# Start with 0.2-0.3, similar to reconstruction weights
semantic_distance_weight=0.3

# Maximum distance for semantic label propagation (typically smaller than reconstruction)
# Semantic labels are more localized, so use 2-4 voxels
semantic_max_distance=3.0

# ============================================================
# TRAINING PHASES (for progressive training)
# ============================================================

# Uncomment the phase you want to run:

# PHASE 1: Initial training (gentle introduction)
# chamfer_weight=0.2
# distance_reg_weight=0.2
# name="pretrain_v7.0_distance_phase1"

# PHASE 2: Mid-training (balanced contribution)
chamfer_weight=0.3
distance_reg_weight=0.3
name="pretrain_v7.0_distance_phase2"

# PHASE 3: Fine-tuning (strong spatial awareness)
# chamfer_weight=0.4
# distance_reg_weight=0.4
# name="pretrain_v7.0_distance_phase3"

# ============================================================

echo "=========================================="
echo "Distance-Aware MAE Pretraining"
echo "=========================================="
echo "Loss Mode: $distance_loss_mode"
echo "Max Distance: $max_distance voxels"
echo "Gamma Distance: $gamma_distance"
echo "Chamfer Weight: $chamfer_weight"
echo "Distance Reg Weight: $distance_reg_weight"
echo "=========================================="
echo ""

# Build command
cmd="python -m train.pretrain \
    --train \
    --stage1 \
    --augmentations_enabled \
    --dataset_path \"$dataset_path\" \
    --metadata_path $metadata_path \
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
    --shardshuffle $shardshuffle \
    --shuffle $shuffle \
    --distance_loss_mode $distance_loss_mode \
    --max_distance $max_distance \
    --gamma_distance $gamma_distance \
    --chamfer_weight $chamfer_weight \
    --distance_reg_weight $distance_reg_weight \
    --temperature_chamfer $temperature_chamfer \
    --semantic_distance_weight $semantic_distance_weight \
    --semantic_max_distance $semantic_max_distance"

# Add use_distance_semantic flag if enabled
if [ "$use_distance_semantic" = true ]; then
    cmd="$cmd --use_distance_semantic"
fi

# Add focal DT flag if enabled
if [ "$use_focal_dt" = true ]; then
    cmd="$cmd --use_focal_dt"
fi

# Add checkpoint loading if specified
if [ -n "$load_checkpoint" ]; then
    cmd="$cmd --load_checkpoint $load_checkpoint"
fi

# Add GPU specification
if [ ${#gpus[@]} -gt 0 ]; then
    gpu_list=$(IFS=,; echo "${gpus[*]}")
    cmd="CUDA_VISIBLE_DEVICES=$gpu_list $cmd"
fi

# Execute
echo "Running command:"
echo "$cmd"
echo ""
eval $cmd
