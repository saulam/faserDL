#!/bin/bash

# Default arguments
dataset_path="/scratch/salonso/sparse-nns/faser/events_new_v5.1*"
metadata_path="/scratch/salonso/sparse-nns/faser/events_new_v5.1b/metadata_stats.pkl"
shardshuffle=200
shuffle=4000
model="tiny"
eps=1e-8
batch_size=8
preprocessing_input="log"
label_smoothing=0.02
dropout=0.0
attn_dropout=0.0
drop_path_rate=0.0
dropout_dec=0.0
attn_dropout_dec=0.0
drop_path_rate_dec=0.0
mask_ratio=0.5
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
name="pretrain_v5.1b_dlnu_log_base_v17"
log_every_n_steps=50
save_top_k=1
checkpoint_path="checkpoints_final"
checkpoint_name="pretrain_v5.1b_dlnu_log_base_v17"
early_stop_patience=200
load_checkpoint="checkpoints_final/mae_v5.1b_noglob_v12/loss_val_total/last.ckpt"
gpus=(1)

python -m train.pretrain \
    --train \
    --stage1 \
    --augmentations_enabled \
    --dataset_path "$dataset_path" \
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
    --gpus "${gpus[@]}"

