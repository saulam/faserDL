#!/bin/bash

# Default arguments
dataset_path="/scratch/salonso/sparse-nns/faser/events_v5.1b"
model="base"
eps=1e-12
batch_size=256
preprocessing_input="sqrt"
standardize_input="unit-var"
mask_ratio=0.75
epochs=800
num_workers=8
blr=1.5e-4
accum_grad_batches=1
warmup_steps=40
cosine_annealing_steps=760
weight_decay=0.05
beta1=0.9
beta2=0.95
save_dir="logs_final"
name="maevit_v5.1b_v1"
log_every_n_steps=10
save_top_k=1
checkpoint_path="checkpoints_final"
checkpoint_name="maevit_v5.1b_v1"
early_stop_patience=100
load_checkpoint="checkpoints_final/mae_v5.1b_noglob_v12/loss_val_total/last.ckpt"
gpus=(0 1)

python -m train.pretrain \
    --train \
    --stage1 \
    --augmentations_enabled \
    --dataset_path $dataset_path \
    --model $model \
    --eps $eps \
    --batch_size $batch_size \
    --preprocessing_input $preprocessing_input \
    --standardize_input $standardize_input \
    --mask_ratio $mask_ratio \
    --epochs $epochs \
    --num_workers $num_workers \
    --blr $blr \
    --accum_grad_batches $accum_grad_batches \
    --warmup_steps $warmup_steps \
    --cosine_annealing_steps $cosine_annealing_steps \
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

