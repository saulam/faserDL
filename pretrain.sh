#!/bin/bash

# Default arguments
dataset_path="/scratch/salonso/sparse-nns/faser/events_v5.1*"
extra_dataset_path="/scratch/salonso/sparse-nns/faser/events_v5.1b_tau_train"
metadata_path="/scratch/salonso/sparse-nns/faser/events_v5.1b/metadata_stats.pkl"
model="base"
eps=1e-12
batch_size=512
preprocessing_input="log"
label_smoothing=0.0
dropout=0.0
mask_ratio=0.5
epochs=800
num_workers=16
blr=1.5e-4
accum_grad_batches=1
warmup_steps=40
cosine_annealing_steps=760
weight_decay=0.05
beta1=0.9
beta2=0.95
save_dir="logs_final"
name="pretrain_v5.1b_dlnu_log_base_v6"
log_every_n_steps=10
save_top_k=1
checkpoint_path="checkpoints_final"
checkpoint_name="pretrain_v5.1b_dlnu_log_base_v6"
early_stop_patience=200
load_checkpoint="checkpoints_final/mae_v5.1b_noglob_v12/loss_val_total/last.ckpt"
gpus=(0 1)

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

