#!/bin/bash

# Default arguments
dataset_path="/scratch/salonso/sparse-nns/faser/events_v5.1"
sets_path="/scratch/salonso/sparse-nns/faser/events_v5.1/sets.pkl"
eps=1e-12
batch_size=32
epochs=50
num_workers=16
lr=1.5e-4
accum_grad_batches=1
warmup_steps=1
cosine_annealing_steps=25
weight_decay=0.0001
beta1=0.9
beta2=0.95
losses=("focal" "dice")
save_dir="logs_final"
name="seg_v5.1_new7"
log_every_n_steps=10
save_top_k=1
checkpoint_path="checkpoints_final"
checkpoint_name="seg_v5.1_new7"
early_stop_patience=10
load_checkpoint="checkpoints_final/seg_v5.1_new4/loss_val_total/last.ckpt"
gpus=(1)

python -m train.train_seg \
    --train \
    --stage1 \
    --augmentations_enabled \
    --dataset_path $dataset_path \
    --sets_path $sets_path \
    --eps $eps \
    --batch_size $batch_size \
    --epochs $epochs \
    --num_workers $num_workers \
    --lr $lr \
    --accum_grad_batches $accum_grad_batches \
    --warmup_steps $warmup_steps \
    --cosine_annealing_steps $cosine_annealing_steps \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --losses "${losses[@]}" \
    --save_dir $save_dir \
    --name $name \
    --log_every_n_steps $log_every_n_steps \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --early_stop_patience $early_stop_patience \
    --gpus "${gpus[@]}"

