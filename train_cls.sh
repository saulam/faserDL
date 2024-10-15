#!/bin/bash

# Default arguments
dataset_path="../events_v3_new"
eps=1e-12
batch_size=16
epochs=150
num_workers=32
lr=5e-4
accum_grad_batches=1
warmup_steps=0
weight_decay=4e-5
beta1=0.9
beta2=0.999
save_dir="logs_cls"
name="v3"
log_every_n_steps=10
save_top_k=1
checkpoint_path="checkpoints_cls"
checkpoint_name="v3"
load_checkpoint=None
gpus=(4)

python -m train.train_cls \
    --train \
    --dataset_path $dataset_path \
    --eps $eps \
    --batch_size $batch_size \
    --epochs $epochs \
    --num_workers $num_workers \
    --lr $lr \
    --accum_grad_batches $accum_grad_batches \
    --warmup_steps $warmup_steps \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --save_dir $save_dir \
    --name $name \
    --log_every_n_steps $log_every_n_steps \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --load_checkpoint $load_checkpoint \
    --gpus "${gpus[@]}"

