#!/bin/bash

# Default arguments
dataset_path="/scratch/salonso/sparse-nns/faser/events_v3_new"
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
losses=("focal" "dice")
save_dir="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL/logs_original"
name="v1"
log_every_n_steps=10
save_top_k=1
checkpoint_path="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL/checkpoints_original"
checkpoint_name="v1"
load_checkpoint=None
gpus=(0)

python -m train.train_seg \
    --train \
    --dataset_path $dataset_path \
    --eps $eps \
    --chunk_size $chunk_size \
    --batch_size $batch_size \
    --epochs $epochs \
    --num_workers $num_workers \
    --lr $lr \
    --accum_grad_batches $accum_grad_batches \
    --warmup_steps $warmup_steps \
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
    --load_checkpoint $load_checkpoint \
    --gpus "${gpus[@]}"

