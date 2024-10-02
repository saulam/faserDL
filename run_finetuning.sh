#!/bin/bash

# Default arguments
pretrained_path="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL/checkpoints/v5/last.ckpt"
dataset_path="/scratch/salonso/sparse-nns/faser/events_v3_new2"
eps=1e-12
chunk_size=2024
batch_size=32
epochs=50
num_workers=32
lr=1e-3
accum_grad_batches=1
warmup_steps=0
weight_decay=4e-5
beta1=0.9
beta2=0.999
losses=("focal" "dice")
save_dir="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL/logs_finetuning"
name="v4"
log_every_n_steps=10
save_top_k=1
checkpoint_path="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL/checkpoints_finetuning"
checkpoint_name="v4"
load_checkpoint=None
gpus=(0)

python -m train.train \
    --train \
    --contrastive \
    --finetuning \
    --softmax \
    --pretrained_path $pretrained_path \
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

