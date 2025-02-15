#!/bin/bash

# Default arguments
dataset_path="/scratch/salonso/sparse-nns/faser/events_v3.5"
eps=1e-12
batch_size=32
epochs=50
num_workers=64
lr=2e-4
accum_grad_batches=1
warmup_steps=1
cosine_annealing_steps=20
weight_decay=1e-4
beta1=0.9
beta2=0.95
losses=("focal" "dice")
save_dir="/scratch/salonso/sparse-nns/faser/deep_learning/faserDL/logs_final"
name="enc_v1"
log_every_n_steps=10
save_top_k=1
checkpoint_path="/scratch2/salonso/faser/checkpoints_final"
checkpoint_name="enc_v1"
gpus=(0)

python -m train.train_enc \
    --train \
    --dataset_path $dataset_path \
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
    --gpus "${gpus[@]}"

