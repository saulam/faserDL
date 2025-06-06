#!/bin/bash

# Default arguments
dataset_path="/scratch/salonso/sparse-nns/faser/events_v5.1b"
eps=1e-12
batch_size=64
preprocessing_input="sqrt"
epochs=5
num_workers=16
lr=1.5e-4
accum_grad_batches=1
warmup_steps=5
cosine_annealing_steps=35
weight_decay=0.05
beta1=0.9
beta2=0.95
gpus=(0)

python -m train.train_ae_optuna \
    --train \
    --stage1 \
    --augmentations_enabled \
    --dataset_path $dataset_path \
    --eps $eps \
    --batch_size $batch_size \
    --preprocessing_input $preprocessing_input \
    --standardize_input \
    --epochs $epochs \
    --num_workers $num_workers \
    --lr $lr \
    --accum_grad_batches $accum_grad_batches \
    --warmup_steps $warmup_steps \
    --cosine_annealing_steps $cosine_annealing_steps \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --gpus "${gpus[@]}"

