#!/bin/bash

# Default arguments
pretrained_path="../checkpoints/original3/epoch=286-step=118531.ckpt"
unfreeze_at_epoch=2
gradual_unfreeze_steps=1
lr_factor=1.0
dataset_path="../events_v3_new"
eps=1e-12
chunk_size=2024
batch_size=8
epochs=50
num_workers=16
lr=1e-4
accum_grad_batches=1
warmup_steps=0
weight_decay=4e-5
beta1=0.9
beta2=0.999
losses=("focal" "dice")
save_dir="logs_finetuning"
name="original3"
log_every_n_steps=10
save_top_k=1
checkpoint_path="../checkpoints_finetuning"
checkpoint_name="original1_v3"
load_checkpoint=None
gpus=(1 2)

python -m train.train_seg \
    --train \
    --contrastive \
    --finetuning \
    --unfreeze_at_epoch $unfreeze_at_epoch \
    --gradual_unfreeze_steps $gradual_unfreeze_steps \
    --lr_factor $lr_factor \
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

