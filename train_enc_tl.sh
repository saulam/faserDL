#!/bin/bash

# Default arguments
dataset_path="/scratch/salonso/sparse-nns/faser/events_v5.1b"
sets_path="/scratch/salonso/sparse-nns/faser/events_v5.1b/sets.pkl"
eps=1e-12
batch_size=64
preprocessing_input="sqrt"
preprocessing_output="log"
epochs=20
phase1_epochs=2
phase2_epochs=15
num_workers=16
lr=1e-4
layer_decay=0.9
accum_grad_batches=1
warmup_steps=5
cosine_annealing_steps=10
weight_decay=0.05
beta1=0.9
beta2=0.999
losses=("focal" "dice")
save_dir="logs_final"
name="enc_v5.1b_tl_ae_v1"
log_every_n_steps=10
save_top_k=1
checkpoint_path="checkpoints_final"
checkpoint_name="enc_v5.1b_tl_ae_v1"
early_stop_patience=10
load_checkpoint="checkpoints_final/ae_v5.1b_v1/loss_val_total/last.ckpt"
gpus=(0)

python -m train.train_enc_tl \
    --train \
    --stage2 \
    --augmentations_enabled \
    --dataset_path $dataset_path \
    --eps $eps \
    --batch_size $batch_size \
    --preprocessing_input $preprocessing_input \
    --standardize_input \
    --preprocessing_output $preprocessing_output \
    --phase1_epochs $phase1_epochs \
    --phase2_epochs $phase2_epochs \
    --num_workers $num_workers \
    --lr $lr \
    --layer_decay $layer_decay \
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
    --load_checkpoint $load_checkpoint \
    --gpus "${gpus[@]}"

