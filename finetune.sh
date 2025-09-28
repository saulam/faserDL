#!/bin/bash

# Default arguments
dataset_path="/scratch/salonso/sparse-nns/faser/events_new_v5.1*"
metadata_path="/scratch/salonso/sparse-nns/faser/events_new_v5.1b/metadata_stats.pkl"
shardshuffle=200
shuffle=2000
model="tiny"
eps=1e-8
batch_size=256
mixup_alpha=0.0
preprocessing_input="log"
preprocessing_output="log"
label_smoothing=0.02
dropout=0.0
attn_dropout=0.0
drop_path_rate=0.1
epochs=100
num_workers=16
blr=5e-4
layer_decay=0.65
accum_grad_batches=1
warmup_epochs=5
cosine_annealing_epochs=45
weight_decay=0.05
beta1=0.9
beta2=0.999
ema_decay=0.9999
head_init=2e-5
save_dir="logs_final"
name="finetune_v5.1b_dlnu_log_base_clariden_0_5_v34"
log_every_n_steps=10
save_top_k=1
checkpoint_path="checkpoints_final"
checkpoint_name="finetune_v5.1b_dlnu_log_base_clariden_0_5_v34"
early_stop_patience=10
load_checkpoint="checkpoints_final/pretrain_v5.1b_dlnu_log_base_clariden_0_5_lowlr_v16/loss_total_val/epoch=114-step=41400.ckpt"
gpus=(0 1)

python -m train.finetune \
    --train \
    --stage2 \
    --augmentations_enabled \
    --dataset_path "$dataset_path" \
    --metadata_path $metadata_path \
    --model $model \
    --eps $eps \
    --mixup_alpha $mixup_alpha \
    --batch_size $batch_size \
    --preprocessing_input $preprocessing_input \
    --preprocessing_output $preprocessing_output \
    --label_smoothing $label_smoothing \
    --dropout $dropout \
    --drop_path_rate $drop_path_rate \
    --epochs $epochs \
    --num_workers $num_workers \
    --blr $blr \
    --layer_decay $layer_decay \
    --accum_grad_batches $accum_grad_batches \
    --warmup_epochs $warmup_epochs \
    --cosine_annealing_epochs $cosine_annealing_epochs \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --ema_decay $ema_decay \
    --head_init $head_init \
    --save_dir $save_dir \
    --name $name \
    --log_every_n_steps $log_every_n_steps \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --early_stop_patience $early_stop_patience \
    --load_checkpoint $load_checkpoint \
    --gpus "${gpus[@]}"

