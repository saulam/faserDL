#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_DIR="${PROJECT_ROOT}/data_efficiency_study/manifests"
SAVE_DIR="${SAVE_DIR:-logs_data_efficiency}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-${PROJECT_ROOT}/checkpoints_data_efficiency}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
DATASET_PATH="${DATASET_PATH:-}"
METADATA_PATH="${METADATA_PATH:-}"
VAL_MANIFEST="${MANIFEST_DIR}/val.txt"
GPUS=(${GPUS_OVERRIDE:-0})
NB_NODES="${NB_NODES_OVERRIDE:-1}"
BUDGETS=(${BUDGETS_OVERRIDE:-100 300 1000 3000 10000 30000 100000})
SEEDS=(${SEEDS_OVERRIDE:-1 2 3})
CONDITIONS=(${CONDITIONS_OVERRIDE:-pretrained scratch})

MODEL="base"
EPS="1e-8"
MIXUP_ALPHA="0.0"
PREPROC_IN="log"
PREPROC_OUT="log"
LABEL_SMOOTHING="0.02"
DROPOUT="0.0"
ATTN_DROPOUT="0.0"
DROP_PATH_RATE="0.2"
KENDALL_W_MAX="5.0"
CLS_LR_SCALE="1.0"
HEAD_DROPOUT_CLS="0.0"
HEAD_DROPOUT_REG="0.0"
NUM_WORKERS=16
WEIGHT_DECAY="0.05"
BETA1="0.9"
BETA2="0.999"
EMA_DECAY="0.9999"
HEAD_INIT="2e-5"
LOG_EVERY_N_STEPS=10
SAVE_TOP_K=3
ACCUM_GRAD_BATCHES=1

die() {
    echo "Error: $*" >&2
    exit 1
}

require_value() {
    local name="$1"
    local value="${!name:-}"
    [[ -n "$value" ]] || die "Set ${name} before running the study."
}

require_file() {
    local path="$1"
    [[ -f "$path" ]] || die "File not found: ${path}"
}

contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

get_hparams() {
    local budget="$1"
    case "$budget" in
        100)
            BATCH_SIZE=64;   EPOCHS=50;  WARMUP_EPOCHS=5;  COSINE_EPOCHS=45;  EARLY_STOP=10 ;;
        300)
            BATCH_SIZE=64;   EPOCHS=50;  WARMUP_EPOCHS=5;  COSINE_EPOCHS=45;  EARLY_STOP=10 ;;
        1000)
            BATCH_SIZE=128;  EPOCHS=50;  WARMUP_EPOCHS=5;  COSINE_EPOCHS=45;  EARLY_STOP=10 ;;
        3000)
            BATCH_SIZE=256;  EPOCHS=40;  WARMUP_EPOCHS=5;  COSINE_EPOCHS=35;  EARLY_STOP=10 ;;
        10000)
            BATCH_SIZE=512;  EPOCHS=30;  WARMUP_EPOCHS=5;  COSINE_EPOCHS=25;  EARLY_STOP=10 ;;
        30000)
            BATCH_SIZE=512;  EPOCHS=25;  WARMUP_EPOCHS=5;  COSINE_EPOCHS=20;  EARLY_STOP=10 ;;
        100000)
            BATCH_SIZE=1024; EPOCHS=20;  WARMUP_EPOCHS=5;  COSINE_EPOCHS=15;  EARLY_STOP=10 ;;
        *)
            echo "ERROR: Unknown budget $budget"; exit 1 ;;
    esac
}

get_blr() {
    local condition="$1"
    case "$condition" in
        pretrained) echo "5e-4" ;;
        scratch)    echo "1e-3" ;;
        *)          echo "5e-4" ;;
    esac
}

get_layer_decay() {
    local condition="$1"
    case "$condition" in
        pretrained) echo "0.75" ;;
        scratch)    echo "1.0" ;;
        *)          echo "0.75" ;;
    esac
}

cd "$PROJECT_ROOT"
require_value DATASET_PATH
require_value METADATA_PATH
require_file "$METADATA_PATH"
require_file "$VAL_MANIFEST"

if ! compgen -G "$DATASET_PATH" > /dev/null; then
    die "DATASET_PATH does not match any directories: ${DATASET_PATH}"
fi

if contains "pretrained" "${CONDITIONS[@]}"; then
    require_value PRETRAINED_CKPT
    require_file "$PRETRAINED_CKPT"
fi

TOTAL_RUNS=$(( ${#BUDGETS[@]} * ${#CONDITIONS[@]} * ${#SEEDS[@]} ))
RUN_IDX=0

echo "Data-efficiency study"
echo "  runs: ${TOTAL_RUNS} (${#BUDGETS[@]} budgets x ${#CONDITIONS[@]} conditions x ${#SEEDS[@]} seeds)"
echo "  gpus: ${GPUS[*]}"
echo "  nb_nodes: ${NB_NODES}"
echo "  manifests: ${MANIFEST_DIR}"
echo "  checkpoints: ${CHECKPOINT_BASE}"
echo

for BUDGET in "${BUDGETS[@]}"; do
    get_hparams "$BUDGET"

    for CONDITION in "${CONDITIONS[@]}"; do
        BLR=$(get_blr "$CONDITION")
        LAYER_DECAY=$(get_layer_decay "$CONDITION")

        for SEED in "${SEEDS[@]}"; do
            RUN_IDX=$((RUN_IDX + 1))
            EXP_NAME="dataeff_${CONDITION}_${BUDGET}ev_seed${SEED}"
            CKPT_DIR="${CHECKPOINT_BASE}/${CONDITION}/${BUDGET}_events/seed_${SEED}"
            TRAIN_MANIFEST="${MANIFEST_DIR}/${BUDGET}_events/seed_${SEED}.txt"

            echo "[${RUN_IDX}/${TOTAL_RUNS}] ${EXP_NAME}"
            echo "  batch_size=${BATCH_SIZE} epochs=${EPOCHS} blr=${BLR} layer_decay=${LAYER_DECAY}"

            if [[ ! -f "$TRAIN_MANIFEST" ]]; then
                die "Train manifest not found: ${TRAIN_MANIFEST}. Run data_efficiency_study/subsample_dataset.py first."
            fi

            LOAD_CKPT_ARGS=()
            if [[ "$CONDITION" == "pretrained" ]]; then
                LOAD_CKPT_ARGS=(--load_checkpoint "$PRETRAINED_CKPT")
            fi

            RESUME_ARGS=()
            LAST_CKPT="${CKPT_DIR}/${EXP_NAME}/loss_total_val/last.ckpt"
            if [[ -f "$LAST_CKPT" ]]; then
                echo "  resuming from ${LAST_CKPT}"
                RESUME_ARGS=(--resume_checkpoint "$LAST_CKPT")
                LOAD_CKPT_ARGS=()
            fi

            python -m data_efficiency_study.train_with_manifest \
                --train \
                --stage2 \
                --augmentations_enabled \
                --train_manifest "$TRAIN_MANIFEST" \
                --val_manifest "$VAL_MANIFEST" \
                --dataset_path "$DATASET_PATH" \
                --metadata_path "$METADATA_PATH" \
                --model "$MODEL" \
                --eps "$EPS" \
                --mixup_alpha "$MIXUP_ALPHA" \
                --batch_size "$BATCH_SIZE" \
                --preprocessing_input "$PREPROC_IN" \
                --preprocessing_output "$PREPROC_OUT" \
                --label_smoothing "$LABEL_SMOOTHING" \
                --dropout "$DROPOUT" \
                --drop_path_rate "$DROP_PATH_RATE" \
                --kendall_w_max "$KENDALL_W_MAX" \
                --cls_lr_scale "$CLS_LR_SCALE" \
                --head_dropout_cls "$HEAD_DROPOUT_CLS" \
                --head_dropout_reg "$HEAD_DROPOUT_REG" \
                --epochs "$EPOCHS" \
                --num_workers "$NUM_WORKERS" \
                --blr "$BLR" \
                --layer_decay "$LAYER_DECAY" \
                --accum_grad_batches "$ACCUM_GRAD_BATCHES" \
                --warmup_epochs "$WARMUP_EPOCHS" \
                --cosine_annealing_epochs "$COSINE_EPOCHS" \
                --weight_decay "$WEIGHT_DECAY" \
                --beta1 "$BETA1" \
                --beta2 "$BETA2" \
                --ema_decay "$EMA_DECAY" \
                --head_init "$HEAD_INIT" \
                --save_dir "$SAVE_DIR" \
                --name "$EXP_NAME" \
                --log_every_n_steps "$LOG_EVERY_N_STEPS" \
                --save_top_k "$SAVE_TOP_K" \
                --checkpoint_path "$CKPT_DIR" \
                --checkpoint_name "$EXP_NAME" \
                --early_stop_patience "$EARLY_STOP" \
                --nb_nodes "$NB_NODES" \
                --pl_seed "$SEED" \
                "${LOAD_CKPT_ARGS[@]}" \
                "${RESUME_ARGS[@]}" \
                --gpus "${GPUS[@]}"

            echo "  completed"
            echo
        done
    done
done

echo "Completed ${TOTAL_RUNS} runs."
echo "Checkpoints: ${CHECKPOINT_BASE}/"
echo "Logs: ${SAVE_DIR}/"
