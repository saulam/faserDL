#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PIPELINE_DIR}/.." && pwd)"
CONFIG="${PIPELINE_DIR}/configs/three_view.yaml"
ENV_NAME="${CONDA_ENV:-platon-flashattn}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
GPU_IDS="${GPU_IDS:-0,1}"
LIGHTNING_DEVICES="${LIGHTNING_DEVICES:-2}"

if [[ -z "${CONDA_BIN}" ]]; then
    echo "ERROR: conda was not found. Activate ${ENV_NAME} or set CONDA_EXE." >&2
    exit 2
fi

PYTHON=("${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" python)
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/faser2d-matplotlib-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/faser2d-cache-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
cd "${REPO_DIR}"

if [[ ! -f "${PIPELINE_DIR}/metadata/v8_2d/metadata.json" ]]; then
    "${PYTHON[@]}" -m faser2d_flash.build_metadata \
        --config "${CONFIG}" \
        --workers "${METADATA_WORKERS:-16}"
fi

read -r CONFIGURED_BATCH CONFIGURED_ACCUMULATION < <(
    "${PYTHON[@]}" -c \
        "from faser2d_flash.config import load_config; c=load_config('${CONFIG}'); print(c['training']['batch_size'], c['training']['accumulation_steps'])"
)
if (( CONFIGURED_BATCH % 2 != 0 )); then
    echo "ERROR: configured batch size ${CONFIGURED_BATCH} is not divisible by two." >&2
    exit 2
fi
THREE_VIEW_BATCH=$((CONFIGURED_BATCH / 2))
THREE_VIEW_ACCUMULATION=$((CONFIGURED_ACCUMULATION * 2))

echo "Three-view per-GPU batch: ${THREE_VIEW_BATCH}"
echo "Three-view accumulation steps: ${THREE_VIEW_ACCUMULATION}"
"${PYTHON[@]}" -m faser2d_flash.runtime --require-flash --precision bf16
"${PYTHON[@]}" -m faser2d_flash.train \
    --config "${CONFIG}" \
    --set "distributed.devices=${LIGHTNING_DEVICES}" \
    --set "training.batch_size=${THREE_VIEW_BATCH}" \
    --set "training.accumulation_steps=${THREE_VIEW_ACCUMULATION}" \
    "$@"
