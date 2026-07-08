#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PIPELINE_DIR}/.." && pwd)"
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
mkdir -p "${MPLCONFIGDIR}"

cd "${REPO_DIR}"

CONDA_EXE="${CONDA_BIN}" GPU_IDS="${GPU_IDS}" \
    LIGHTNING_DEVICES="${LIGHTNING_DEVICES}" \
    bash "${SCRIPT_DIR}/run_two_view.sh"
CONDA_EXE="${CONDA_BIN}" GPU_IDS="${GPU_IDS}" \
    LIGHTNING_DEVICES="${LIGHTNING_DEVICES}" \
    bash "${SCRIPT_DIR}/run_three_view.sh"

"${PYTHON[@]}" -m faser2d_flash.evaluate \
    --config "${PIPELINE_DIR}/configs/two_view.yaml" \
    --selection taskwise --split both
"${PYTHON[@]}" -m faser2d_flash.evaluate \
    --config "${PIPELINE_DIR}/configs/three_view.yaml" \
    --selection taskwise --split both

"${PYTHON[@]}" -m faser2d_flash.compare \
    --two-run "${PIPELINE_DIR}/artifacts/xz_yz" \
    --three-run "${PIPELINE_DIR}/artifacts/xz_yz_xy" \
    --selection taskwise \
    --output-dir "${PIPELINE_DIR}/artifacts/comparison"
