#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${PIPELINE_DIR}/.." && pwd)"
ENV_NAME="${CONDA_ENV:-platon-flashattn}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
    echo "ERROR: conda was not found. Activate ${ENV_NAME} or set CONDA_EXE." >&2
    exit 2
fi
PYTHON=("${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" python)

cd "${REPO_DIR}"

if [[ ! -f "${PIPELINE_DIR}/metadata/v8_2d/metadata.json" ]]; then
    "${PYTHON[@]}" -m faser2d_flash.build_metadata \
        --config "${PIPELINE_DIR}/configs/two_view.yaml" \
        --workers "${METADATA_WORKERS:-16}"
fi

"${PYTHON[@]}" -m faser2d_flash.runtime --require-flash --precision bf16
"${PYTHON[@]}" -m faser2d_flash.train --config "${PIPELINE_DIR}/configs/two_view.yaml"
"${PYTHON[@]}" -m faser2d_flash.train --config "${PIPELINE_DIR}/configs/three_view.yaml"

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
