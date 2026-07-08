#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${CONDA_ENV:-platon-flashattn}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
    echo "ERROR: conda was not found. Activate ${ENV_NAME} or set CONDA_EXE." >&2
    exit 2
fi

cd "${REPO_DIR}"
"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python -m unittest discover -s faser2d_flash/tests -v
"${CONDA_BIN}" run --no-capture-output -n "${ENV_NAME}" \
    python -m faser2d_flash.runtime
