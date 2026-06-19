#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
READER="${SCRIPT_DIR}/read_root_v8.py"

DATA_ROOT="${DATA_ROOT:-/scratch2/salonso/faser}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch4/salonso/faser}"
MAX_JOBS="${MAX_JOBS:-30}"
CONDA_ENV="${CONDA_ENV:-rootenv}"
FASER_CLASSES_DICT="${FASER_CLASSES_DICT:-/scratch/fcufino/FASER/Python_io/lib/ClassesDict.so}"

VERSIONS=("8.0_1000" "8.0_1001" "8.0_1002")
if (( $# > 0 )); then
    VERSIONS=("$@")
fi

if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_JOBS must be a positive integer, got '${MAX_JOBS}'." >&2
    exit 2
fi

if [[ ! -f "${READER}" ]]; then
    echo "ERROR: Reader not found: ${READER}" >&2
    exit 2
fi

if [[ ! -f "${FASER_CLASSES_DICT}" ]]; then
    echo "ERROR: ROOT dictionary not found: ${FASER_CLASSES_DICT}" >&2
    exit 2
fi

source "${SCRIPT_DIR}/setup_env.sh"

declare -a PIDS=()
declare -a LABELS=()
failures=0
launched=0

wait_for_oldest_job() {
    local pid="${PIDS[0]}"
    local label="${LABELS[0]}"

    if ! wait "${pid}"; then
        echo "FAILED: ${label}" >&2
        failures=$((failures + 1))
    fi

    PIDS=("${PIDS[@]:1}")
    LABELS=("${LABELS[@]:1}")
}

for raw_version in "${VERSIONS[@]}"; do
    version="${raw_version#v}"
    reco_dir="${DATA_ROOT}/FASERCALRECODATA_v${version}"
    cal_dir="${DATA_ROOT}/FASERCALDATA_v${version}"
    output_dir="${OUTPUT_ROOT}/events_v${version}_2d"
    log_dir="${output_dir}/logs"

    if [[ ! -d "${reco_dir}" ]]; then
        echo "ERROR: RECO directory not found: ${reco_dir}" >&2
        exit 2
    fi

    if [[ ! -d "${cal_dir}" ]]; then
        echo "ERROR: CAL directory not found: ${cal_dir}" >&2
        exit 2
    fi

    mapfile -d '' reco_files < <(
        find "${reco_dir}" -mindepth 2 -type f -name '*.root' -print0 | sort -z
    )
    chunks="${#reco_files[@]}"

    if (( chunks == 0 )); then
        echo "ERROR: No RECO ROOT files found in ${reco_dir}" >&2
        exit 2
    fi

    mkdir -p "${log_dir}"
    echo "Launching v${version}: ${chunks} workers -> ${output_dir}"

    for ((number = 0; number < chunks; number++)); do
        label="v${version} chunk ${number}/${chunks}"
        log_file="${log_dir}/chunk_${number}.log"
        completion_marker="Successfully processed chunk ${number}/${chunks}."

        if [[ -f "${log_file}" ]] && grep -Fq "${completion_marker}" "${log_file}"; then
            echo "Skipping completed ${label}"
            continue
        fi

        if (( ${#PIDS[@]} >= MAX_JOBS )); then
            wait_for_oldest_job
        fi

        (
            FASER_CLASSES_DICT="${FASER_CLASSES_DICT}" \
                conda run --no-capture-output -n "${CONDA_ENV}" \
                python "${READER}" \
                --number "${number}" \
                --chunks "${chunks}" \
                --disable \
                --base-path "${DATA_ROOT}" \
                --output-dir "${output_dir}" \
                --skip-existing \
                --version "${version}" \
                --save-mode 2d
        ) >"${log_file}" 2>&1 &

        PIDS+=("$!")
        LABELS+=("${label}")
        launched=$((launched + 1))
    done
done

while (( ${#PIDS[@]} > 0 )); do
    wait_for_oldest_job
done

echo "Finished ${launched} workers with ${failures} failure(s)."
if (( failures > 0 )); then
    exit 1
fi
