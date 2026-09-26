#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

if [ "$#" -ne 0 ]; then
    echo "Usage: R2E_GYM_OUTPUT_DIR=/shared/path R2E_GYM_LIMIT=1 $0" >&2
    exit 2
fi

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -n "${RELAX_INTEGRATION_ROOT:-}" ]; then
    EXAMPLE_DIR="${RELAX_INTEGRATION_ROOT}/examples/nemo_gym_agentic"
else
    EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
fi
R2E_GYM_OUTPUT_DIR="${R2E_GYM_OUTPUT_DIR:-/data/nemo-gym/r2e-gym}"
R2E_GYM_DATASET="${R2E_GYM_DATASET:-R2E-Gym/R2E-Gym-Lite}"
R2E_GYM_SPLIT="${R2E_GYM_SPLIT:-train}"
R2E_GYM_LIMIT="${R2E_GYM_LIMIT:-1}"
R2E_GYM_BUILD_SIFS="${R2E_GYM_BUILD_SIFS:-1}"
R2E_GYM_AGENT_NAME="${R2E_GYM_AGENT_NAME:-swe_agents}"
R2E_GYM_BUILD_TMPDIR="${R2E_GYM_BUILD_TMPDIR:-${TMPDIR:-/tmp}}"
R2E_GYM_SIF_PREFIX="${R2E_GYM_SIF_PREFIX:-}"

case "${R2E_GYM_OUTPUT_DIR}" in
    /*) ;;
    *)
        echo "R2E_GYM_OUTPUT_DIR must be an absolute path" >&2
        exit 2
        ;;
esac
if ! [[ "${R2E_GYM_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "task-limit must be a positive integer" >&2
    exit 2
fi
if [[ "${R2E_GYM_SIF_PREFIX}" == */* ]]; then
    echo "R2E_GYM_SIF_PREFIX must be a filename prefix without '/'" >&2
    exit 2
fi

SOURCE_DATA="${R2E_GYM_OUTPUT_DIR}/r2e_gym_${R2E_GYM_SPLIT}.jsonl"
MANIFEST="${R2E_GYM_OUTPUT_DIR}/r2e_gym_${R2E_GYM_SPLIT}_sifs.jsonl"
RELAX_DATA="${R2E_GYM_OUTPUT_DIR}/r2e_gym_${R2E_GYM_SPLIT}_relax.jsonl"
SIF_DIR="${R2E_GYM_SIF_DIR:-${R2E_GYM_OUTPUT_DIR}/sif}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${R2E_GYM_OUTPUT_DIR}/apptainer-cache}"
export APPTAINER_CACHEDIR

mkdir -p "${R2E_GYM_OUTPUT_DIR}" "${SIF_DIR}" "${APPTAINER_CACHEDIR}"

PREPARE_LOCK_DIR="${R2E_GYM_OUTPUT_DIR}/.prepare_r2e_gym.lockdir"
if ! mkdir "${PREPARE_LOCK_DIR}" 2>/dev/null; then
    echo "Another R2E-Gym prepare process holds ${PREPARE_LOCK_DIR}" >&2
    echo "Stop the other prepare process before retrying." >&2
    exit 1
fi

build_tmp_dir=""
local_sif=""
shared_partial=""
cleanup() {
    if [ -n "${shared_partial}" ]; then
        rm -f "${shared_partial}"
    fi
    if [ -n "${local_sif}" ]; then
        rm -f "${local_sif}"
    fi
    if [ -n "${build_tmp_dir}" ]; then
        rmdir "${build_tmp_dir}" 2>/dev/null || true
    fi
    rmdir "${PREPARE_LOCK_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Preparing R2E-Gym data locally:"
echo "  dataset=${R2E_GYM_DATASET}"
echo "  split=${R2E_GYM_SPLIT}"
echo "  limit=${R2E_GYM_LIMIT}"
echo "  output_dir=${R2E_GYM_OUTPUT_DIR}"
echo "  sif_dir=${SIF_DIR}"
echo "  sif_prefix=${R2E_GYM_SIF_PREFIX:-<none>}"

"${GYM_ROOT}/.venv/bin/python" \
    "${SCRIPT_DIR}/prepare_r2e_gym.py" \
    --dataset "${R2E_GYM_DATASET}" \
    --split "${R2E_GYM_SPLIT}" \
    --limit "${R2E_GYM_LIMIT}" \
    --agent-name "${R2E_GYM_AGENT_NAME}" \
    --output "${SOURCE_DATA}" \
    --manifest "${MANIFEST}"

/usr/bin/python3 \
    "${EXAMPLE_DIR}/scripts/convert_dataset.py" \
    --input "${SOURCE_DATA}" \
    --output "${RELAX_DATA}"

if [ "${R2E_GYM_BUILD_SIFS}" = "1" ]; then
    while IFS=$'\t' read -r docker_image sif_name; do
        sif_path="${SIF_DIR}/${R2E_GYM_SIF_PREFIX}${sif_name}"
        if [ -s "${sif_path}" ]; then
            echo "Reusing ${sif_path}"
            continue
        fi

        mkdir -p "${R2E_GYM_BUILD_TMPDIR}"
        build_tmp_dir="$(mktemp -d "${R2E_GYM_BUILD_TMPDIR%/}/r2e-gym-sif.XXXXXX")"
        local_sif="${build_tmp_dir}/${sif_name}"
        shared_partial="${sif_path}.partial.${BASHPID}.${RANDOM}"

        apptainer build "${local_sif}" "docker://${docker_image}"
        if [ ! -s "${local_sif}" ]; then
            echo "Apptainer reported success but did not create ${local_sif}" >&2
            exit 1
        fi

        if [ -s "${sif_path}" ]; then
            echo "Reusing ${sif_path}; another process completed it during this build"
        else
            cp "${local_sif}" "${shared_partial}"
            if [ ! -s "${shared_partial}" ]; then
                echo "Failed to stage completed SIF at ${shared_partial}" >&2
                exit 1
            fi
            mv "${shared_partial}" "${sif_path}"
        fi

        rm -f "${local_sif}"
        rmdir "${build_tmp_dir}"
        build_tmp_dir=""
        local_sif=""
        shared_partial=""
    done < <(jq -r '[.docker_image,.sif_name] | @tsv' "${MANIFEST}")
elif [ "${R2E_GYM_BUILD_SIFS}" != "0" ]; then
    echo "R2E_GYM_BUILD_SIFS must be 0 or 1" >&2
    exit 2
fi

awk 'NF { count++ } END { print count, FILENAME }' "${SOURCE_DATA}"
awk 'NF { count++ } END { print count, FILENAME }' "${RELAX_DATA}"
echo "SIF directory: ${SIF_DIR}"
