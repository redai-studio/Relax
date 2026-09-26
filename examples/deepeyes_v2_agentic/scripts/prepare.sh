#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# One-shot prep for DeepEyes V2: prepares the node-local host app environment,
# builds the SIF, downloads + converts the training parquets, and generates the
# smoke parquet under ${DATA_DIR}. Idempotent.
#
# Required:
#   DATA_DIR    root dir for shared data assets. After this script:
#                 ${DATA_DIR}/sif/deepeyes_v2_kernel.sif    (~115 MiB)
#                 ${DATA_DIR}/data/raw/*.parquet            (~10 GiB raw)
#                 ${DATA_DIR}/data/*.parquet                (~10 GiB converted)
#                 ${DATA_DIR}/data/smoke.parquet            (~16 KiB synthetic)
#
# Optional (set in env.sh — see env.sh.example):
#   DEEPEYES_V2_APP_ENV_ROOT        node-local app environment root (default /tmp/deepeyes-v2-app-env)
#   HF_ENDPOINT                    default https://huggingface.co
#   HF_HTTP_PROXY                  HTTPS proxy URL (no default)
#   HF_NO_PROXY                    comma-separated no_proxy list
#   BOOTSTRAP_FROM_IMAGE           docker image for SIF build (default python:3.11-slim)
#   BOOTSTRAP_PIP_INDEX_URL        pip mirror to use inside the SIF build
#   BOOTSTRAP_PIP_TRUSTED_HOST     pip trusted host inside the SIF build
#   SKIP_SIF=1, SKIP_TRAIN=1, SKIP_SMOKE=1
#
# Usage:
#   cp examples/deepeyes_v2_agentic/env.sh.example examples/deepeyes_v2_agentic/env.sh
#   # edit env.sh: at minimum set DATA_DIR
#   bash examples/deepeyes_v2_agentic/scripts/prepare.sh

set -eu
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

# Auto-source env.sh if present (gitignored, machine-specific overrides).
# shellcheck source=/dev/null
[ -f "${EXAMPLE_DIR}/env.sh" ] && source "${EXAMPLE_DIR}/env.sh"

if [ -z "${DATA_DIR:-}" ]; then
    echo "ERROR: DATA_DIR must be set (in env.sh or the environment)." >&2
    echo "Bootstrap from the example: cp ${EXAMPLE_DIR}/env.sh.example ${EXAMPLE_DIR}/env.sh" >&2
    exit 1
fi

SIF_DIR="${DATA_DIR}/sif"
DATA_DIR_INNER="${DATA_DIR}/data"
RAW_DIR="${DATA_DIR_INNER}/raw"

SIF_OUT="${SIF_DIR}/deepeyes_v2_kernel.sif"
SMOKE_PARQUET="${DATA_DIR_INNER}/smoke.parquet"

mkdir -p "${SIF_DIR}" "${DATA_DIR_INNER}" "${RAW_DIR}"

bash "${SCRIPT_DIR}/prepare_app_env.sh"
echo

echo "[prepare] DATA_DIR : ${DATA_DIR}"
echo "[prepare] sif      : ${SIF_OUT}"
echo "[prepare] data     : ${DATA_DIR_INNER}/"
echo "[prepare] raw      : ${RAW_DIR}/"
echo

# ---------------------------------------------------------------------------
# 1. SIF
# ---------------------------------------------------------------------------
if [ -n "${SKIP_SIF:-}" ]; then
    echo "[1/3 sif] SKIP_SIF set — skipping"
elif [ -s "${SIF_OUT}" ]; then
    size_mib=$(( $(stat -c '%s' "${SIF_OUT}") / 1024 / 1024 ))
    echo "[1/3 sif] ${SIF_OUT} already present (${size_mib} MiB) — skip"
else
    echo "[1/3 sif] building apptainer image…"
    if ! command -v apptainer >/dev/null 2>&1; then
        echo "ERROR: apptainer not in PATH" >&2; exit 1
    fi
    BUILD_DIR=$(mktemp -d)
    trap 'rm -rf "${BUILD_DIR}"' EXIT
    cp "${EXAMPLE_DIR}/apptainer_env/deepeyes_v2_kernel.def" "${BUILD_DIR}/"
    cd "${BUILD_DIR}"
    BUILD_ARGS=(
        --build-arg "FROM_IMAGE=${BOOTSTRAP_FROM_IMAGE:-python:3.11-slim}"
        --build-arg "PIP_INDEX_URL=${BOOTSTRAP_PIP_INDEX_URL:-}"
        --build-arg "PIP_TRUSTED_HOST=${BOOTSTRAP_PIP_TRUSTED_HOST:-}"
    )
    if ! apptainer build "${BUILD_ARGS[@]}" "${BUILD_DIR}/deepeyes_v2_kernel.sif" "${BUILD_DIR}/deepeyes_v2_kernel.def"; then
        echo "[1/3 sif] plain build failed, retrying with --fakeroot"
        rm -f "${BUILD_DIR}/deepeyes_v2_kernel.sif"
        apptainer build --fakeroot "${BUILD_ARGS[@]}" "${BUILD_DIR}/deepeyes_v2_kernel.sif" "${BUILD_DIR}/deepeyes_v2_kernel.def"
    fi
    mv "${BUILD_DIR}/deepeyes_v2_kernel.sif" "${SIF_OUT}"
    cd - >/dev/null
    echo "[1/3 sif] verifying kernel deps…"
    apptainer exec "${SIF_OUT}" python -c \
        "import ipykernel, PIL, matplotlib, autopep8, numpy; print('all deps OK')"
fi
echo

# ---------------------------------------------------------------------------
# 2. Training parquets (download from HF mirror + convert)
# ---------------------------------------------------------------------------
TRAIN_FILES=(
    perception_all_1.parquet
    perception_all_2.parquet
    perception_all_3.parquet
    perception_all_4.parquet
    perception_all_5.parquet
    reason.parquet
    search.parquet
    vstar_test.parquet
)
REPO_ID="honglyhly/DeepEyesV2_RL"
# Default to direct huggingface.co. Override HF_ENDPOINT in env.sh if you
# need to use a mirror (note: hf-mirror.com 308-redirects the largest file
# back to upstream, which huggingface_hub refuses — direct HF is more robust
# when reachable, with or without a corporate proxy).
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# Optional outbound HTTPS proxy for the download step.
if [ -n "${HF_HTTP_PROXY:-}" ]; then
    export http_proxy="${HF_HTTP_PROXY}"
    export https_proxy="${HF_HTTP_PROXY}"
    [ -n "${HF_NO_PROXY:-}" ] && export no_proxy="${HF_NO_PROXY}"
fi

if [ -n "${SKIP_TRAIN:-}" ]; then
    echo "[2/3 train] SKIP_TRAIN set — skipping"
else
    echo "[2/3 train] repo=${REPO_ID} endpoint=${HF_ENDPOINT}"
    if ! python -c "import huggingface_hub" 2>/dev/null; then
        echo "[2/3 train] installing huggingface_hub…"
        pip install --quiet --upgrade huggingface_hub
    fi
    missing=()
    for f in "${TRAIN_FILES[@]}"; do
        if [ ! -s "${RAW_DIR}/${f}" ]; then
            missing+=("${f}")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "[2/3 train] downloading ${#missing[@]} raw file(s) via huggingface_hub Python API…"
        REPO_ID="${REPO_ID}" RAW_DIR="${RAW_DIR}" MISSING="${missing[*]}" python <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
repo = os.environ["REPO_ID"]
out = os.environ["RAW_DIR"]
for name in os.environ["MISSING"].split():
    print(f"[2/3 train]   -> {name}", flush=True)
    hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", local_dir=out)
PY
    else
        echo "[2/3 train] all raw files present, skipping download"
    fi
    # Convert (idempotent: only if any raw is newer than its converted twin or twin missing)
    need_convert=0
    for f in "${TRAIN_FILES[@]}"; do
        if [ ! -f "${DATA_DIR_INNER}/${f}" ] || [ "${RAW_DIR}/${f}" -nt "${DATA_DIR_INNER}/${f}" ]; then
            need_convert=1; break
        fi
    done
    if [ ${need_convert} -eq 1 ]; then
        echo "[2/3 train] running rl_data_convert.py…"
        python "${EXAMPLE_DIR}/convert_tool/rl_data_convert.py" \
            --input "${RAW_DIR}" \
            --output "${DATA_DIR_INNER}"
    else
        echo "[2/3 train] all converted files up to date"
    fi
fi
echo

# ---------------------------------------------------------------------------
# 3. Smoke parquet (4 synthetic rows, one per data_source)
# ---------------------------------------------------------------------------
if [ -n "${SKIP_SMOKE:-}" ]; then
    echo "[3/3 smoke] SKIP_SMOKE set — skipping"
elif [ -s "${SMOKE_PARQUET}" ]; then
    echo "[3/3 smoke] ${SMOKE_PARQUET} already present — skip"
else
    echo "[3/3 smoke] generating ${SMOKE_PARQUET}…"
    python "${SCRIPT_DIR}/build_smoke_parquet.py" --output "${SMOKE_PARQUET}"
fi
echo

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "[done] layout:"
[ -f "${SIF_OUT}" ] && printf "  %-50s %s\n" "${SIF_OUT}" "$(du -h "${SIF_OUT}" | cut -f1)"
for f in "${TRAIN_FILES[@]}" smoke.parquet; do
    p="${DATA_DIR_INNER}/${f}"
    if [ -f "${p}" ]; then
        printf "  %-50s %s\n" "${p}" "$(du -h "${p}" | cut -f1)"
    fi
done
echo
echo "Launch now with the same DATA_DIR:"
echo "  DATA_DIR=${DATA_DIR} bash examples/deepeyes_v2_agentic/scripts/smoke.sh"
echo "  DATA_DIR=${DATA_DIR} bash examples/deepeyes_v2_agentic/run_deepeyes_v2_agentic.sh"
