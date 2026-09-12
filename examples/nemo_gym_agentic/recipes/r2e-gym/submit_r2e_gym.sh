#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <ray-head-ip> [golden|train]" >&2
    exit 2
fi

R2E_RAY_HEAD="$1"
R2E_GYM_MODE="${2:-golden}"
if [[ "${R2E_RAY_HEAD}" == *"://"* ]] || [[ "${R2E_RAY_HEAD}" == *":"* ]]; then
    echo "ray-head-ip must be a bare host or IP without a scheme or port" >&2
    exit 2
fi

existing_no_proxy="${no_proxy:-${NO_PROXY:-}}"
NO_PROXY="127.0.0.1,localhost,${R2E_RAY_HEAD}"
if [ -n "${existing_no_proxy}" ]; then
    NO_PROXY="${NO_PROXY},${existing_no_proxy}"
fi
no_proxy="${NO_PROXY}"
export NO_PROXY no_proxy

case "${R2E_GYM_MODE}" in
    golden)
        R2E_GYM_VERIFY_GOLDEN_PATCH="1"
        default_submission_id="nemo-gym-r2e-golden"
        ;;
    train)
        R2E_GYM_VERIFY_GOLDEN_PATCH="0"
        default_submission_id="nemo-gym-r2e-train"
        ;;
    *)
        echo "mode must be golden or train" >&2
        exit 2
        ;;
esac

if [ -z "${RAY_CLI:-}" ]; then
    if [ -x /opt/nemo-gym/.venv/bin/ray ]; then
        RAY_CLI="/opt/nemo-gym/.venv/bin/ray"
    elif command -v ray >/dev/null 2>&1; then
        RAY_CLI="$(command -v ray)"
    else
        echo "ray CLI not found; install the cluster-compatible Ray client or run this script in the NeMo Gym image" >&2
        exit 2
    fi
fi

RAY_ADDRESS="${RAY_ADDRESS:-${R2E_RAY_HEAD}:6379}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://${R2E_RAY_HEAD}:8265}"
GYM_HOST="${GYM_HOST:-${R2E_RAY_HEAD}}"
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"
R2E_DATA_DIR="${R2E_DATA_DIR:-/data/nemo-gym/r2e-gym}"
R2E_GYM_DATA="${R2E_GYM_DATA:-${R2E_DATA_DIR}/r2e_gym_train.jsonl}"
R2E_GYM_SIF_DIR="${R2E_GYM_SIF_DIR:-${R2E_DATA_DIR}/sif}"
R2E_GYM_SIF_PREFIX="${R2E_GYM_SIF_PREFIX:-}"
R2E_GYM_SUBMISSION_ID="${R2E_GYM_SUBMISSION_ID:-${default_submission_id}-$(date -u +%Y%m%d-%H%M%S)}"
R2E_GYM_WORKING_DIR="${R2E_GYM_WORKING_DIR:-${EXAMPLE_DIR}}"
R2E_GYM_START_SCRIPT="${R2E_GYM_START_SCRIPT:-recipes/r2e-gym/start_r2e_gym_local.sh}"
R2E_GYM_MAX_CONCURRENCY="${R2E_GYM_MAX_CONCURRENCY:-1}"

echo "Submitting R2E-Gym:"
echo "  mode=${R2E_GYM_MODE}"
echo "  ray_address=${RAY_ADDRESS}"
echo "  dashboard=${RAY_DASHBOARD_ADDRESS}"
echo "  gym_host=${GYM_HOST}"
echo "  data=${R2E_GYM_DATA}"
echo "  sif_dir=${R2E_GYM_SIF_DIR}"
echo "  sif_prefix=${R2E_GYM_SIF_PREFIX:-<none>}"
echo "  submission_id=${R2E_GYM_SUBMISSION_ID}"

runtime_env="$(
    jq -cn \
        --arg ray_address "${RAY_ADDRESS}" \
        --arg callback_networks "${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-}" \
        --arg http_proxy "${http_proxy:-${HTTP_PROXY:-}}" \
        --arg https_proxy "${https_proxy:-${HTTPS_PROXY:-}}" \
        --arg no_proxy "${no_proxy:-${NO_PROXY:-}}" \
        '{
          env_vars: {
            RAY_ADDRESS: $ray_address,
            NEMO_GYM_CALLBACK_ALLOWED_NETWORKS: $callback_networks,
            HTTP_PROXY: $http_proxy,
            HTTPS_PROXY: $https_proxy,
            NO_PROXY: $no_proxy,
            http_proxy: $http_proxy,
            https_proxy: $https_proxy,
            no_proxy: $no_proxy
          }
        }'
)"

exec "${RAY_CLI}" job submit \
    --address="${RAY_DASHBOARD_ADDRESS}" \
    --submission-id="${R2E_GYM_SUBMISSION_ID}" \
    --working-dir="${R2E_GYM_WORKING_DIR}" \
    --runtime-env-json="${runtime_env}" \
    -- bash "${R2E_GYM_START_SCRIPT}" \
        --data-dir "${R2E_DATA_DIR}" \
        --mode "${R2E_GYM_MODE}" \
        --sif-dir "${R2E_GYM_SIF_DIR}" \
        --sif-prefix "${R2E_GYM_SIF_PREFIX}" \
        --max-concurrency "${R2E_GYM_MAX_CONCURRENCY}"
