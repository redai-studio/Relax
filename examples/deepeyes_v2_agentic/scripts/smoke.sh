#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Single-sample smoke: drive ONE trajectory through the full agent app
# (app/agent.py + sandbox + tools) against an external OpenAI-compatible
# chat endpoint. No Ray, no training, no judge service.
#
# Run this FIRST to verify the agent loop / sandbox / message wiring before
# launching the cluster training script.
#
# Required env (set in env.sh):
#   DATA_DIR             workspace produced by scripts/prepare.sh
#                        (must contain data/smoke.parquet + sif/deepeyes_v2_kernel.sif)
#   OPENAI_BASE_URL      OpenAI-compatible chat endpoint (e.g. http://host:30000/v1)
#   OPENAI_API_KEY       any string (SGLang doesn't validate)
#
# Optional:
#   SMOKE_ROW            row index in smoke.parquet to drive (default: 0)

set -eu
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
# shellcheck source=/dev/null
[ -f "${EXAMPLE_DIR}/env.sh" ] && source "${EXAMPLE_DIR}/env.sh"

for var in DATA_DIR OPENAI_BASE_URL OPENAI_API_KEY; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: ${var} not set. See env.sh / env.sh.example."
        exit 1
    fi
done

SMOKE_PARQUET="${DATA_DIR}/data/smoke.parquet"
if [ ! -f "${SMOKE_PARQUET}" ]; then
    echo "ERROR: ${SMOKE_PARQUET} not found. Run: DATA_DIR=${DATA_DIR} bash ${SCRIPT_DIR}/prepare.sh"
    exit 1
fi
APP_PYTHON="${DEEPEYES_V2_APP_ENV_ROOT:-/tmp/deepeyes-v2-app-env}/.venv/bin/python"
if [ ! -x "${APP_PYTHON}" ]; then
    echo "ERROR: DeepEyes V2 app environment not found. Run: bash ${SCRIPT_DIR}/prepare_app_env.sh"
    exit 1
fi

exec "${APP_PYTHON}" "${SCRIPT_DIR}/run_single_session.py" \
    --parquet "${SMOKE_PARQUET}" \
    --row "${SMOKE_ROW:-0}"
