#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

: "${NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON:?NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON must be set}"
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
RELAX_INTEGRATION_ROOT="${RELAX_INTEGRATION_ROOT:-/opt/relax-integration}"
GATEWAY_PYTHON="${GATEWAY_PYTHON:-${GYM_ROOT}/responses_api_models/relax_gateway_model/.venv/bin/python}"
export PYTHONPATH="${RELAX_INTEGRATION_ROOT}:${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${GATEWAY_PYTHON}" -m uvicorn \
  examples.nemo_gym_agentic.service.app:create_app_from_env \
  --factory \
  --host "${NEMO_GYM_GATEWAY_HOST:-0.0.0.0}" \
  --port "${NEMO_GYM_GATEWAY_PORT:-${GYM_PORT:-8000}}" \
  --workers 1
