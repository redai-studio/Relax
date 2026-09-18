#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# One subprocess per Relax-managed session; RELAX_* values come from Relax.
set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[ ! -f "${SCRIPT_DIR}/env.local.sh" ] || source "${SCRIPT_DIR}/env.local.sh"
source "${SCRIPT_DIR}/env.sh"
cd "${SCRIPT_DIR}"

# Normalize the URL before adding the endpoint prefix.
export OPENAI_BASE_URL="${RELAX_BASE_URL%/}/v1"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"
export PYTHONPATH="${SCRIPT_DIR}:${OSWORLD_REPO}:${PYTHONPATH:-}"

# Direct redirection avoids orphaned tee subprocesses across many sessions.
if [ -n "${AGENT_DEBUG_LOG_DIR:-}" ]; then
    mkdir -p "${AGENT_DEBUG_LOG_DIR}"
    exec >> "${AGENT_DEBUG_LOG_DIR}/${RELAX_SESSION_ID}.log" 2>&1
fi

exec python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
