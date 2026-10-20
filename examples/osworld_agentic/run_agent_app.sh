#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# One subprocess per Relax-managed session; RELAX_* values come from Relax.
set -eu
set +x  # Private configuration must not appear in per-session logs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
agent_python="${OSWORLD_AGENT_PYTHON:-}"
if [ -f "${SCRIPT_DIR}/env.local.sh" ]; then
    source "${SCRIPT_DIR}/env.local.sh"
fi
# The launcher's resolved environment takes precedence over private defaults.
export OSWORLD_AGENT_PYTHON="${agent_python:-${OSWORLD_AGENT_PYTHON:-${DATA_DIR:?Set DATA_DIR}/osworld-agent-venv/bin/python}}"
unset agent_python
: "${OSWORLD_REPO:?Set OSWORLD_REPO in env.local.sh}"
export OSWORLD_CACHE_DIR="${OSWORLD_CACHE_DIR:-${OSWORLD_REPO}/cache}"
export MCP_SRC_ROOT="${MCP_SRC_ROOT:-${OSWORLD_REPO}}"
# Do not fall back to deployment-specific defaults in the external Nex client.
: "${NEX_API_BASE:?Set NEX_API_BASE in env.local.sh}"
: "${NEX_WORKSPACE_ID:?Set NEX_WORKSPACE_ID in env.local.sh}"
: "${NEX_API_KEY:?Set NEX_API_KEY in env.local.sh or the worker environment}"
if [ -z "${NEX_TEMPLATE_ID:-}${NEX_TEMPLATE:-}" ]; then
    echo "Set NEX_TEMPLATE or NEX_TEMPLATE_ID in env.local.sh." >&2
    exit 1
fi
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

exec "${OSWORLD_AGENT_PYTHON}" -B -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
