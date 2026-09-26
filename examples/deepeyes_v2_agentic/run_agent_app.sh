#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Strip the trailing slash so clients that append /chat/completions do not
# produce a double-slash URL.
export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

# Persist each session's stdout and stderr without spawning helper processes.
# Functional agent output is written separately through RELAX_OUTPUT_JSON.
if [ -n "${AGENT_DEBUG_LOG_DIR:-}" ]; then
    mkdir -p "${AGENT_DEBUG_LOG_DIR}"
    AGENT_LOG_FILE="${AGENT_DEBUG_LOG_DIR}/${RELAX_SESSION_ID:-unknown}.log"
    exec >> "${AGENT_LOG_FILE}" 2>&1
fi

exec "${DEEPEYES_V2_APP_PYTHON}" -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
