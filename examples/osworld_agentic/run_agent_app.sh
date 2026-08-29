#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Agent-side entrypoint for the OSWorld agentic session subprocess.
# Invoked by Relax via `--agent-command '. run_agent_app.sh'`, once per session.
#
# Environment provided by Relax:
#   RELAX_BASE_URL    - AgenticChatAPIService base URL (may end with '/')
#   RELAX_SESSION_ID  - opaque session key, doubles as OpenAI API key
#   RELAX_INPUT_JSON  - path to session input JSON
#   RELAX_OUTPUT_JSON - path to write session output JSON
#
# Environment provided by the training launcher (via a single --agent-env):
#   AGENT_DEBUG_LOG_DIR       - optional per-run log directory
#   OSWORLD_CONFIG_PATH       - override for app/osworld_config*.yaml
#   OPENAI_MODEL              - model name string threaded into chat completion payloads
#   OSWORLD_CACHE_DIR         - VM cache dir (defaults to /tmp/osworld_cache)

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Route the OpenAI SDK / raw httpx clients through Relax's per-session endpoint.
#
# Trailing-slash strip (memory f3c6b412): the SDK normalises, but any raw
# f"{base_url}/chat/completions" client would end up with a double slash and
# hit AgenticChatAPIService's fallback route (422 / silent success).
#
# `/v1` suffix (memory a61a163b): only `/v1/chat/completions` is stably
# registered on AgenticChatAPIService; the bare `/chat/completions` route
# is a fragile stack of decorators. Always postfix `/v1`.
export OPENAI_BASE_URL="${RELAX_BASE_URL%/}/v1"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

# Per-session stdout+stderr capture. Plain append redirect — NEVER use `tee`
# process substitution here: it leaves orphaned tee children which the
# container's non-reaping PID 1 (`sleep`) turns into zombies, exhausting host
# PIDs after enough sessions. See deepeyes_v2 run_agent_app.sh for the incident.
if [ -n "${AGENT_DEBUG_LOG_DIR:-}" ]; then
    mkdir -p "${AGENT_DEBUG_LOG_DIR}"
    AGENT_LOG_FILE="${AGENT_DEBUG_LOG_DIR}/${RELAX_SESSION_ID:-unknown}.log"
    exec >> "${AGENT_LOG_FILE}" 2>&1
fi

# Vendored trees live under vendor/; put them on PYTHONPATH so that the
# `mm_agents` / `agents` names resolve without
# monkey-patching site-packages.
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/vendor:${SCRIPT_DIR}/vendor/agents:${PYTHONPATH:-}"

exec python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
