#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Per-session WebShop agent launcher. Relax spawns this once per rollout session
# via `--agent-command`, injecting RELAX_BASE_URL / RELAX_SESSION_ID /
# RELAX_INPUT_JSON / RELAX_OUTPUT_JSON.
#
# The cluster-shared WebShop server is started independently before training. This
# command launches the thin per-session client that connects to it.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Some nodes have a global http_proxy/https_proxy that routes EVEN 127.0.0.1 and
# internal cluster IPs through an external (squid) proxy -> "Connection refused"
# to the local WebShop env server and to the internal LLM endpoint, so agent
# sessions launch but can never connect (stuck at launch_submitted, 0 warming).
# The agent only talks to localhost + internal IPs, so bypass the proxy entirely.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"
export NO_PROXY="${no_proxy}"

export WEBSHOP_PORT="${WEBSHOP_PORT:-36001}"
export WEBSHOP_URL="${WEBSHOP_URL:-http://127.0.0.1:${WEBSHOP_PORT}}"

# Policy endpoint (SGLang). Trailing-slash strip avoids a double-slash 404 if any
# client builds f"{base_url}/chat/completions".
export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

exec python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
