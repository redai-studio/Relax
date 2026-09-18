#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

: "${GYM_HOST:?GYM_HOST must be reachable from every Relax worker}"
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"
: "${REASONING_GYM_CC_DATA_JSONL:?Set the prepared reasoning-gym JSONL path}"

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
RELAX_INTEGRATION_ROOT="${RELAX_INTEGRATION_ROOT:-/opt/relax-integration}"
GYM_BIND_HOST="${GYM_BIND_HOST:-${GYM_HOST}}"
GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS:-8}"
GYM_RAY_PORT="${GYM_RAY_PORT:-6386}"
GYM_DASHBOARD_PORT="${GYM_DASHBOARD_PORT:-31509}"
GYM_RAY_ADDRESS="${GYM_RAY_ADDRESS:-${GYM_HOST}:${GYM_RAY_PORT}}"
GYM_START_PRIVATE_RAY="${GYM_START_PRIVATE_RAY:-1}"
GYM_RAY_TEMP_DIR="${GYM_RAY_TEMP_DIR:-$(mktemp -d /tmp/nemo-gym-reasoning-gym-cc-ray.XXXXXX)}"
REASONING_GYM_CC_PORT_BASE="${REASONING_GYM_CC_PORT_BASE:-29200}"
REASONING_GYM_CC_MAX_CONCURRENCY="${REASONING_GYM_CC_MAX_CONCURRENCY:-64}"
REASONING_GYM_CC_MAX_TURNS="${REASONING_GYM_CC_MAX_TURNS:-30}"
REASONING_GYM_CC_AGENT_TIMEOUT="${REASONING_GYM_CC_AGENT_TIMEOUT:-300}"
REASONING_GYM_CC_MAX_DEADLINE_S="${REASONING_GYM_CC_MAX_DEADLINE_S:-900}"
REASONING_GYM_CC_SETTINGS="${REASONING_GYM_CC_SETTINGS:-${RELAX_INTEGRATION_ROOT}/examples/nemo_gym_agentic/recipes/reasoning-gym-cc/configs/claude_settings.json}"

command -v bwrap >/dev/null
command -v socat >/dev/null
test -s "${REASONING_GYM_CC_SETTINGS}"
export PATH="${GYM_ROOT}/responses_api_agents/claude_code_agent/.claude_node/bin:${PATH}"
if [ "$(claude --version | awk '{print $1}')" != "2.1.237" ]; then
    echo "Claude Code 2.1.237 is required" >&2
    exit 2
fi

# Disable Claude Code CLI's bubblewrap sandbox for the Bash tool, and tell the
# CLI to skip its refuse-to-run-as-root guard.
#
# Why:
#   bwrap 0.9 + kernel 5.10.134 (TencentOS an8) reject the seccomp filter
#   (prctl(PR_SET_SECCOMP) EINVAL / apply-seccomp errno 524), so every Bash
#   tool call under the sandbox returns "Exit code 1" — see the 2026-09-09
#   rollout-11 analysis (215/217 tool responses failed).
#
# Two edits to the env dict in ${GYM_ROOT}/.../claude_code_agent/app.py:
#   1. IS_SANDBOX "1" -> "0" : CLI no longer starts bwrap around the Bash tool.
#   2. Insert CLAUDE_CODE_BUBBLEWRAP "1" alongside it : CLI's root check is
#      `getuid()==0 && IS_SANDBOX!="1" && !CLAUDE_CODE_BUBBLEWRAP`; without
#      this, --dangerously-skip-permissions refuses to run under root and
#      claude-code exits 1 with "produced no assistant message".
#
# Runs before the agent server starts, so the patched values are imported.
# Idempotent; hard-fails if the upstream env dict moves so we notice on the
# next image bump.
_CLAUDE_APP_PY="${GYM_ROOT}/responses_api_agents/claude_code_agent/app.py"
if grep -q '"IS_SANDBOX": "1"' "${_CLAUDE_APP_PY}"; then
    sed -i 's|"IS_SANDBOX": "1",|"IS_SANDBOX": "0",\n                "CLAUDE_CODE_BUBBLEWRAP": "1",|' \
        "${_CLAUDE_APP_PY}"
elif ! grep -q '"CLAUDE_CODE_BUBBLEWRAP": "1"' "${_CLAUDE_APP_PY}"; then
    echo "IS_SANDBOX / CLAUDE_CODE_BUBBLEWRAP toggle not found in ${_CLAUDE_APP_PY}; upstream nemo-gym layout changed" >&2
    exit 2
fi
unset _CLAUDE_APP_PY

export REASONING_GYM_CC_SETTINGS

if ! [[ "${REASONING_GYM_CC_PORT_BASE}" =~ ^[0-9]+$ ]] \
    || ((10#${REASONING_GYM_CC_PORT_BASE} < 1 || 10#${REASONING_GYM_CC_PORT_BASE} > 65532)); then
    echo "REASONING_GYM_CC_PORT_BASE must be an integer between 1 and 65532" >&2
    exit 2
fi
port_base=$((10#${REASONING_GYM_CC_PORT_BASE}))
export REASONING_GYM_CC_GATEWAY_PORT="${port_base}"
export REASONING_GYM_CC_AGENT_PORT="$((port_base + 1))"
export REASONING_GYM_CC_RESOURCE_PORT="$((port_base + 2))"
export REASONING_GYM_CC_HEAD_PORT="$((port_base + 3))"
export REASONING_GYM_CC_MAX_CONCURRENCY REASONING_GYM_CC_MAX_TURNS
export REASONING_GYM_CC_AGENT_TIMEOUT REASONING_GYM_CC_MAX_DEADLINE_S
export REASONING_GYM_CC_DATA_JSONL
export PYTHONPATH="${RELAX_INTEGRATION_ROOT}:${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

recipe_config="${RELAX_INTEGRATION_ROOT}/examples/nemo_gym_agentic/recipes/reasoning-gym-cc/configs/reasoning_gym_cc.yaml"
test -s "${REASONING_GYM_CC_DATA_JSONL}"
test -s "${recipe_config}"

export NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON
NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON="$(
    "${GYM_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

host = os.environ["GYM_HOST"]
agent_port = int(os.environ["REASONING_GYM_CC_AGENT_PORT"])
resource_port = int(os.environ["REASONING_GYM_CC_RESOURCE_PORT"])
max_concurrency = int(os.environ["REASONING_GYM_CC_MAX_CONCURRENCY"])
max_deadline_s = int(os.environ.get("REASONING_GYM_CC_MAX_DEADLINE_S", "900"))
print(
    json.dumps(
        {
            "reasoning-gym-cc-v1": {
                "environment": "reasoning_gym",
                "agent_name": "reasoning_gym_claude_code_agent_model_server",
                "agent_url": f"http://{host}:{agent_port}",
                "readiness_urls": [f"http://{host}:{agent_port}", f"http://{host}:{resource_port}"],
                "abort_url": f"http://{host}:{agent_port}/cleanup/{{rollout_id}}",
                "force_cleanup_url": f"http://{host}:{agent_port}/cleanup/{{rollout_id}}",
                "cleanup_probe_url": f"http://{host}:{agent_port}/cleanup/{{rollout_id}}",
                "interrupt_policy": "protected",
                "max_concurrency": max_concurrency,
                "queue_capacity": max_concurrency * 4,
                "max_deadline_s": max_deadline_s,
            }
        },
        separators=(",", ":"),
    )
)
PY
)"

unset RAY_ADDRESS RAY_JOB_SUBMISSION_ID
if [ "${GYM_START_PRIVATE_RAY}" = "1" ]; then
    "${GYM_ROOT}/.venv/bin/ray" start --head \
        --node-ip-address="${GYM_HOST}" \
        --port="${GYM_RAY_PORT}" \
        --num-cpus="${GYM_RAY_NUM_CPUS}" \
        --dashboard-port="${GYM_DASHBOARD_PORT}" \
        --temp-dir="${GYM_RAY_TEMP_DIR}" \
        --include-dashboard=false \
        --disable-usage-stats
elif [ "${GYM_START_PRIVATE_RAY}" != "0" ]; then
    echo "GYM_START_PRIVATE_RAY must be 0 or 1" >&2
    exit 2
fi

cd "${GYM_ROOT}"
exec "${GYM_ROOT}/.venv/bin/gym" env start \
    "+config_paths=[responses_api_models/relax_gateway_model/configs/relax_gateway_model.yaml,${recipe_config}]" \
    +observability_enabled=true \
    +skip_venv_if_present=true \
    +ray_head_node_address="${GYM_RAY_ADDRESS}" \
    +default_host="${GYM_HOST}" \
    ++head_server.host="${GYM_BIND_HOST}" \
    ++head_server.port="${REASONING_GYM_CC_HEAD_PORT}" \
    ++policy_model.responses_api_models.relax_gateway_model.host="${GYM_BIND_HOST}" \
    ++policy_model.responses_api_models.relax_gateway_model.port="${REASONING_GYM_CC_GATEWAY_PORT}" \
    ++reasoning_gym_claude_code_agent_model_server.responses_api_agents.claude_code_agent.host="${GYM_BIND_HOST}" \
    ++reasoning_gym_claude_code_agent_model_server.responses_api_agents.claude_code_agent.port="${REASONING_GYM_CC_AGENT_PORT}" \
    ++reasoning_gym.resources_servers.reasoning_gym.host="${GYM_BIND_HOST}" \
    ++reasoning_gym.resources_servers.reasoning_gym.port="${REASONING_GYM_CC_RESOURCE_PORT}"
