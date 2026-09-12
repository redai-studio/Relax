#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

# This launcher belongs to the Workplace Assistant recipe.
: "${GYM_HOST:?GYM_HOST must be a routable address}"
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
RELAX_INTEGRATION_ROOT="${RELAX_INTEGRATION_ROOT:-/opt/relax-integration}"
GYM_BIND_HOST="${GYM_BIND_HOST:-${GYM_HOST}}"
GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS:-8}"
GYM_RAY_PORT="${GYM_RAY_PORT:-6382}"
GYM_DASHBOARD_PORT="${GYM_DASHBOARD_PORT:-28365}"
GYM_RAY_ADDRESS="${GYM_RAY_ADDRESS:-${GYM_HOST}:${GYM_RAY_PORT}}"
GYM_START_PRIVATE_RAY="${GYM_START_PRIVATE_RAY:-1}"
GYM_RAY_TEMP_DIR="${GYM_RAY_TEMP_DIR:-$(mktemp -d /tmp/nemo-gym-workplace-ray.XXXXXX)}"
WORKPLACE_ASSISTANT_PORT_BASE="${WORKPLACE_ASSISTANT_PORT_BASE:-29000}"
if ! [[ "${WORKPLACE_ASSISTANT_PORT_BASE}" =~ ^[0-9]+$ ]] \
    || ((10#${WORKPLACE_ASSISTANT_PORT_BASE} < 1 || 10#${WORKPLACE_ASSISTANT_PORT_BASE} > 65532)); then
    echo "WORKPLACE_ASSISTANT_PORT_BASE must be an integer between 1 and 65532" >&2
    exit 2
fi
workplace_port_base=$((10#${WORKPLACE_ASSISTANT_PORT_BASE}))
export WORKPLACE_ASSISTANT_GATEWAY_PORT="${workplace_port_base}"
export WORKPLACE_ASSISTANT_AGENT_PORT="$((workplace_port_base + 1))"
export WORKPLACE_ASSISTANT_RESOURCE_PORT="$((workplace_port_base + 2))"
export WORKPLACE_ASSISTANT_HEAD_PORT="$((workplace_port_base + 3))"
export WORKPLACE_ASSISTANT_MAX_CONCURRENCY="${WORKPLACE_ASSISTANT_MAX_CONCURRENCY:-8}"
export PYTHONPATH="${RELAX_INTEGRATION_ROOT}:${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON
NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON="$(
    "${GYM_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

host = os.environ["GYM_HOST"]
max_concurrency = int(os.environ["WORKPLACE_ASSISTANT_MAX_CONCURRENCY"])
agent_port = int(os.environ["WORKPLACE_ASSISTANT_AGENT_PORT"])
resource_port = int(os.environ["WORKPLACE_ASSISTANT_RESOURCE_PORT"])
print(
    json.dumps(
        {
            "workplace-assistant-v1": {
                "environment": "workplace_assistant",
                "agent_name": "workplace_assistant_simple_agent",
                "agent_url": f"http://{host}:{agent_port}",
                "readiness_urls": [
                    f"http://{host}:{agent_port}",
                    f"http://{host}:{resource_port}",
                ],
                "abort_url": f"http://{host}:{resource_port}/cleanup/{{rollout_id}}",
                "force_cleanup_url": f"http://{host}:{resource_port}/cleanup/{{rollout_id}}",
                "cleanup_probe_url": f"http://{host}:{resource_port}/cleanup/{{rollout_id}}",
                "interrupt_policy": "protected",
                "max_concurrency": max_concurrency,
                "queue_capacity": max_concurrency * 4,
                "max_deadline_s": 1800,
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
    "+config_paths=[responses_api_models/relax_gateway_model/configs/relax_gateway_model.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +observability_enabled=true \
    +skip_venv_if_present=true \
    +ray_head_node_address="${GYM_RAY_ADDRESS}" \
    +default_host="${GYM_HOST}" \
    ++head_server.host="${GYM_BIND_HOST}" \
    ++head_server.port="${WORKPLACE_ASSISTANT_HEAD_PORT}" \
    ++policy_model.responses_api_models.relax_gateway_model.host="${GYM_BIND_HOST}" \
    ++policy_model.responses_api_models.relax_gateway_model.port="${WORKPLACE_ASSISTANT_GATEWAY_PORT}" \
    ++workplace_assistant_simple_agent.responses_api_agents.simple_agent.host="${GYM_BIND_HOST}" \
    ++workplace_assistant_simple_agent.responses_api_agents.simple_agent.port="${WORKPLACE_ASSISTANT_AGENT_PORT}" \
    ++workplace_assistant.resources_servers.workplace_assistant.host="${GYM_BIND_HOST}" \
    ++workplace_assistant.resources_servers.workplace_assistant.port="${WORKPLACE_ASSISTANT_RESOURCE_PORT}"
