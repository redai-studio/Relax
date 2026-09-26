#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

: "${GYM_HOST:?GYM_HOST must be a routable address}"
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
RELAX_INTEGRATION_ROOT="${RELAX_INTEGRATION_ROOT:-/opt/relax-integration}"
GYM_BIND_HOST="${GYM_BIND_HOST:-${GYM_HOST}}"
GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS:-16}"
GYM_RAY_PORT="${GYM_RAY_PORT:-6384}"
GYM_DASHBOARD_PORT="${GYM_DASHBOARD_PORT:-30365}"
GYM_RAY_ADDRESS="${GYM_RAY_ADDRESS:-${GYM_HOST}:${GYM_RAY_PORT}}"
GYM_START_PRIVATE_RAY="${GYM_START_PRIVATE_RAY:-1}"
GYM_RAY_TEMP_DIR="${GYM_RAY_TEMP_DIR:-$(mktemp -d /tmp/nemo-gym-multienv-ray.XXXXXX)}"
MULTIENV_PORT_BASE="${MULTIENV_PORT_BASE:-29300}"
if ! [[ "${MULTIENV_PORT_BASE}" =~ ^[0-9]+$ ]] \
    || ((10#${MULTIENV_PORT_BASE} < 1 || 10#${MULTIENV_PORT_BASE} > 65530)); then
    echo "MULTIENV_PORT_BASE must be an integer between 1 and 65530" >&2
    exit 2
fi
port_base=$((10#${MULTIENV_PORT_BASE}))
export MULTIENV_GATEWAY_PORT="${port_base}"
export MATH_AGENT_PORT="$((port_base + 1))"
export MATH_RESOURCE_PORT="$((port_base + 2))"
export WORKPLACE_AGENT_PORT="$((port_base + 3))"
export WORKPLACE_RESOURCE_PORT="$((port_base + 4))"
export MULTIENV_HEAD_PORT="$((port_base + 5))"
export MATH_MAX_CONCURRENCY="${MATH_MAX_CONCURRENCY:-256}"
export WORKPLACE_MAX_CONCURRENCY="${WORKPLACE_MAX_CONCURRENCY:-256}"
export PYTHONPATH="${RELAX_INTEGRATION_ROOT}:${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON
NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON="$(
    "${GYM_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

host = os.environ["GYM_HOST"]
math_agent_port = int(os.environ["MATH_AGENT_PORT"])
math_resource_port = int(os.environ["MATH_RESOURCE_PORT"])
math_concurrency = int(os.environ["MATH_MAX_CONCURRENCY"])
workplace_agent_port = int(os.environ["WORKPLACE_AGENT_PORT"])
workplace_resource_port = int(os.environ["WORKPLACE_RESOURCE_PORT"])
workplace_concurrency = int(os.environ["WORKPLACE_MAX_CONCURRENCY"])

print(
    json.dumps(
        {
            "math-with-judge-v1": {
                "environment": "math_with_judge",
                "agent_name": "math_with_judge_simple_agent",
                "agent_url": f"http://{host}:{math_agent_port}",
                "readiness_urls": [
                    f"http://{host}:{math_agent_port}",
                    f"http://{host}:{math_resource_port}",
                ],
                "interrupt_policy": "protected",
                "max_concurrency": math_concurrency,
                "queue_capacity": math_concurrency * 4,
                "max_deadline_s": 1800,
            },
            "workplace-assistant-v1": {
                "environment": "workplace_assistant",
                "agent_name": "workplace_assistant_simple_agent",
                "agent_url": f"http://{host}:{workplace_agent_port}",
                "readiness_urls": [
                    f"http://{host}:{workplace_agent_port}",
                    f"http://{host}:{workplace_resource_port}",
                ],
                "abort_url": f"http://{host}:{workplace_resource_port}/cleanup/{{rollout_id}}",
                "force_cleanup_url": f"http://{host}:{workplace_resource_port}/cleanup/{{rollout_id}}",
                "cleanup_probe_url": f"http://{host}:{workplace_resource_port}/cleanup/{{rollout_id}}",
                "interrupt_policy": "protected",
                "max_concurrency": workplace_concurrency,
                "queue_capacity": workplace_concurrency * 4,
                "max_deadline_s": 1800,
            },
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
    "+config_paths=[responses_api_models/relax_gateway_model/configs/relax_gateway_model.yaml,resources_servers/math_with_judge/configs/math_with_judge.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +observability_enabled=true \
    +skip_venv_if_present=true \
    +ray_head_node_address="${GYM_RAY_ADDRESS}" \
    +default_host="${GYM_HOST}" \
    ++head_server.host="${GYM_BIND_HOST}" \
    ++head_server.port="${MULTIENV_HEAD_PORT}" \
    ++policy_model.responses_api_models.relax_gateway_model.host="${GYM_BIND_HOST}" \
    ++policy_model.responses_api_models.relax_gateway_model.port="${MULTIENV_GATEWAY_PORT}" \
    ++math_with_judge_simple_agent.responses_api_agents.simple_agent.host="${GYM_BIND_HOST}" \
    ++math_with_judge_simple_agent.responses_api_agents.simple_agent.port="${MATH_AGENT_PORT}" \
    ++math_with_judge.resources_servers.math_with_judge.host="${GYM_BIND_HOST}" \
    ++math_with_judge.resources_servers.math_with_judge.port="${MATH_RESOURCE_PORT}" \
    ++workplace_assistant_simple_agent.responses_api_agents.simple_agent.host="${GYM_BIND_HOST}" \
    ++workplace_assistant_simple_agent.responses_api_agents.simple_agent.port="${WORKPLACE_AGENT_PORT}" \
    ++workplace_assistant.resources_servers.workplace_assistant.host="${GYM_BIND_HOST}" \
    ++workplace_assistant.resources_servers.workplace_assistant.port="${WORKPLACE_RESOURCE_PORT}"
