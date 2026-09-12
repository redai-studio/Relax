#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

: "${GYM_HOST:?GYM_HOST must be reachable from every Relax worker}"
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
RELAX_INTEGRATION_ROOT="${RELAX_INTEGRATION_ROOT:-/opt/relax-integration}"
GYM_BIND_HOST="${GYM_BIND_HOST:-0.0.0.0}"
GYM_RAY_NUM_CPUS="${GYM_RAY_NUM_CPUS:-8}"
GYM_RAY_PORT="${GYM_RAY_PORT:-6383}"
GYM_DASHBOARD_PORT="${GYM_DASHBOARD_PORT:-29365}"
GYM_RAY_TEMP_DIR="${GYM_RAY_TEMP_DIR:-$(mktemp -d /tmp/nemo-gym-calendar-ray.XXXXXX)}"
CALENDAR_PORT_BASE="${CALENDAR_PORT_BASE:-29100}"
export CALENDAR_MAX_CONCURRENCY="${CALENDAR_MAX_CONCURRENCY:-8}"

if ! [[ "${CALENDAR_PORT_BASE}" =~ ^[0-9]+$ ]] \
    || ((10#${CALENDAR_PORT_BASE} < 1 || 10#${CALENDAR_PORT_BASE} > 65532)); then
    echo "CALENDAR_PORT_BASE must be an integer between 1 and 65532" >&2
    exit 2
fi
if ! [[ "${CALENDAR_MAX_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CALENDAR_MAX_CONCURRENCY must be a positive integer" >&2
    exit 2
fi

calendar_port_base=$((10#${CALENDAR_PORT_BASE}))
export CALENDAR_GATEWAY_PORT="${calendar_port_base}"
export CALENDAR_AGENT_PORT="$((calendar_port_base + 1))"
export CALENDAR_RESOURCE_PORT="$((calendar_port_base + 2))"
export CALENDAR_HEAD_PORT="$((calendar_port_base + 3))"
export PYTHONPATH="${RELAX_INTEGRATION_ROOT}:${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON
NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON="$(
    "${GYM_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

host = os.environ["GYM_HOST"]
agent_port = int(os.environ["CALENDAR_AGENT_PORT"])
resource_port = int(os.environ["CALENDAR_RESOURCE_PORT"])
max_concurrency = int(os.environ["CALENDAR_MAX_CONCURRENCY"])
print(
    json.dumps(
        {
            "calendar-v1": {
                "environment": "calendar",
                "agent_name": "calendar_simple_agent",
                "agent_url": f"http://{host}:{agent_port}",
                "readiness_urls": [
                    f"http://{host}:{agent_port}",
                    f"http://{host}:{resource_port}",
                ],
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

"${GYM_ROOT}/.venv/bin/ray" start --head \
    --node-ip-address="${GYM_HOST}" \
    --port="${GYM_RAY_PORT}" \
    --num-cpus="${GYM_RAY_NUM_CPUS}" \
    --dashboard-port="${GYM_DASHBOARD_PORT}" \
    --temp-dir="${GYM_RAY_TEMP_DIR}" \
    --include-dashboard=false \
    --disable-usage-stats

cd "${GYM_ROOT}"
exec "${GYM_ROOT}/.venv/bin/gym" env start \
    "+config_paths=[responses_api_models/relax_gateway_model/configs/relax_gateway_model.yaml,environments/calendar/config.yaml]" \
    +observability_enabled=true \
    ++global_aiohttp_client_request_debug=True \
    +skip_venv_if_present=true \
    +ray_head_node_address="${GYM_HOST}:${GYM_RAY_PORT}" \
    +default_host="${GYM_HOST}" \
    ++head_server.host="${GYM_BIND_HOST}" \
    ++head_server.port="${CALENDAR_HEAD_PORT}" \
    ++policy_model.responses_api_models.relax_gateway_model.host="${GYM_BIND_HOST}" \
    ++policy_model.responses_api_models.relax_gateway_model.port="${CALENDAR_GATEWAY_PORT}" \
    ++calendar_simple_agent.responses_api_agents.simple_agent.host="${GYM_BIND_HOST}" \
    ++calendar_simple_agent.responses_api_agents.simple_agent.port="${CALENDAR_AGENT_PORT}" \
    ++calendar.resources_servers.calendar.host="${GYM_BIND_HOST}" \
    ++calendar.resources_servers.calendar.port="${CALENDAR_RESOURCE_PORT}"
