#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

# This launcher belongs to the GSM8K recipe.
: "${GYM_HOST:?GYM_HOST must be a routable address}"
export NEMO_GYM_CALLBACK_ALLOWED_NETWORKS="${NEMO_GYM_CALLBACK_ALLOWED_NETWORKS:-10.0.0.0/8}"

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
RELAX_INTEGRATION_ROOT="${RELAX_INTEGRATION_ROOT:-/opt/relax-integration}"
GYM_RAY_PORT="${GYM_RAY_PORT:-6381}"
GYM_DASHBOARD_PORT="${GYM_DASHBOARD_PORT:-28365}"
export PYTHONPATH="${RELAX_INTEGRATION_ROOT}:${GYM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON
NEMO_GYM_GATEWAY_ENVIRONMENTS_JSON="$(
    "${GYM_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

host = os.environ["GYM_HOST"]
print(
    json.dumps(
        {
            "gsm8k-v1": {
                "environment": "gsm8k",
                "agent_name": "gsm8k_math_with_judge_simple_agent",
                "agent_url": f"http://{host}:28101",
                "readiness_urls": [
                    f"http://{host}:28101",
                    f"http://{host}:28102",
                ],
                "interrupt_policy": "protected",
                "max_concurrency": 8,
                "queue_capacity": 32,
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
    --dashboard-port="${GYM_DASHBOARD_PORT}" \
    --include-dashboard=false \
    --disable-usage-stats

cd "${GYM_ROOT}"
exec "${GYM_ROOT}/.venv/bin/gym" env start \
    "+config_paths=[responses_api_models/relax_gateway_model/configs/relax_gateway_model.yaml,benchmarks/gsm8k/config.yaml]" \
    +observability_enabled=true \
    +skip_venv_if_present=true \
    +ray_head_node_address="${GYM_HOST}:${GYM_RAY_PORT}" \
    +default_host="${GYM_HOST}" \
    ++head_server.host="${GYM_HOST}" \
    ++head_server.port=28103 \
    ++policy_model.responses_api_models.relax_gateway_model.host="${GYM_HOST}" \
    ++policy_model.responses_api_models.relax_gateway_model.port=28100 \
    ++math_with_judge_simple_agent.responses_api_agents.simple_agent.host="${GYM_HOST}" \
    ++math_with_judge_simple_agent.responses_api_agents.simple_agent.port=28101 \
    ++math_with_judge.resources_servers.math_with_judge.host="${GYM_HOST}" \
    ++math_with_judge.resources_servers.math_with_judge.port=28102
