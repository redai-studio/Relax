#!/bin/bash

# Code-generation demo: reuse the binary baseline and Relax split GenRM.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export GAGAR_GRADER_MODE=sandbox
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/gagar/deepcoder}"
export EXP_NAME="${EXP_NAME:-gagar-deepcoder-qwen35-4b-think16k-8gpu}"
: "${GAGAR_SANDBOX_IMAGE:?Set the local Python/SandboxFusion SIF path}"
: "${MINI_SWE_AGENT_VENV:?Set the host mini-swe-agent venv path}"
export GAGAR_AUDIT_DIR="${GAGAR_AUDIT_DIR:-${SAVE_DIR}/${EXP_NAME}/grader}"
export GAGAR_MODEL_PATH="${GAGAR_MODEL_PATH:-${MODEL_DIR}/Qwen3.8-27B}"
RESOURCE='{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}'
GENRM_OPTIONS=$(python3 "${SCRIPT_DIR}/app/config.py" --task code_generate --genrm-args)
readarray -t GENRM_ARGS <<<"${GENRM_OPTIONS}"
GAGAR_ARGS=(
   # Eight training GPUs: TP2, PP1, CP1, DP4.
   --context-parallel-size 1
   --group-rm
   --custom-rm-path examples.gagar.app.code_generate.group_rm.reward_group
   --reward-key score
   --eval-reward-key test_reward
   --custom-rollout-log-function-path examples.gagar.app.metrics.log_rollout
   --reward-max-concurrency 2
   --disable-grpo-std-normalization
   --genrm-model-path "${GAGAR_MODEL_PATH}"
   --genrm-num-gpus 4
   --genrm-num-gpus-per-engine 2
   "${GENRM_ARGS[@]}"
)
RUNTIME_ENV_JSON=$(python3 - <<'ENV'
import json
import os
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
env.update({key: value for key, value in os.environ.items()
         if key.startswith("GAGAR_") or key == "MINI_SWE_AGENT_VENV"})
env["RAY_JOB_STOP_WAIT_TIME_S"] = "180"
print(json.dumps(runtime))
ENV
)
export RESOURCE RUNTIME_ENV_JSON
exec bash "${SCRIPT_DIR}/run_code_generate_baseline.sh" "${GAGAR_ARGS[@]}" "$@"
