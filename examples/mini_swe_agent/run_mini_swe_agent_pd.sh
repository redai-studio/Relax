#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Mini SWE Agent with two prefill and two decode engines, each TP2 (8 GPUs).

set -eo pipefail

if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
   echo "This example requires Bash 5.1 or newer." >&2
   exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# The Ray driver owns the agent server, including when submission is asynchronous.
# This internal mode receives the complete training command from the submitter.
if [[ "${1:-}" == "--ray-driver" ]]; then
   shift
   cd "${RELAX_DIR}"
   AGENT_SERVER_PID=""
   TRAIN_PID=""
   RAY_STOP=0
   cleanup() {
      trap - EXIT
      trap '' TERM INT
      if [[ -n "${TRAIN_PID}" ]]; then
         if [[ "${RAY_STOP}" == 0 ]]; then kill -TERM "${TRAIN_PID}" 2>/dev/null || true; fi
         wait "${TRAIN_PID}" 2>/dev/null || true
      fi
      if [[ -n "${AGENT_SERVER_PID}" ]]; then
         if [[ "${RAY_STOP}" == 0 ]]; then kill -TERM "${AGENT_SERVER_PID}" 2>/dev/null || true; fi
         wait "${AGENT_SERVER_PID}" 2>/dev/null || true
      fi
   }
   trap cleanup EXIT
   # Ray signals the driver AND its descendants; do not signal train.py twice.
   trap 'RAY_STOP=1; exit 143' TERM
   trap 'RAY_STOP=1; exit 130' INT

   # Fail before starting training if another service owns this port.
   python3 - <<'PORT'
import os
import socket
with socket.socket() as sock:
    sock.bind((os.environ["AGENT_SERVER_HOST"], int(os.environ["AGENT_SERVER_PORT"])))
PORT
   (
      export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
      export MSWEA_CONFIGURED=true MSWEA_SILENT_STARTUP=true MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1
      exec "${MINI_SWE_AGENT_VENV}/bin/python" "${SCRIPT_DIR}/agent_server.py" \
         --host "${AGENT_SERVER_HOST}" --port "${AGENT_SERVER_PORT}"
   ) >"${AGENT_SERVER_WORK_DIR}/server.log" 2>&1 &
   AGENT_SERVER_PID=$!
   ready=0
   for ((attempt=0; attempt<120; attempt++)); do
      kill -0 "${AGENT_SERVER_PID}" 2>/dev/null || { cat "${AGENT_SERVER_WORK_DIR}/server.log" >&2; exit 1; }
      # The existing server has no health route; a Flask 404 proves HTTP readiness.
      if [[ "$(curl --noproxy '*' -s -o /dev/null -w '%{http_code}' --max-time 1 "${AGENT_SERVER_URL}/" || true)" == 404 ]]; then
         ready=1
         break
      fi
      sleep 1
   done
   [[ "${ready}" == 1 ]] || { echo "Agent server startup timed out; see ${AGENT_SERVER_WORK_DIR}/server.log" >&2; exit 1; }
   python3 "${RELAX_DIR}/relax/entrypoints/train.py" "$@" &
   TRAIN_PID=$!
   result=0
   wait -n -p finished "${TRAIN_PID}" "${AGENT_SERVER_PID}" || result=$?
   if [[ "${finished}" == "${AGENT_SERVER_PID}" ]]; then
      AGENT_SERVER_PID=""
      echo "Agent server exited before training completed; see server.log" >&2
      exit 1
   fi
   TRAIN_PID=""
   exit "${result}"
fi

if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" || -z "${RUNTIME_ENV_JSON:-}" ]]; then
   echo "Launch through scripts/entrypoint/ray-job.sh on a dedicated, single-node, 8-GPU Ray cluster." >&2
   exit 1
fi
: "${MODEL_DIR:?Set MODEL_DIR to the parent of Qwen3.5-35B-A3B}"
: "${SAVE_DIR:?Set SAVE_DIR for checkpoints}"
: "${R2E_DATA_PATH:?Set R2E_DATA_PATH to the prepared parquet data directory}"
: "${R2E_SIF_DIR:?Set R2E_SIF_DIR to the prepared SIF directory}"
: "${EXP_ROOT:?Set EXP_ROOT to a new directory for this run}"
: "${MINI_SWE_AGENT_VENV:?Set MINI_SWE_AGENT_VENV to the mini-swe-agent venv}"
[[ -x "${MINI_SWE_AGENT_VENV}/bin/python" ]] || { echo "Missing mini-swe-agent venv" >&2; exit 1; }
[[ -f "${MODEL_DIR}/Qwen3.5-35B-A3B/config.json" ]] || { echo "Missing Qwen3.5-35B-A3B model" >&2; exit 1; }
source "${RELAX_DIR}/scripts/models/qwen35-35B-A3B.sh"

RUN_ID="${RUN_ID:-mini-swe-pd-$(date +%Y%m%d-%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-Relax/mini_swe_agent_pd}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
# agent_config.yaml leases each task to exactly eight samples.
N_SAMPLES_PER_PROMPT=8
[[ "${ROLLOUT_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || { echo "ROLLOUT_BATCH_SIZE must be positive" >&2; exit 1; }
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
export AGENT_SERVER_TRAIN_CONCURRENCY="${AGENT_SERVER_TRAIN_CONCURRENCY:-${GLOBAL_BATCH_SIZE}}"
[[ "${AGENT_SERVER_TRAIN_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] && \
   ((AGENT_SERVER_TRAIN_CONCURRENCY >= GLOBAL_BATCH_SIZE)) || {
   echo "Agent server needs at least ${GLOBAL_BATCH_SIZE} train slots for all resident sessions." >&2
   exit 1
}
export AGENT_SERVER_EVAL_CONCURRENCY="${AGENT_SERVER_EVAL_CONCURRENCY:-64}"
if [[ "${ENABLE_EVAL:-0}" == 1 ]] && ((AGENT_SERVER_EVAL_CONCURRENCY < 50)); then
   echo "Enabled Eval requires at least 50 dedicated Eval slots." >&2
   exit 1
fi
export AGENT_SERVER_HOST="${MASTER_ADDR:-127.0.0.1}"
export AGENT_SERVER_PORT="${AGENT_SERVER_PORT:-8765}"
export AGENT_SERVER_URL="http://${AGENT_SERVER_HOST}:${AGENT_SERVER_PORT}"
export AGENT_SERVER_WORK_DIR="${EXP_ROOT}/agent_server"
export AGENT_SERVER_SHUFFLE="${AGENT_SERVER_SHUFFLE:-1}"
export AGENT_SERVER_SEED="${AGENT_SERVER_SEED:-42}"
export AGENT_SERVER_REWARD_CONCURRENCY="${AGENT_SERVER_REWARD_CONCURRENCY:-24}"
export MINI_SWE_AGENT_VENV R2E_DATA_PATH R2E_SIF_DIR
mkdir -p "${SAVE_DIR}" "${EXP_ROOT}/dump" "${AGENT_SERVER_WORK_DIR}"
DUMMY_DATA="${EXP_ROOT}/relax_dummy.jsonl"
seq 50 | sed 's/.*/{"input":[{"role":"user","content":""}]}/' > "${DUMMY_DATA}"

# Explicitly forward server settings: Ray Jobs do not inherit the submitter's env.
export RELAX_DIR
RUNTIME_ENV_JSON=$(python3 - <<'ENV'
import json
import os
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
keys = ["MINI_SWE_AGENT_VENV", "R2E_DATA_PATH", "R2E_SIF_DIR"]
keys += [key for key in os.environ if key.startswith("AGENT_SERVER_")]
env.update({key: os.environ[key] for key in keys})
# Ray's default stop grace is shorter than sandbox cleanup and trainer shutdown.
env["RAY_JOB_STOP_WAIT_TIME_S"] = "180"
env["PYTHONPATH"] = os.environ["RELAX_DIR"] + ":" + env.get("PYTHONPATH", "")
print(json.dumps(runtime))
ENV
)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.5-35B-A3B"
   --ref-load "${MODEL_DIR}/Qwen3.5-35B-A3B"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/Qwen3.5-35B-A3B-R2E-Gym-MiniSWEAgent-PD-Checkpoint"
   --save-interval 20
   --max-actor-ckpt-to-keep 1
)

ROLLOUT_ARGS=(
   --prompt-data "${DUMMY_DATA}"
   --dump-details "${EXP_ROOT}/dump"
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len "${MAX_RESPONSE_LEN:-8192}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN:-32768}"
   --rollout-temperature 1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --use-fault-tolerance
)

EVAL_ARGS=()
if [ "${ENABLE_EVAL:-0}" = "1" ]; then
   EVAL_ARGS=(
      --eval-interval 100
      --eval-prompt-data r2e_eval "${DUMMY_DATA}"
      --n-samples-per-eval-prompt 1
      --agentic-eval-concurrency 50
   )
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis

)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-config "${SCRIPT_DIR}/pd-2p2d-tp2.yaml"
   --sglang-mem-fraction-static 0.75
   --sglang-load-format dummy
   --sglang-router-policy consistent_hashing
   --sglang-router-prefill-policy consistent_hashing
   --sglang-router-decode-policy power_of_two
)
if [[ -n "${SGLANG_MAX_TOTAL_TOKENS:-}" ]]; then
   SGLANG_ARGS+=(--sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}")
fi

LOG_ARGS=(
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${RUN_ID}"
)

MEGATRON_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --data-pad-size-multiplier 2048
   --log-probs-max-tokens-per-gpu 20480
   --calculate-per-token-loss
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
)

RAY_RESOURCE_ARGS=(
   --resource '{"actor": [1, 8], "rollout": [1, 8]}'
   --max-staleness 0
   --num-data-storage-units 1
   --colocate
)

# agentic rollout: agent exec + parsers ──────────────────────────────────────────────────────────────
AGENTIC_ARGS=(
   --use-agentic-rollout
   --agentic-concurrency "${ROLLOUT_BATCH_SIZE}"
   --agent-command "bash ${SCRIPT_DIR}/agent_client.sh"
   --agent-cwd "${SCRIPT_DIR}"
   --agent-timeout 7200
   --agent-env "AGENT_SERVER_URL=${AGENT_SERVER_URL}" "AGENT_CLIENT_TRACE_DIR=${AGENT_SERVER_WORK_DIR}/client_events"
   --agentic-tool-call-parser qwen3_coder
   --agentic-reasoning-parser qwen3
)

ray job submit --no-wait --address "${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}" \
   --submission-id "${RUN_ID}" --runtime-env-json "${RUNTIME_ENV_JSON}" \
   -- bash "${SCRIPT_DIR}/run_mini_swe_agent_pd.sh" --ray-driver \
   "${RAY_RESOURCE_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${LOG_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${AGENTIC_ARGS[@]}" \
   "${MEGATRON_ARGS[@]}" \
   "$@" 2>&1 | tee "${EXP_ROOT}/submit.log"

echo "Job: ${RUN_ID}"
echo "Logs: ray job logs --address ${RAY_JOB_ADDRESS:-http://127.0.0.1:8265} ${RUN_ID}"
