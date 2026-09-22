#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -e
set -o pipefail

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ] || [ -z "${R2E_DATA_PATH:-}" ] || [ -z "${R2E_SIF_DIR:-}" ] || [ -z "${EXP_ROOT:-}" ]; then
   echo "ERROR: MODEL_DIR, SAVE_DIR, R2E_DATA_PATH, R2E_SIF_DIR, and EXP_ROOT must be set." >&2
   exit 1
fi
: "${MINI_SWE_AGENT_VENV:?}"

if [ ! -f "${MINI_SWE_AGENT_VENV}/bin/activate" ]; then
   echo "ERROR: mini-swe-agent venv not found: ${MINI_SWE_AGENT_VENV}" >&2
   exit 1
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
   source "${RELAX_DIR}/scripts/entrypoint/local.sh"
   set +x
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"
set +x

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/mini_swe_agent}"
EXP_NAME="qwen35-35B-A3B-r2egym-minisweagent-gpu8-${TIMESTAMP}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:=8}"
ENABLE_AGENTIC_SESSION_LIFECYCLE="${ENABLE_AGENTIC_SESSION_LIFECYCLE:-0}"
ENABLE_AGENTIC_PROGRAM_ADMISSION="${ENABLE_AGENTIC_PROGRAM_ADMISSION:-0}"

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
AGENT_SERVER_URL="http://127.0.0.1:8765"
AGENT_SERVER_WORK_DIR="${EXP_ROOT}/agent_server"
AGENT_SERVER_TRAIN_CONCURRENCY="${AGENT_SERVER_TRAIN_CONCURRENCY:=128}"
AGENT_SERVER_EVAL_CONCURRENCY="${AGENT_SERVER_EVAL_CONCURRENCY:=128}"
# Shuffle the training sample order (per-epoch reshuffle seeded by SEED+epoch).
AGENT_SERVER_SHUFFLE="${AGENT_SERVER_SHUFFLE:=1}"
AGENT_SERVER_SEED="${AGENT_SERVER_SEED:=42}"
mkdir -p "${SAVE_DIR}" "${EXP_ROOT}/dump" "${AGENT_SERVER_WORK_DIR}"

DUMMY_DATA="${SAVE_DIR}/relax_dummy.jsonl"
seq 50 | sed 's/.*/{"input":[{"role":"user","content":""}]}/' > "${DUMMY_DATA}"

(
   # Per-run log: each launch writes its own timestamped server log so runs no
   # longer share (and interleave into) a single server.log -- that made
   # cross-run debugging impossible. `server.log` stays as a symlink to the
   # newest run for convenience (`tail server.log`).
   AGENT_SERVER_LOG="${AGENT_SERVER_WORK_DIR}/server-$(date +%Y-%m-%d-%H-%M-%S).log"
   ln -sfn "$(basename "${AGENT_SERVER_LOG}")" "${AGENT_SERVER_WORK_DIR}/server.log"
   exec > "${AGENT_SERVER_LOG}" 2>&1
   source "${MINI_SWE_AGENT_VENV}/bin/activate"
   # Cap BLAS/OpenMP threads for the agent_server and the apptainer sandboxes it
   # spawns (children inherit these). The R2E tasks run pandas/numpy tests that
   # do not need multi-threaded BLAS; without this cap they inherit the Ray
   # runtime_env default of 24 (local.sh), so 128 concurrent sandboxes create
   # thousands of idle BLAS threads -> context-switch storm on the colocated
   # node. 2 is plenty for the test workloads. Does NOT affect the training
   # actors, which keep local.sh's 24.
   R2E_DATA_PATH="${R2E_DATA_PATH}" \
      R2E_SIF_DIR="${R2E_SIF_DIR}" \
      OMP_NUM_THREADS="${AGENT_SERVER_OMP_NUM_THREADS:-2}" \
      MKL_NUM_THREADS="${AGENT_SERVER_OMP_NUM_THREADS:-2}" \
      OPENBLAS_NUM_THREADS="${AGENT_SERVER_OMP_NUM_THREADS:-2}" \
      AGENT_SERVER_WORK_DIR="${AGENT_SERVER_WORK_DIR}" \
      AGENT_SERVER_TRAIN_CONCURRENCY="${AGENT_SERVER_TRAIN_CONCURRENCY}" \
      AGENT_SERVER_EVAL_CONCURRENCY="${AGENT_SERVER_EVAL_CONCURRENCY}" \
      AGENT_SERVER_SHUFFLE="${AGENT_SERVER_SHUFFLE}" \
      AGENT_SERVER_SEED="${AGENT_SERVER_SEED}" \
      AGENT_SERVER_REWARD_CONCURRENCY="${AGENT_SERVER_REWARD_CONCURRENCY:=24}" \
      MSWEA_CONFIGURED=true \
      MSWEA_SILENT_STARTUP=true \
      MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=1 \
      python "${SCRIPT_DIR}/agent_server.py" --host 127.0.0.1 --port 8765
) &
AGENT_SERVER_PID=$!
stop_agent_server() {
   kill "${AGENT_SERVER_PID}" 2>/dev/null || true
   wait "${AGENT_SERVER_PID}" 2>/dev/null || true
}
trap stop_agent_server EXIT

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B
   --megatron-to-hf-mode bridge
   # --dist-ckpt-optim-fully-reshardable
   # --load ${SAVE_DIR}/Qwen3.5-35B-A3B-R2E-Gym-MiniSWEAgent-Checkpoint
   --save ${SAVE_DIR}/Qwen3.5-35B-A3B-R2E-Gym-MiniSWEAgent-Checkpoint
   --save-interval 20
   --max-actor-ckpt-to-keep 1
)

ROLLOUT_ARGS=(
   --prompt-data "${DUMMY_DATA}"
   --dump-details "${EXP_ROOT}/dump"
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
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
   # DAPO dynamic sampling: drop groups with zero intra-group reward std
   # (all-correct / all-wrong -> zero advantage). Aligns with AReaL's
   # filter_function. The agentic step commits until rollout_batch_size groups
   # PASS the filter (committed-count gate), pulling fresh prompts to replace
   # dropped groups -- so the committed batch size is preserved. Over-sampling
   # only affects how much in-flight slack speeds up the backfill, not whether
   # it happens.
   # --dynamic-sampling-filter-path relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std
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
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.75
   --sglang-load-format dummy
)
if [[ -n "${SGLANG_MAX_TOTAL_TOKENS:-}" ]]; then
   SGLANG_ARGS+=(--sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}")
fi
if [[ "${ENABLE_AGENTIC_SESSION_LIFECYCLE}" != "0" ]]; then
   SGLANG_ARGS+=(
      --sglang-enable-session-radix-cache
      --sglang-radix-eviction-policy priority
   )
fi

LOG_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}
)

PARTIAL_ROLLOUT_ARGS=(
   --partial-rollout
   --over-sampling-batch-size 12
   --partial-rollout-max-aborted-count 3
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
   # --use-health-check
)

# agentic rollout: agent exec + parsers ──────────────────────────────────────────────────────────────
AGENTIC_ARGS=(
   --use-agentic-rollout
   --agent-command "bash ${SCRIPT_DIR}/agent_client.sh"
   --agent-cwd "${SCRIPT_DIR}"
   --agent-timeout 7200
   --agent-env "AGENT_SERVER_URL=${AGENT_SERVER_URL}" "AGENT_CLIENT_TRACE_DIR=${AGENT_SERVER_WORK_DIR}/client_events"
   --agentic-tool-call-parser qwen3_coder
   --agentic-reasoning-parser qwen3
)
if [[ "${ENABLE_AGENTIC_SESSION_LIFECYCLE}" != "0" ]]; then
   AGENTIC_ARGS+=(--agentic-session-lifecycle)
fi
if [[ "${ENABLE_AGENTIC_PROGRAM_ADMISSION}" != "0" ]]; then
   AGENTIC_ARGS+=(
      --agentic-program-admission
      --agentic-admission-headroom 0.90
      --agentic-admission-pressure-threshold 0.92
   )
fi

   # "${PARTIAL_ROLLOUT_ARGS[@]}" \
mkdir -p logs
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
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
   2>&1 | tee "logs/${EXP_NAME}.log"
