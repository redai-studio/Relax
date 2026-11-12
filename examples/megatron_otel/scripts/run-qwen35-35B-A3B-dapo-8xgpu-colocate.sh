#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B DAPO-math RL and native Megatron OTel smoke run.
# Scope: one node, eight GPUs, colocate, BF16, TP4/PP1/CP1/EP8/ETP1.
# See ../README.md for the candidate image, assets, and verification boundary.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RAY_DEDUP_LOGS=0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

: "${MODEL_DIR:?Set MODEL_DIR to the parent directory containing Qwen3.5-35B-A3B}"
: "${DATA_DIR:?Set DATA_DIR to the parent directory containing dapo-math-17k}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3.5-35B-A3B}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl}"
PROJECT_NAME="${PROJECT_NAME:-Relax/observability/megatron-otel}"
EXP_NAME="${EXP_NAME:-qwen35-35b-dapo-8gpu-otel-smoke}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES="${N_SAMPLES:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES))}"
now=$(date "+%Y%m%d-%H%M%S")

# All workers must inherit the same run ID. The native auto-generated fallback
# is process-local outside a scheduler and is unsuitable for Ray correlation.
export NEMO_LENS_RUN_ID="${NEMO_LENS_RUN_ID:-${EXP_NAME}-${now}-$$}"
export MEGATRON_OTEL_ENABLED="${MEGATRON_OTEL_ENABLED:-1}"
export MEGATRON_OTEL_EXPORTER="${MEGATRON_OTEL_EXPORTER:-console}"
export MEGATRON_OTEL_SPAN_GROUPS="${MEGATRON_OTEL_SPAN_GROUPS:-per_step,microbatch}"
export MEGATRON_OTEL_EXPORT_STRATEGY="${MEGATRON_OTEL_EXPORT_STRATEGY:-single_rank}"
export MEGATRON_OTEL_EXPORT_RANK="${MEGATRON_OTEL_EXPORT_RANK:-0}"
export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-relax-megatron-smoke}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --use-streaming-dataset
    --streaming-buffer-size 64
    --prefetch-max-cached 0
    --rm-type dapo
    --reward-key score
    --reward-num-workers 2
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES}"
    --rollout-max-prompt-len "${PROMPT_LEN:-4096}"
    --rollout-max-response-len "${RESPONSE_LEN:-512}"
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU:-8192}"
    --moe-token-dispatcher-type alltoall
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.0
    --entropy-coef 0.0
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
    --selective-offload
    --no-rope-fusion
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
    --sglang-disable-custom-all-reduce
    --sglang-cuda-graph-backend-decode disabled
    --sglang-cuda-graph-backend-prefill disabled
)

METRICS_ARGS=(
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
if [ "${USE_CLEARML:-0}" = 1 ]; then
    METRICS_ARGS+=(--use-clearml)
fi

MISC_ARGS=(
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --skip-eval-before-train
    --custom-megatron-init-path relax.backends.megatron.telemetry.initialize
    --tq-rdma-mode off
)

TRAIN_ARGS=(
    --resource '{"actor": [1, 8], "rollout": [1, 8]}'
    --colocate
    --max-staleness 0
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "$@"
)

# A dry run never sources a launcher or starts/stops Ray processes.
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi
test -f "${HF_CHECKPOINT}/config.json"
test -f "${PROMPT_SET}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
# The entrypoint enables xtrace. Do not echo OTLP authorization headers.
set +x
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
env_vars = runtime.setdefault("env_vars", {})
env_vars["RAY_DEDUP_LOGS"] = "0"
env_vars.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")
for key, value in os.environ.items():
    if key.startswith(("MEGATRON_OTEL_", "NEMO_LENS_", "OTEL_")):
        env_vars[key] = value
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON

mkdir -p "${SCRIPT_DIR}/log"
ray job submit \
    ${RAY_NO_WAIT:+--no-wait} \
    --address="http://${HOST_IP}:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
