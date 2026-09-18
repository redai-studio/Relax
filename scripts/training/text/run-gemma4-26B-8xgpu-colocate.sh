#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# gemma-4-26B-A4B-it (MoE) GRPO **colocate smoke** on 8xGPU (TP=8 EP=8, actor &
# rollout time-share the same 8 cards).  Purpose: prove gemma-4 runs the full GRPO loop
# -- rollout(sglang) -> reward -> actor(megatron) train -> weight sync back to
# rollout -- reusing the exact bridge/provider/attention that SFT already landed.
# No new engine code for gemma-4; this script only wires the launch.
#
# Colocate first (not fully-async) because it exercises the SAME bridge export
# path with the least infra (one GPU cluster, no DCS coordinator), so it is the
# cheapest way to smoke-test weight sync.  Do NOT use hybrid mode: its
# UpdateWeightFromDistributed -> convert_to_hf path has no gemma branch and
# raises. See memory: gemma4-grpo-adaptation-plan.
#
# Usage:
#   MODEL_DIR=<dir with gemma-4-26B-A4B-it> \
#   DATA_DIR=<dir with dapo-math-17k> \
#   bash scripts/training/text/run-gemma4-26B-8xgpu-colocate.sh

set -ex
set -o pipefail

unset NCCL_NVLS_ENABLE

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

GPUS="${GPUS:-8}"
TP_SIZE="${TP_SIZE:-${GPUS}}"
EP_SIZE="${EP_SIZE:-${GPUS}}"
if [ "$((8 % TP_SIZE))" -ne 0 ] || [ "$((16 % TP_SIZE))" -ne 0 ]; then
    echo "ERROR: TP_SIZE=${TP_SIZE} must divide both 8 KV heads and 16 heads." >&2
    exit 1
fi
if [ "$((GPUS % TP_SIZE))" -ne 0 ]; then
    echo "ERROR: GPUS=${GPUS} is not divisible by TP_SIZE=${TP_SIZE}." >&2
    exit 1
fi
if [ "${EP_SIZE}" -ne "${GPUS}" ]; then
    echo "ERROR: EP_SIZE must equal GPUS (=${GPUS}); got ${EP_SIZE}." >&2
    exit 1
fi
if [ "$((128 % EP_SIZE))" -ne 0 ]; then
    echo "ERROR: EP_SIZE=${EP_SIZE} must divide the 128 routed experts." >&2
    exit 1
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

RUNTIME_ENV_JSON=$(printf '%s' "${RUNTIME_ENV_JSON}" \
    | jq -c '.env_vars.GEMMA4_CONVERSION_MODE = "text"')
export RUNTIME_ENV_JSON

source "${MODEL_CONFIG_DIR}/gemma4-26B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/gemma4-grpo}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the dir containing gemma-4-26B-A4B-it}"
CKPT="${CKPT:-${MODEL_DIR}/gemma-4-26B-A4B-it}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the dir containing dapo-math-17k/}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl}"

EXP_NAME="${EXP_NAME:-gemma4-26b-grpo-colocate-smoke}"
CKPT_ROOT="${CKPT_ROOT:-/data/temp}"
SAVE_DIR="${SAVE_DIR:-${CKPT_ROOT}/gemma4-26B-grpo}"
NUM_ROLLOUT="${NUM_ROLLOUT:=3}"   # smoke: a few rollout steps, not a real run

RAY_ADDRESS="${RAY_ADDRESS:-http://${MASTER_ADDR:-127.0.0.1}:${RAY_DASHBOARD_PORT:-8265}}"

WITH_REF="${WITH_REF:-1}"
REF_ARG=(); [ "${WITH_REF}" != "0" ] && REF_ARG=(--ref-load "${CKPT}")

CKPT_ARGS=(
   --hf-checkpoint "${CKPT}"
   "${REF_ARG[@]}"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --save          "${SAVE_DIR}"
   ${LOAD_ARG:+--load "${LOAD_ARG}"}
   --save-interval ${SAVE_INTERVAL:-1000000}
   --max-actor-ckpt-to-keep ${MAX_ACTOR_CKPT_TO_KEEP:-1}
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key   "${INPUT_KEY:-prompt}"
   --label-key   "${LABEL_KEY:-label}"
   --apply-chat-template
   --rollout-skip-special-tokens
   --rollout-shuffle

   --rm-type      "${RM_TYPE:-dapo}"
   --reward-key   "${REWARD_KEY:-score}"

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-8}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-4096}
   --rollout-temperature 1

   --global-batch-size ${GLOBAL_BATCH_SIZE:-32}
   --balance-data
   --use-fault-tolerance

   --get-mismatch-metrics
)

KL_ARG=(); [ "${WITH_REF}" != "0" ] && KL_ARG=(--use-kl-loss --kl-loss-coef ${KL_LOSS_COEF:-0.00} --kl-loss-type low_var_kl)

GRPO_ARGS=(
   --advantage-estimator grpo
   "${KL_ARG[@]}"
   --entropy-coef ${ENTROPY_COEF:-0.00}
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
   --custom-tis-function-path relax.backends.megatron.loss.icepop_function
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LR:-1e-6}
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
[ "${OPTIMIZER_CPU_OFFLOAD:-1}" = "1" ] && OPTIMIZER_ARGS+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --calculate-per-token-loss
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-4096}
   --log-probs-max-tokens-per-gpu ${LOG_PROBS_MAX_TOKENS_PER_GPU:-8192}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${GPUS}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.45}
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}-${now}
)

MISC_ARGS=(
   --bf16
   --attention-backend auto
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --use-health-check
   --distributed-timeout-minutes ${DISTRIBUTED_TIMEOUT_MINUTES:-30}
)

mkdir -p "${SAVE_DIR}" log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"actor\": [1, ${GPUS}], \"rollout\": [1, ${GPUS}]}" \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee log/${EXP_NAME}-${now}.log
