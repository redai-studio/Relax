#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5-35B-A3B CPT 128K H800 DeepEP experiment: 16 nodes x 8 GPUs.
# Defaults: TP2/PP2(22/18)/CP2/EP8/DP16, GBS256, dynamic 64K-token
# microbatches, full recompute, flex/DeepEP MoE dispatch, and MTP disabled.
#
# Usage from the Ray head node:
#   NUM_NODES=16 GPUS_PER_NODE=8 bash scripts/entrypoint/ray-job.sh \
#     scripts/training/cpt/run-qwen3.5-35B-A3B-cpt-128k-128xgpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/../../models/qwen35-35B-A3B.sh"

CPT_ROOT="${CPT_ROOT:-${REPO_ROOT}/../Relax-CPT-data}"
HF_CHECKPOINT="${HF_CHECKPOINT:-${REPO_ROOT}/../Qwen3.5-35B-A3B}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}"
PROMPT_DATA="${PROMPT_DATA:-${CPT_ROOT}/cpt-real-200-20260909/train-128k-smoke-256.jsonl}"
TRAIN_STEPS="${TRAIN_STEPS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-65536}"
SEQ_LENGTH="${SEQ_LENGTH:-131072}"
CP_SIZE="${CP_SIZE:-2}"
((128 % (2 * 2 * CP_SIZE) == 0)) || {
    echo "TP2 * PP2 * CP_SIZE must divide 128 GPUs" >&2
    exit 1
}
DP_SIZE=$((128 / (2 * 2 * CP_SIZE)))
LR_DECAY_STEPS="${LR_DECAY_STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
RUN_NAME="${RUN_NAME:-qwen35-cpt-128k-128gpu-$(date +%Y%m%d-%H%M%S)}"
TB_PROJECT_NAME="${TB_PROJECT_NAME:-qwen35-cpt}"
TB_EXPERIMENT_NAME="${TB_EXPERIMENT_NAME:-${RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CPT_ROOT}/outputs/cpt-perf}"
LOG_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train.log"
SUBMIT_LOG="${LOG_DIR}/submit.log"

[[ -f "${HF_CHECKPOINT}/config.json" ]] || { echo "Missing model: ${HF_CHECKPOINT}" >&2; exit 1; }
[[ -f "${PROMPT_DATA}" ]] || { echo "Missing dataset: ${PROMPT_DATA}" >&2; exit 1; }
[[ -d "${MEGATRON_DIR}/megatron/core" ]] || { echo "Missing Megatron: ${MEGATRON_DIR}" >&2; exit 1; }
for value in "${TRAIN_STEPS}" "${GLOBAL_BATCH_SIZE}" "${MAX_TOKENS_PER_GPU}" "${SEQ_LENGTH}" "${LR_DECAY_STEPS}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Batch, token and step values must be positive integers" >&2; exit 1; }
done
[[ "${WARMUP_STEPS}" =~ ^[0-9]+$ ]] || { echo "WARMUP_STEPS must be a non-negative integer" >&2; exit 1; }
((GLOBAL_BATCH_SIZE % DP_SIZE == 0)) || {
    echo "GLOBAL_BATCH_SIZE must be divisible by DP_SIZE=${DP_SIZE}" >&2
    exit 1
}

CPT_MODEL_ARGS=()
SKIP_MODEL_ARG=0
for arg in "${MODEL_ARGS[@]}"; do
    if ((SKIP_MODEL_ARG)); then
        SKIP_MODEL_ARG=0
        continue
    fi
    [[ "${arg}" == --use-gated-attention ]] && continue
    if [[ "${arg}" == --moe-token-dispatcher-type ]]; then
        SKIP_MODEL_ARG=1
        continue
    fi
    CPT_MODEL_ARGS+=("${arg}")
done

CMD=(
    python3 -m relax.entrypoints.train
    --resource '{"sft": [1, 0], "actor": [1, 128]}'
    "${CPT_MODEL_ARGS[@]}"
    --hf-checkpoint "${HF_CHECKPOINT}" --megatron-to-hf-mode bridge
    --loss-type sft --sft-training-mode cpt --sft-cpt-template qwen3_5
    --tool-key "" --metadata-key ""
    --prompt-data "${PROMPT_DATA}" --input-key auto --sft-oversize-strategy truncate_right
    --num-rollout "${TRAIN_STEPS}" --seed 42
    --rollout-batch-size "${GLOBAL_BATCH_SIZE}" --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --n-samples-per-prompt 1 --num-steps-per-rollout 1 --micro-batch-size 1
    --use-dynamic-batch-size --sft-packing-strategy binpack --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" --balance-data
    --seq-length "${SEQ_LENGTH}" --qkv-format thd --data-pad-size-multiplier 1
    --sft-max-in-flight-steps 2 --per-rank-fetch
    --num-data-storage-units 4
    --train-env-vars '{"PYTORCH_ALLOC_CONF": "expandable_segments:True", "NVTE_USE_CUTLASS_GROUPED_GEMM": "1"}'
    --decoder-first-pipeline-num-layers 22 --decoder-last-pipeline-num-layers 18
    --bf16 --tensor-model-parallel-size 2 --pipeline-model-parallel-size 2
    --context-parallel-size "${CP_SIZE}" --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
    --linear-cp-mode headwise
    --sequence-parallel
    --calculate-per-token-loss
    --mtp-num-layers 0 --mtp-loss-scaling-factor 0 --moe-aux-loss-coeff 0
    --moe-router-dtype fp32 --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep --moe-shared-expert-overlap
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --attention-backend flash --no-rope-fusion --cross-entropy-loss-fusion
    --attention-dropout 0 --hidden-dropout 0 --rollout-temperature 1
    --overlap-grad-reduce --overlap-param-gather
    --optimizer adam --use-distributed-optimizer --accumulate-allreduce-grads-in-fp32
    --lr 1e-5 --min-lr 1e-6 --lr-decay-style cosine
    --lr-decay-iters "${LR_DECAY_STEPS}" --lr-warmup-iters "${WARMUP_STEPS}"
    --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 --adam-eps 1e-8 --clip-grad 1
    --attention-softmax-in-fp32 --freeze-vision-model --freeze-vision-projection
    --sft-chunked-logits --sft-logits-chunk-size 1024
    --use-clearml --use-metrics-service
    --tb-project-name "${TB_PROJECT_NAME}" --tb-experiment-name "${TB_EXPERIMENT_NAME}"
)

mkdir -p "${LOG_DIR}"
printf 'Log: %s\n' "${LOG_FILE}"
printf '%q ' "${CMD[@]}"; printf '\n'

: "${RUNTIME_ENV_JSON:?Launch this script through scripts/entrypoint/ray-job.sh}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NVTE_USE_CUTLASS_GROUPED_GEMM="${NVTE_USE_CUTLASS_GROUPED_GEMM:-1}"
export GLOBAL_BATCH_SIZE
RUNTIME_ENV_JSON="$(python3 - <<'PY_ENV'
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime_env.setdefault("env_vars", {}).update({
    "NVTE_USE_CUTLASS_GROUPED_GEMM": os.environ["NVTE_USE_CUTLASS_GROUPED_GEMM"],
    "PYTORCH_ALLOC_CONF": os.environ["PYTORCH_ALLOC_CONF"],
    "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": os.environ.get("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", "eth0"),
    "RELAX_SFT_TQ_SHARDS": "4",
    "TOKENIZERS_PARALLELISM": "true",
    "TQ_PRE_ALLOC_SAMPLE_NUM": os.environ["GLOBAL_BATCH_SIZE"],
    "TQ_ZERO_COPY_SERIALIZATION": "true",
})
print(json.dumps(runtime_env))
PY_ENV
)"
export RUNTIME_ENV_JSON

if [[ -n "${RAY_NO_WAIT:-}" ]]; then
    RAY_SUBMIT_OUTPUT="${SUBMIT_LOG}"
else
    RAY_SUBMIT_OUTPUT="${LOG_FILE}"
fi

[[ "${DRY_RUN:-0}" == 1 ]] && exit 0

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${CMD[@]}" 2>&1 | tee "${RAY_SUBMIT_OUTPUT}"

if [[ -n "${RAY_NO_WAIT:-}" ]]; then
    JOB_ID="$(sed -n "s/.*Job '\([^']*\)' submitted successfully.*/\1/p" "${SUBMIT_LOG}" | tail -n 1)"
    [[ -n "${JOB_ID}" ]] || { echo "Failed to extract Ray job ID from ${SUBMIT_LOG}" >&2; exit 1; }
    nohup ray job logs --address="${RAY_JOB_ADDRESS}" --follow --log-style=record --log-color=false "${JOB_ID}" \
        >"${LOG_FILE}" 2>&1 </dev/null &
    echo "Ray job ${JOB_ID} logs are being written to ${LOG_FILE}"
fi
