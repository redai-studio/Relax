#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full 40-layer DeepSeek-V4.1 Flash THD SFT on OpenMathReasoning-mini, 128 H800 GPUs.
# Keep the four-layer SFT data, loss, optimizer and 8192-token acceptance budget.
# TP2 / PP2 / CP2 / DP16, EP64 / ETP1 / expert-DP1, SP, full recompute and CPU Adam.
# Use alltoall dispatch (DeepEP is unavailable on the current cluster).
# Non-LoRA training retains the model's frozen Engram tables, indexer and routers.
# No rollout engines, routing replay, prediction or debug rollout dumps.
# Requires the same DeepSeek-V4.1 Megatron runtime patch as the full-model RL run.
# The 4096-token per-GPU budget has not yet been validated for full-model SFT.
#
# Run on the head of an existing 16-node x 8-GPU Ray cluster:
#   RELAX_SKIP_CLUSTER_CLEANUP=1 \
#     MODEL_DIR=/path/to/models DATA_DIR=/path/to/data SAVE_DIR=/path/to/checkpoints \
#     bash scripts/entrypoint/ray-job.sh \
#       scripts/training/sft/run-deepseek-v4.1-flash-sft-thd-128xgpu.sh
# Direct bash also uses the existing-cluster entrypoint; it does not create Ray.
# DRY_RUN=1 prints argv/runtime env without starting Ray or importing ML packages.
# Keep SAVE_DIR/EXP_NAME to resume; change EXP_NAME for an independent comparison.
# Miles comparisons must use identical accepted samples, rendered tokens and loss masks.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/DeepSeek-V4.1-bf16}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/deepseek-v41-openmath}"
EXP_NAME="${EXP_NAME:-deepseek-v41-flash-sft-openmath-thd-tp2-pp2-cp2-ep64-gpu128}"
NUM_STEPS="${NUM_STEPS:-100}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
# SFT filters after rendering at CP2 * MAX_TOKENS_PER_GPU; changing this changes the dataset.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
SEED="${SEED:-1234}"
source "${RELAX_ROOT}/scripts/models/deepseek-v4.1-flash.sh"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${HF_CKPT}"
    --megatron-to-hf-mode bridge
    # --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}-${now}"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 1
)

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key problem
    --label-key generated_solution
    --num-rollout "${NUM_STEPS}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --sft-tq-timeout-minutes 60
    --sft-oversize-strategy skip
    --balance-data
    # Same padding quantum as the four-layer THD SFT baseline.
    --data-pad-size-multiplier 1024
)

PERF_ARGS=(
    --qkv-format thd
    --use-dynamic-batch-size
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --context-parallel-size 2
    --expert-model-parallel-size 64
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --allgather-cp
    --cp-partition-mode contiguous
    --calculate-per-token-loss
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --recompute-loss-function
    --log-probs-chunk-size 4096
    --moe-token-dispatcher-type alltoall
    --sft-max-in-flight-steps 2
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --adam-eps 1e-8
    --clip-grad 1.0
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --bf16
    --transformer-impl transformer_engine
    --seed "${SEED}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --train-memory-margin-bytes 1073741824
    --trust-remote-code
    --log-interval 1
    --use-health-check
)

TRAIN_ARGS=(
    --resource '{"actor": [1, 128], "sft": [1, 0]}'
    --num-gpus-per-node 8
    --max-staleness 0
    --num-data-storage-units 16
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "$@"
)

if [ "${DRY_RUN:-0}" != 1 ] && [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
runtime.setdefault("env_vars", {}).update({
    # Physical H800 GPUs; reported device names may be masked.
    "RELAX_DEVICE_PEAK_TFLOPS": os.environ.get(
        "RELAX_DEVICE_PEAK_TFLOPS", runtime.get("env_vars", {}).get("RELAX_DEVICE_PEAK_TFLOPS", "989")
    ),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_ALGO": "Ring",
    # Bound host staging for synchronous CPU-optimizer checkpoints.
    "MEGATRON_SYNC_SAVE_BOUNDED_STAGING": os.environ.get("MEGATRON_SYNC_SAVE_BOUNDED_STAGING", "1"),
    "MEGATRON_SYNC_SAVE_STAGE_BYTES": os.environ.get("MEGATRON_SYNC_SAVE_STAGE_BYTES", "0"),
})
for name in ("CLEARML_CONFIG_FILE", "CUDA_HOME", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NO_VCS_VERSION"):
    if os.environ.get(name):
        runtime["env_vars"][name] = os.environ[name]
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n%s\n' "${RUNTIME_ENV_JSON}"
    exit 0
fi

mkdir -p "${RELAX_ROOT}/log"
ray job submit \
    ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS:-${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RELAX_ROOT}/log/${EXP_NAME}-${now}.log"
