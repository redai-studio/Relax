#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# DeepSeek-V4.1 four-layer SFT on OpenMathReasoning-mini, 8 H800 GPUs.
# Based on run-deepseek-v4-flash-sft-0731-128xgpu.sh, not the RL/log-prob recipe.
# Uses the prepared BF16 four-layer checkpoint, with its frozen Engram table.
# TP2 / PP2 / CP2 / DP1, EP4 / ETP1 / expert-DP1, SP and DeepEP.
# No inference engines or prediction task.
# BSHD uses fixed micro-batches; overlong samples are skipped after rendering.
# The 4096-token starting budget is not an 80-GB GPU memory guarantee.
#
# Usage on a dedicated training host:
#   MODEL_DIR=/path/to/models DATA_DIR=/path/to/data SAVE_DIR=/path/to/checkpoints \
#     bash scripts/training/sft/run-deepseek-v4.1-4layer-sft-8xgpu.sh
# With an existing Ray cluster, use scripts/entrypoint/ray-job.sh as usual.
# DRY_RUN=1 prints argv/runtime env without starting Ray or importing ML packages.
# Reuse SAVE_DIR/EXP_NAME to resume; change EXP_NAME for an independent baseline.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/DeepSeek-V4.1-4layer}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/deepseek-v41-openmath}"
EXP_NAME="${EXP_NAME:-deepseek-v41-4layer-sft-openmath-tp2-pp2-cp2-ep4-deepep-gpu8}"
NUM_STEPS="${NUM_STEPS:-100}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
# CP2 keeps the full-sequence filtering budget at 2 * 2048 = 4096 tokens.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
SEED="${SEED:-1234}"
LOG_DIR="${LOG_DIR:-${RELAX_ROOT}/deepseek_v41_log}"
source "${RELAX_ROOT}/scripts/models/deepseek-v4.1-4layer.sh"


CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${HF_CKPT}"
    --megatron-to-hf-mode bridge
    # --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}-${now}"
    --save-interval 100
    --max-actor-ckpt-to-keep 1
)

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key problem
    --label-key generated_solution
    --num-rollout "${NUM_STEPS}"
    --rollout-batch-size "${GLOBAL_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --n-samples-per-prompt 1
    --num-steps-per-rollout 1
    --num-iters-per-train-update 1
    --rollout-temperature 1.0
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --sft-tq-timeout-minutes 60
    --sft-oversize-strategy skip
    --balance-data
    # Padding alignment: 2 * CP2 * TP2 * 512 = 4096.
    --data-pad-size-multiplier 1024
)

PERF_ARGS=(
    --qkv-format bshd
    # --seq-length "$((MAX_TOKENS_PER_GPU * 2))"
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --context-parallel-size 2
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --allgather-cp
    --micro-batch-size 1
    --calculate-per-token-loss
    # Leave room for lazily initialized optimizer/NCCL communicators on 80-GB cards.
    --empty-unused-memory-level 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --recompute-loss-function
    --log-probs-chunk-size 4096
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-router-dtype fp32
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
    --no-rope-fusion
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
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
    --log-interval 1
    --use-health-check
)

TRAIN_ARGS=(
    --resource '{"actor": [1, 8], "sft": [1, 0]}'
    --num-gpus-per-node 8
    --max-staleness 0
    --num-data-storage-units 1
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
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
runtime.setdefault("env_vars", {}).update({
    "NCCL_NVLS_ENABLE": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
for name in ("CLEARML_CONFIG_FILE", "CUDA_HOME", "NO_VCS_VERSION"):
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

RAY_JOB_ADDRESS="${RAY_DASHBOARD_ADDRESS:-${RAY_ADDRESS:-http://127.0.0.1:8265}}"
if [[ "${RAY_JOB_ADDRESS}" != http://* && "${RAY_JOB_ADDRESS}" != https://* ]]; then
    echo "Set RAY_DASHBOARD_ADDRESS to the Ray Jobs HTTP address, not the GCS address." >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${EXP_NAME}-${now}.log"
