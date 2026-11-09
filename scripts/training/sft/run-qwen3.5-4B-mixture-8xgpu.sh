#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Short multimodal/tool SFT validation; checkpoint saving is opt-in.
# MODEL_DIR=/models DATA_DIR=/data SAVE_DIR=/checkpoints RAY_ADDRESS=host:6379 \
#   bash scripts/entrypoint/ray-job.sh scripts/training/sft/run-qwen3.5-4B-mixture-8xgpu.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${SCRIPT_DIR}/../../entrypoint/ray-job.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen35-4B.sh"
: "${MODEL_DIR:?Set MODEL_DIR to the directory containing Qwen3.5-4B}"
: "${DATA_DIR:?Set DATA_DIR to the directory containing mixture.yaml}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/dataset-loss}"
EXP_NAME="${EXP_NAME:-qwen3.5-4b-mixture-sft-gpu8}"
now=$(date +%Y%m%d-%H%M%S)
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

# The official 4B checkpoint ties the embedding/output weights. Its gate uses
# attention-output-gate rather than the obsolete use-gated-attention option.
SFT_MODEL_ARGS=()
for arg in "${MODEL_ARGS[@]}"; do
    case "${arg}" in
        --untie-embeddings-and-output-weights|--use-gated-attention) ;;
        *) SFT_MODEL_ARGS+=("${arg}") ;;
    esac
done
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3.5-4B"
    --ref-load "${MODEL_DIR}/Qwen3.5-4B"
    --megatron-to-hf-mode bridge
)
if [[ "${SAVE_CHECKPOINT:-0}" == 1 ]]; then
    : "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
    CKPT_ARGS+=(
        --load "${SAVE_DIR}/${EXP_NAME}"
        --save "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-50}"
        --max-actor-ckpt-to-keep 2
    )
fi
WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
# Pass paths, never credentials, through the runtime environment.
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
for key in ("CLEARML_CONFIG_FILE", "RELAX_SFT_FUSED_LINEAR_CE"):
    if key in os.environ:
        env[key] = os.environ[key]
print(json.dumps(runtime))
PY
)
SFT_ARGS=(
    --loss-type sft
    --sft-training-mode sft
    --prompt-data "${DATA_DIR}/mixture.yaml"
    --input-key messages
    --multimodal-keys '{"image":"images"}'
    --tool-key tools
    --metadata-key ""
    --sft-oversize-strategy skip
    --num-rollout "${TRAIN_STEPS:-8}"
    --seed 42
    --rollout-batch-size "${GLOBAL_BATCH_SIZE:-32}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
    --seq-length "${SEQ_LENGTH:-16384}"
    --balance-data
    --qkv-format thd
    --sft-max-in-flight-steps 2
    --per-rank-fetch
    --sft-prefetch-num-workers 4
    --sft-prefetch-buffer-size 128
    --mm-processor-pool-size 4
    --image-max-token-num 1024
    --image-min-token-num 4
)
PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --num-data-storage-units 1
    --max-staleness 0
    --calculate-per-token-loss
    --bf16
    --use-distributed-optimizer
    --accumulate-allreduce-grads-in-fp32
    --mtp-num-layers 0
    --mtp-loss-scaling-factor 0
    --freeze-vision-model
    --freeze-vision-projection
    --sft-chunked-logits
    --sft-logits-chunk-size 512
    --recompute-loss-function
    --no-rope-fusion
    --attention-backend flash
    --attention-softmax-in-fp32
    --attention-dropout 0
    --hidden-dropout 0
    --rollout-temperature 1
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style constant
    --weight-decay 0.1
    --clip-grad 1
)
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    --submission-id "${EXP_NAME}-${now}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft": [1, 0], "actor": [1, 8]}' \
    "${SFT_MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${WANDB_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
