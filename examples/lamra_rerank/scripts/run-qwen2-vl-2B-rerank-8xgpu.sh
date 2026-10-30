#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# LamRA pointwise + listwise rank SFT; 192 sequences/update on eight DP ranks.

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the original model root}"
: "${DATA_DIR:?Set DATA_DIR to the prepared rerank dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray Jobs HTTP address}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_LORA_A_INIT_METHOD="${RELAX_LORA_A_INIT_METHOD:-kaiming}"
export RELAX_HF_PROCESSOR_USE_FAST=0
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime.setdefault("env_vars", {})["RELAX_LORA_A_INIT_METHOD"] = os.environ["RELAX_LORA_A_INIT_METHOD"]
runtime["env_vars"]["RELAX_HF_PROCESSOR_USE_FAST"] = os.environ["RELAX_HF_PROCESSOR_USE_FAST"]
runtime["env_vars"]["TOKENIZERS_PARALLELISM"] = os.environ["TOKENIZERS_PARALLELISM"]
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON
source "${RELAX_ROOT}/scripts/models/qwen2-vl-2B.sh"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/lamra-rerank}"
EXP_NAME="${EXP_NAME:-qwen2-vl-2b-lamra-rank-r128-dp8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
read -r DATASET_STEPS WARMUP_STEPS < <(python3 - "${DATA_DIR}/rerank/train.manifest.json" "${GLOBAL_BATCH_SIZE}" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as stream:
    manifest = json.load(stream)
assert manifest["status"] == "prepared"
steps = math.ceil(manifest["sequences"] / int(sys.argv[2]))
print(steps, math.ceil(steps * 0.03))
PY
)
now=$(date "+%Y%m%d-%H%M%S")

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen2-VL-2B-Instruct"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-1000}"
    --max-actor-ckpt-to-keep 2
)
SFT_ARGS=(
    --loss-type sft
    --task-type causal_lm
    --custom-dataset-class examples.lamra_rerank.dataset.LamraRerankDataset
    --prompt-data "${DATA_DIR}/rerank/train.jsonl"
    --input-key messages
    --multimodal-keys '{"image":"images"}'
    --calculate-per-token-loss
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --n-samples-per-prompt 1
    --num-rollout "${DATASET_STEPS}"
    --num-steps-per-rollout 1
    --seq-length 4096
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768
    --sft-oversize-strategy keep
    --sft-prefetch-num-workers 32
    --sft-prefetch-chunk-size 64
    --sft-prefetch-buffer-size 384
    --image-min-token-num 4
    --image-max-token-num 300
    --balance-data
    --per-rank-fetch
)
LORA_ARGS=(
    --lora-rank 128
    --lora-alpha 256
    --lora-dropout 0.05
    --lora-target-modules linear_q linear_k linear_v linear_proj linear_fc1_gate linear_fc1_up linear_fc2
    --lora-scope language
    --lora-merge-mode
    --freeze-vision-model
    --freeze-vision-projection
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 4e-5
    --lr-decay-style cosine
    --min-lr 0
    --lr-warmup-iters "${WARMUP_STEPS}"
    --weight-decay 0
    --adam-beta1 0.9
    --adam-beta2 0.999
    --clip-grad 1.0
)
PERF_ARGS=(
    --bf16
    --seed 42
    --data-parallel-random-init
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --qkv-format thd
    --use-gloo-process-groups
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --no-rope-fusion
    --attention-dropout 0
    --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)
TRACKING_ARGS=(
    --use-clearml
    --use-metrics-service
    --no-use-tensorboard
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
RUN_LOG_DIR="${RELAX_RUN_LOG_DIR:-${SCRIPT_DIR}/log}"
mkdir -p "${RUN_LOG_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft": [1, 0], "actor": [1, 8]}' \
    --sft-max-in-flight-steps 2 --num-data-storage-units 8 \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${LORA_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" "${PERF_ARGS[@]}" "${TRACKING_ARGS[@]}" "$@" \
    2>&1 | tee "${RUN_LOG_DIR}/${EXP_NAME}-${now}.log"
