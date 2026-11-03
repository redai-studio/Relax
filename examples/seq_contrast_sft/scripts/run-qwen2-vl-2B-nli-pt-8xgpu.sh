#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen2-VL NLI contrastive PT through the standard SFT/Megatron loop.
# Launch via scripts/entrypoint/ray-job.sh; see ../README.md for input contracts.

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the prepared model root}"
: "${DATA_DIR:?Set DATA_DIR to the NLI dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray Jobs HTTP address}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

# Select Bridge's initializer for this example and pass it to Ray workers.
export RELAX_LORA_A_INIT_METHOD="${RELAX_LORA_A_INIT_METHOD:-kaiming}"
RUNTIME_ENV_JSON=$(python3 - <<'PYENV'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime.setdefault("env_vars", {})["RELAX_LORA_A_INIT_METHOD"] = os.environ["RELAX_LORA_A_INIT_METHOD"]
print(json.dumps(runtime))
PYENV
)
export RUNTIME_ENV_JSON

QWEN2_VL_VOCAB_SIZE=$(python3 - "${MODEL_DIR}/Qwen2-VL-2B-Instruct-emb/config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    config = json.load(file)
assert config["model_type"] == "qwen2_vl"
print(config.get("text_config", config)["vocab_size"])
PY
)
source "${RELAX_ROOT}/scripts/models/qwen2-vl-2B.sh"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/seq-contrast}"
# Keep this standard TE/Kaiming recipe separate from older precision profiles.
EXP_NAME="${EXP_NAME:-qwen2-vl-2b-nli-bridge-lora-kaiming-dp8}"
now=$(date "+%Y%m%d-%H%M%S")

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen2-VL-2B-Instruct-emb"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval 200
    --max-actor-ckpt-to-keep 2
)
SFT_ARGS=(
    --loss-type sft
    --task-type seq_contrast
    --contrastive-temperature 0.05
    --custom-dataset-class relax.engine.sft.dataset.retrieval.NLIPairDataset
    --prompt-data "${DATA_DIR}/nli/pairs.jsonl"
    --global-batch-size 512
    --n-samples-per-prompt 1
    --num-rollout 1078
    --num-steps-per-rollout 1
    --seq-length 1024
    --use-dynamic-batch-size
    --max-tokens-per-gpu 131072
    --sft-oversize-strategy keep
    --balance-data
    --per-rank-fetch
)
# Shared Bridge CanonicalLoRA retains seven independent projection adapters.
LORA_ARGS=(
    --lora-rank 64
    --lora-alpha 128
    --lora-dropout 0.05
    --lora-target-modules linear_q linear_k linear_v linear_proj linear_fc1_gate linear_fc1_up linear_fc2
    --lora-scope language
    --lora-merge-mode
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 2e-4
    --lr-decay-style cosine
    --min-lr 0
    --lr-warmup-iters 33
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
