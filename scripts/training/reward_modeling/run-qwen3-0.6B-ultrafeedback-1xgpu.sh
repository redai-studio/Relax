#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR}/Qwen3-0.6B-${MODEL_REVISION}}"
PROMPT_DATA="${PROMPT_DATA:?set PROMPT_DATA to the Task 31 train JSONL or Parquet}"
EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA:?set EVAL_PROMPT_DATA to the Task 31 held-out JSONL or Parquet}"
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/../../../checkpoints/task31-reward-modeling}"
EXP_NAME="${EXP_NAME:-qwen3-0.6b-ultrafeedback-rm-gpu1}"
now=$(date "+%Y-%m-%d-%H:%M:%S")

mkdir -p log "${SAVE_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft":[1,0],"actor":[1,1]}' \
    --loss-type sft \
    --sft-objective reward_model \
    --prompt-data "${PROMPT_DATA}" \
    --eval-prompt-data task31 "${EVAL_PROMPT_DATA}" \
    --eval-interval "${EVAL_INTERVAL:-200}" \
    --input-key prompt \
    --preference-pair-id-key prompt_id \
    --preference-max-length 1024 \
    --preference-max-completion-length 512 \
    --preference-chat-template-sha256 56965952fc78cd889bcd1864d70e85271861eef93385410b879c0c4c2d40564d \
    --preference-require-no-generation-marker \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${HF_CHECKPOINT}" \
    --megatron-to-hf-mode bridge \
    --save "${SAVE_DIR}/${EXP_NAME}" \
    --load "${SAVE_DIR}/${EXP_NAME}" \
    --save-interval 50 \
    --num-rollout "${NUM_ROLLOUT:-200}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE:-32}" \
    --use-dynamic-batch-size \
    --use-gloo-process-groups \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --optimizer adam \
    --lr "${LR:-1e-5}" \
    --lr-decay-style cosine \
    --min-lr 0 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --attention-backend flash \
    --no-rope-fusion \
    --colocate \
    "${MODEL_ARGS[@]}" 2>&1 | tee "log/${EXP_NAME}-${now}.log"
