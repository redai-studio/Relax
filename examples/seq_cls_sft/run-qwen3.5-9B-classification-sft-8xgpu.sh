#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B sequence-classification SFT, 8xGPU single-node, Ray Job launch.
#
# Prepare the default full SST-2 data first:
#   python examples/seq_cls_sft/tools/prepare_classification_sft_data.py \
#     --dataset sst2 --subset full --global-batch-size 64 \
#     --output-dir "${DATA_DIR}/sft/seq_cls"
#
# Select another prepared dataset with CLASSIFICATION_DATASET=ag_news or
# CLASSIFICATION_DATASET=go_emotions. Override TRAIN_DATA/EVAL_DATA to use
# custom JSONL files with {"messages": [...], "label": ...} rows.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

CLASSIFICATION_DATASET="${CLASSIFICATION_DATASET:-sst2}"
case "${CLASSIFICATION_DATASET}" in
    sst2)
        NUM_LABELS="${NUM_LABELS:-2}"
        PROBLEM_TYPE="${PROBLEM_TYPE:-single_label_classification}"
        ;;
    ag_news)
        NUM_LABELS="${NUM_LABELS:-4}"
        PROBLEM_TYPE="${PROBLEM_TYPE:-single_label_classification}"
        ;;
    go_emotions)
        NUM_LABELS="${NUM_LABELS:-28}"
        PROBLEM_TYPE="${PROBLEM_TYPE:-multi_label_classification}"
        ;;
    *)
        echo "Unsupported CLASSIFICATION_DATASET=${CLASSIFICATION_DATASET}" >&2
        exit 2
        ;;
esac

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/classification}"
EXP_NAME="${EXP_NAME:-qwen3.5-9b-${CLASSIFICATION_DATASET}-seq-cls-gpu8}"
EXP_DIR="${EXP_DIR:-${RELAX_ROOT}/../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
SEQ_CLS_DATA_DIR="${SEQ_CLS_DATA_DIR:-${DATA_DIR}/sft/seq_cls}"
DATA_SUBSET="${DATA_SUBSET:-full}"
TRAIN_DATA="${TRAIN_DATA:-${SEQ_CLS_DATA_DIR}/${CLASSIFICATION_DATASET}/${DATA_SUBSET}/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-${SEQ_CLS_DATA_DIR}/${CLASSIFICATION_DATASET}/${DATA_SUBSET}/validation.jsonl}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/qwen3.5-9B-classification-sft}"
LOAD_DIR="${LOAD_DIR:-${SAVE_DIR}/${EXP_NAME}}"
RAY_ADDRESS="${RAY_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.5-9B"
   --ref-load "${MODEL_DIR}/Qwen3.5-9B"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/${EXP_NAME}"
   --load "${LOAD_DIR}"
   --save-interval "${SAVE_INTERVAL:-100}"
   --max-actor-ckpt-to-keep 1
   --num-epoch "${NUM_EPOCH:-10}"
)

SFT_ARGS=(
   --loss-type sft
   --task-type seq_cls
   --num-labels "${NUM_LABELS}"
   --problem-type "${PROBLEM_TYPE}"
   --classification-threshold "${CLASSIFICATION_THRESHOLD:-0.5}"
   --prompt-data "${TRAIN_DATA}"
   --input-key messages
   --label-key label
   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-20480}"
   --sft-oversize-strategy truncate_right
   --balance-data
)

EVAL_ARGS=(
   --eval-prompt-data validation "${EVAL_DATA}"
   --eval-interval "${EVAL_INTERVAL:-10}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --calculate-per-token-loss
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --no-rope-fusion
   --colocate
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-5}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR:-1e-6}"
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

TRACKING_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --sft-max-in-flight-steps 4 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${TRACKING_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/${EXP_NAME}-${now}.log"
