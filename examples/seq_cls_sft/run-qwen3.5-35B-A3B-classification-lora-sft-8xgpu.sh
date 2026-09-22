#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B LoRA sequence-classification SFT, 8xGPU single-node.
# The default validation target is the 28-label GoEmotions simplified task.
#
# Prepare data:
#   python examples/seq_cls_sft/tools/prepare_classification_sft_data.py \
#     --dataset go_emotions --subset full --global-batch-size 64 \
#     --output-dir "${DATA_DIR}/sft/seq_cls"

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

CLASSIFICATION_DATASET="${CLASSIFICATION_DATASET:-go_emotions}"
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
EXP_NAME="${EXP_NAME:-qwen3.5-35b-a3b-${CLASSIFICATION_DATASET}-seq-cls-lora-gpu8}"
EXP_DIR="${EXP_DIR:-${RELAX_ROOT}/../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
SEQ_CLS_DATA_DIR="${SEQ_CLS_DATA_DIR:-${DATA_DIR}/sft/seq_cls}"
DATA_SUBSET="${DATA_SUBSET:-full}"
TRAIN_DATA="${TRAIN_DATA:-${SEQ_CLS_DATA_DIR}/${CLASSIFICATION_DATASET}/${DATA_SUBSET}/train.jsonl}"
EVAL_DATA="${EVAL_DATA:-${SEQ_CLS_DATA_DIR}/${CLASSIFICATION_DATASET}/${DATA_SUBSET}/validation.jsonl}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/checkpoints/qwen3.5-35B-A3B-classification-lora-sft}"
LOAD_DIR="${LOAD_DIR:-${SAVE_DIR}/${EXP_NAME}}"
RAY_ADDRESS="${RAY_ADDRESS:-http://${HOST_IP:-127.0.0.1}:8265}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.5-35B-A3B"
   --ref-load "${MODEL_DIR}/Qwen3.5-35B-A3B"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/${EXP_NAME}"
   --load "${LOAD_DIR}"
   --save-interval "${SAVE_INTERVAL:-100}"
   --num-epoch "${NUM_EPOCH:-10}"
)

LORA_ARGS=(
   --lora-rank "${LORA_RANK:-32}"
   --lora-alpha "${LORA_ALPHA:-64}"
   --lora-target-modules '*decoder.layers.*.linear_qkv' '*decoder.layers.*.linear_proj'
   --lora-dropout 0.0
   --lora-merge-mode
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
   --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-16384}"
   --sft-oversize-strategy truncate_right
   --balance-data
   --per-rank-fetch
   --sft-prefetch-num-workers "${SFT_PREFETCH_NUM_WORKERS:-16}"
   --sft-prefetch-buffer-size "${SFT_PREFETCH_BUFFER_SIZE:-512}"
)

EVAL_ARGS=(
   --eval-prompt-data validation "${EVAL_DATA}"
   --eval-interval "${EVAL_INTERVAL:-20}"
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --calculate-per-token-loss
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --no-rope-fusion
   --colocate
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-1e-4}"
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
   --use-tensorboard
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
   --sft-max-in-flight-steps 1 \
   --num-data-storage-units 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${TRACKING_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/${EXP_NAME}-${now}.log"
