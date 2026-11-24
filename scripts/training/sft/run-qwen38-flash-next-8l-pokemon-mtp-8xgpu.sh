#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.8-Flash-Next-8L MTP SFT on pokemon-gpt4o-captions, 8xGPU single-node, ray-submit launch.
#
# Usage:
#   bash scripts/training/sft/run-qwen38-flash-next-8l-pokemon-mtp-8xgpu.sh
# H20 and H800 runs must use the same reduced checkpoint and input files.
# For a shorter smoke run, set NUM_STEPS and set LR_DECAY_ITERS to the
# unrestricted iteration count for the same dataset and global batch size.
# Example for the default Pokemon recipe (choose a fresh SAVE_DIR):
#   SAVE_DIR=/path/to/fresh/smoke-checkpoints NUM_STEPS=5 LR_DECAY_ITERS=58 \
#     bash scripts/training/sft/run-qwen38-flash-next-8l-pokemon-mtp-8xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
# The checkpoint must contain a matching eight-layer HF config and weights.
source "${MODEL_CONFIG_DIR}/qwen38-flash-next.sh"
MODEL_ARGS+=(--num-layers 8)

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/pokemon}"
EXP_NAME=qwen38-flash-next-8l-mtp-sft-pokemon-gpu8
MODEL_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
DATA_DIR="${DATA_DIR:=${SCRIPT_DIR}/data}"
TRAIN_FILES=(
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet'"
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
SAVE_DIR="${SAVE_DIR:=${SCRIPT_DIR}/../../../checkpoints/qwen38-flash-next-8l-mtp-pokemon-sft}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.8-Flash-Next-8L"
   --ref-load "${MODEL_DIR}/Qwen3.8-Flash-Next-8L"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}/sft/${EXP_NAME}"
   --load "${SAVE_DIR}/sft/${EXP_NAME}"
   --save-interval 100
   --max-actor-ckpt-to-keep 2
   --num-epoch 10
)
if [[ -n "${NUM_STEPS:-}" ]]; then
    CKPT_ARGS+=(--num-rollout "${NUM_STEPS}")
fi

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key conversations
   --multimodal-keys '{"image":"images"}'
   --conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
   --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --per-rank-fetch
   --sft-prefetch-num-workers 16
   --sft-prefetch-buffer-size 512
   --freeze-vision-model
   --freeze-vision-projection
   --freeze-params-name-list 'self_attention\.indexer\.' 'per_layer_embedding\.'
)

MTP_ARGS=(
   --mtp-num-layers 1
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
   --mtp-detach-paths none
   # --ci-test
)

EVAL_ARGS=(
    --eval-size 0.1
    --eval-interval 20
)

PREDICT_ARGS=(
    # --sft-predict-interval 10
    # --eval-temperature 0.0
    # --eval-max-response-len 512
    # --rollout-num-gpus-per-engine 2
    # --sglang-mem-fraction-static 0.6
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
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
   --moe-grouped-gemm
   # Hopper kernels; streamed PLE and the 48-layer retention policy do not
   # apply to this single-node, CP=1 recipe.
   --train-env-vars '{"NVTE_USE_CUTLASS_GROUPED_GEMM":"0","MCORE_QSA_SPARSE_BACKEND":"flex","RELAX_QWEN38_QSA_FUSED_ATTENTION":"triton_v2","RELAX_QWEN38_QSA_COMMUNITY_BACKEND":"","RELAX_QWEN38_PLE_STREAMED":"0","QWEN38_PLE_CPU_OFFLOAD":"1","RELAX_QWEN38_RETAIN_QSA_ATTENTION":"0","RELAX_QWEN38_QSA_FUSED_SELECTOR":"1","RELAX_QWEN38_QSA_SPARSE_BLOCKMASK":"1","RELAX_QWEN38_QSA_RECOMPUTE_REPLAY":"1","RELAX_QWEN38_QSA_LOCAL_QUERY":"1","RELAX_QWEN38_FLA_CONV_CACHE_REUSE":"1","RELAX_QWEN38_QSA_ROUTED_QKV":"1"}'

   --no-rope-fusion

   --cross-entropy-loss-fusion
   --sft-chunked-logits
   --sft-logits-chunk-size ${SFT_LOGITS_CHUNK_SIZE:-2048}
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
   --moe-aux-loss-coeff 0
)
if [[ -n "${LR_DECAY_ITERS:-}" ]]; then
    OPTIMIZER_ARGS+=(--lr-decay-iters "${LR_DECAY_ITERS}")
fi

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --use-tensorboard
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --seed "${SEED:-1234}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-health-check
)

source "${SCRIPT_DIR}/../../entrypoint/ray-status.sh"

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --sft-max-in-flight-steps 4 \
   --num-data-storage-units 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PREDICT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen38-flash-next-8l-mtp-sft-pokemon-gpu8-${now}.log
