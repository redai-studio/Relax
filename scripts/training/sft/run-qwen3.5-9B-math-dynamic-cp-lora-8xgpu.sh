#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B LoRA SFT on OpenMathReasoning-mini, 8xGPU single-node, ray-submit launch.
#
# CP mode:
#   - Default (no CP env var): --dynamic-context-parallel enabled;
#     cp_max = next_pow2(ceil(rollout_max_context_len / max_tokens_per_gpu))
#     = next_pow2(ceil(32768 / 8192)) = 4. Parallelism: TP=2, PP=1, CP=1..4, DP=1.
#   - CP=<int> passed: static context parallel with the given size, dynamic CP off.
#     Must satisfy world_size % (TP*PP*CP) == 0.
#
# Usage:
#   bash scripts/training/sft/run-qwen3.5-9B-math-dynamic-cp-lora-8xgpu.sh          # dynamic CP
#   CP=4 bash scripts/training/sft/run-qwen3.5-9B-math-dynamic-cp-lora-8xgpu.sh     # static CP=4
#   CP=1 bash scripts/training/sft/run-qwen3.5-9B-math-dynamic-cp-lora-8xgpu.sh     # no CP

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

# CP env var: unset → dynamic CP; integer → static CP with that size.
CP="${CP:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/math}"
if [ -n "${CP}" ]; then
    EXP_NAME=qwen3.5-9b-lora-sft-math-cp${CP}-gpu8
else
    EXP_NAME=qwen3.5-9b-lora-sft-math-dyncp-gpu8
fi
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
DATA_DIR="${DATA_DIR:=${SCRIPT_DIR}/data}"
PROMPT_DATA="${PROMPT_DATA:=${DATA_DIR}/sft/data/OpenMathReasoning-mini/data/cot-00000-of-00001.parquet}"
SAVE_DIR="${SAVE_DIR:=${SCRIPT_DIR}/../../../checkpoints/qwen3.5-9B-math-lora-sft-dyncp}"

CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3.5-9B
   --ref-load ${EXP_DIR}/Qwen3.5-9B

   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}/sft/${EXP_NAME}
   --load ${SAVE_DIR}/sft/${EXP_NAME}
   --save-interval 1000
   --num-epoch 10
)

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 64
   --lora-target-modules linear_qkv linear_proj
   --lora-dropout 0.0
   --lora-merge-mode
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key problem
   --label-key generated_solution
   --global-batch-size 64
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --rollout-max-context-len 32768
   --balance-data
)

EVAL_ARGS=(
    # NOTE: not supported yet
    --eval-size 0.01
    --eval-interval 20
)

PREDICT_ARGS=(
    # --sft-predict-interval 10
    # --eval-temperature 0.0
    # --eval-max-response-len 10240
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --calculate-per-token-loss
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --no-rope-fusion

   --colocate
   --cross-entropy-loss-fusion
   --sft-chunked-logits
   --sft-logits-chunk-size ${SFT_LOGITS_CHUNK_SIZE:-1024}
)
if [ -n "${CP}" ]; then
    PERF_ARGS+=(--context-parallel-size "${CP}")
else
    PERF_ARGS+=(--dynamic-context-parallel)
fi

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-4
   --lr-decay-style cosine
   --min-lr 1e-6
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}-${now}
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

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"sft": [1, 0], "actor": [1, 8], "rollout": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PREDICT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/${EXP_NAME}-${now}.log
