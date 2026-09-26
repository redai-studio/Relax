#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B LoRA + MTP SFT on pokemon-gpt4o-captions, 8xGPU single-node, ray-submit launch.
#
# LoRA is scoped to the main transformer's attention (linear_qkv / linear_proj) via
# wildcard patterns; MTP layers (mtp.layers.*.mtp_model_layer.*) are intentionally
# NOT wrapped with adapters — they stay frozen alongside the rest of the base model.
#
# Usage:
#   bash scripts/training/sft/run-qwen3.5-35B-A3B-pokemon-lora-mtp-8xgpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/pokemon}"
EXP_NAME=qwen3.5-35B-A3B-lora-mtp-sft-pokemon-gpu8
EXP_DIR="${MODEL_DIR:=${SCRIPT_DIR}/../../../../exps}"
DATA_DIR="${DATA_DIR:=${SCRIPT_DIR}/data}"
TRAIN_FILES=(
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet'"
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
SAVE_DIR="${SAVE_DIR:=${SCRIPT_DIR}/../../../checkpoints/qwen3.5-35B-A3B-pokemon-lora-mtp-sft}"

CKPT_ARGS=(
   --hf-checkpoint ${EXP_DIR}/Qwen3.5-35B-A3B
   --ref-load ${EXP_DIR}/Qwen3.5-35B-A3B
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}/sft/${EXP_NAME}
   --load ${SAVE_DIR}/sft/${EXP_NAME}
   --save-interval 100
   --num-epoch 10
)

LORA_ARGS=(
   --lora-rank 32
   --lora-alpha 64
   # Scoped to main transformer only. Bridge's LoRA matcher walks the full model
   # (including `mtp.layers.*.mtp_model_layer.*`) — using bare short names like
   # `linear_qkv` would also match MTP attention. The `*decoder.layers.*` prefix
   # anchors matching to the main decoder path, so MTP layers stay frozen.
   --lora-target-modules '*decoder.layers.*.linear_qkv' '*decoder.layers.*.linear_proj'
   --lora-dropout 0.0
   --lora-merge-mode
)

MTP_ARGS=(
   --mtp-num-layers 1
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
   # --ci-test
)

SFT_ARGS=(
   --loss-type sft
   --prompt-data "${PROMPT_DATA}"
   --input-key conversations
   --multimodal-keys '{"image":"images"}'
   --conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
   --global-batch-size 64
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --balance-data
   --per-rank-fetch
   --sft-prefetch-num-workers 16
   --sft-prefetch-buffer-size 512
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
   --tensor-model-parallel-size 4
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

   --no-rope-fusion

   --colocate
   --cross-entropy-loss-fusion
)

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
   --use-tensorboard
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
   --resource '{"sft": [1, 0], "actor": [1, 8]}' \
   --sft-max-in-flight-steps 4 \
   --num-data-storage-units 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${LORA_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${MTP_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${PREDICT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/${EXP_NAME}-${now}.log
