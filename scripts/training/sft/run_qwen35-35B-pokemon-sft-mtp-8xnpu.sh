#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-35B-A3B MTP SFT on pokemon-gpt4o-captions, 8xNPU single-node, ray-submit launch.
#
# Usage:
#   bash scripts/training/sft/run_qwen35-35B-pokemon-sft-mtp-8xnpu.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo 当前时间:

export ASCEND_COREDUMP_SIGNAL=none
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_HOST_SOCKET_PORT_RANGE=63000-63150
export HCCL_NPU_SOCKET_PORT_RANGE=64000-64150

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/sft/pokemon}"
EXP_NAME=qwen3.5-35B-A3B-mtp-sft-pokemon-gpu8
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
TRAIN_FILES=(
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_en.parquet'"
    "'${DATA_DIR}/sft/data/pokemon-gpt4o-captions/pokemon_gpt4o_zh.parquet'"
)
PROMPT_DATA="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
SAVE_DIR="${SAVE_DIR:=${MODEL_DIR}/checkpoint/checkpoints/qwen3.5-35B-A3B-mtp-pokemon-sft-0821}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-35B-A3B
   --ref-load ${MODEL_DIR}/Qwen3.5-35B-A3B
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}/sft/${EXP_NAME}
   # --load ${SAVE_DIR}/sft/${EXP_NAME}
   --save-interval 100
   --num-epoch 10
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

if [[ "${MTP_NUM_LAYERS:-1}" != "1" ]]; then
    echo "ERROR: MTP_NUM_LAYERS must be 1 for Qwen3.5 (checkpoint has mtp_num_hidden_layers=1)." >&2
    exit 1
fi

MTP_ARGS=(
   --mtp-num-layers ${MTP_NUM_LAYERS:-1}
   --enable-mtp-training
   --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.2}
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
   --no-gradient-accumulation-fusion

   # --moe-flex-dispatcher-backend deepep
   # --moe-token-dispatcher-type flex
   --cross-entropy-loss-fusion
   --no-rope-fusion
   --sft-chunked-logits
   --sft-logits-chunk-size ${SFT_LOGITS_CHUNK_SIZE:-2048}

   --colocate
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
   --use-flash-attn
)

mkdir -p log

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${MASTER_ADDR}:8265" \
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
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3.5-35B-A3B-mtp-sft-pokemon-npu8-${now}.log
