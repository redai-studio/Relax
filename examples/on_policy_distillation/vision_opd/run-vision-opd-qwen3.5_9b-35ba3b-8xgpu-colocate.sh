#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Vision OPD, single-node 8 GPU colocate:
#   actor   : 8 GPUs, trains on the full shared placement group
#   rollout : 4 GPUs, uses the first 4 bundles during rollout
#   teacher : 4 GPUs, uses bundles after rollout and is managed by Relax

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
export RELAX_OPD_PREEXPANDED_PATCH=1
# export RELAX_OPD_PER_POS_TOKEN_IDS=1 

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/recipes/vision-opd}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-9B}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen3.5-35B-A3B}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/Vision-OPD-6K/train.relax.jsonl}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/save/vision-opd-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

ACTOR_GPUS="${ACTOR_GPUS:-8}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
TEACHER_GPUS="${TEACHER_GPUS:-4}"

mkdir -p "${SAVE_DIR}" log

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/${STUDENT_MODEL_NAME}/"
   --ref-load "${MODEL_DIR}/${STUDENT_MODEL_NAME}/"
   --megatron-to-hf-mode bridge
   --save "${SAVE_DIR}"
   --save-interval "${SAVE_INTERVAL:-1000}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key messages
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle

   --multimodal-keys '{"image":"images"}'
   --image-min-token-num "${IMAGE_MIN_TOKEN_NUM:-64}"
   --image-max-token-num "${IMAGE_MAX_TOKEN_NUM:-16384}"

   --rm-type random
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-8192}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"

   --use-fault-tolerance
   --use-streaming-dataset
   --mm-processor-pool-size "${MM_PROCESSOR_POOL_SIZE:-8}"
   --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL:-60}"
   --rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT:-120}"
   --rollout-health-check-first-wait "${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-120}"
   --rollout-health-check-max-consecutive-failures "${ROLLOUT_HEALTH_CHECK_MAX_CONSECUTIVE_FAILURES:-5}"
)

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --teacher-hf-checkpoint "${MODEL_DIR}/${TEACHER_MODEL_NAME}/"
   --teacher-sglang-mem-fraction-static "${TEACHER_MEM_FRACTION:-0.8}"
   --teacher-sglang-chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
   --teacher-sglang-max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-32}"
   --teacher-sglang-disable-cuda-graph

   --opd-kl-coef "${OPD_KL_COEF:-1.0}"
   --opd-loss-coef "${OPD_LOSS_COEF:-0.0}"
   --opd-kl-type "${OPD_KL_TYPE:-reverse_kl}"
   
   --opd-token-selection "${OPD_TOKEN_SELECTION:-student_sampled}" # student_sampled
   
   --opd-teacher-image-key bbox_images
   --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-6000}"
   --opd-disable-rl-reward

   --use-rollout-logprobs
   --opd-is-clip 2.0
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.3
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR:-2e-6}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --lr-warmup-iters 10
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${ACTOR_MAX_TOKENS_PER_GPU:-4096}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
   --sglang-mem-fraction-static "${STUDENT_MEM_FRACTION:-0.8}"
   --sglang-load-format dummy
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "vision-opd-8xgpu-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-colocate-${now}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

RAY_DASHBOARD="${RAY_DASHBOARD:-http://${HOST_IP:-127.0.0.1}:8265}"
RESOURCE_JSON="{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}"

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "${RESOURCE_JSON}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPD_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   2>&1 | tee "log/vision-opd-8xgpu-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-colocate-${now}.log"
