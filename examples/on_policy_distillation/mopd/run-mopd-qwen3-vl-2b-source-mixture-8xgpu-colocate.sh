#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Multi-Teacher OPD (MOPD) with source-mixture quotas, single-node 8 GPU colocate:
#   student : Qwen3-VL-2B-Instruct, actor trains on ALL 8 GPUs, rollout uses 4
#   teacher : 2 teachers x 2 replicas each, share the actor pool (4 GPUs total)
#     GSM8K      -> Qwen3-4B-Instruct-2507 (text): 2 replicas x TP=1
#     Geometry3K -> Qwen3-VL-4B-Instruct   (VL):   2 replicas x TP=1
#
# Each 32-sample optimizer batch contains 24 GSM8K + 8 Geometry3K prompts (N=1).
# Routing x replicas: each sample is routed by data_source to its teacher, then
# round-robined across that teacher's replicas.
#
# Teacher GPU split: teacher total (4) / num_teachers (2) = 2 GPU/teacher;
# per teacher, 2 GPU / --teacher-num-gpus-per-engine (1) = 2 replicas (TP=1).
# Raise TEACHER_NUM_GPUS_PER_ENGINE (e.g. 2) + TEACHER_GPUS for larger teachers.
#
# Constraint (enforced): rollout_gpus + total_teacher_gpus == actor_gpus (4 + 4 == 8 ✓).
#
# Usage:
#   bash run-mopd-qwen3-vl-2b-source-mixture-8xgpu-colocate.sh

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
export SGLANG_MM_AVOID_RETOKENIZE=true
# Preserve response IDs in both local and remote-node native image processors.
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=8192
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}SGLANG_MM_AVOID_RETOKENIZE,RELAX_OPD_PER_POS_TOKEN_IDS,SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK,SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"

now=$(date "+%Y%m%d-%H%M%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export RELAX_ENTRYPOINT_MODE="${RELAX_ENTRYPOINT_MODE:-external}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${RELAX_ROOT}/scripts/models/qwen3-vl-2B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/recipes/mopd}"
EXP_NAME="${EXP_NAME:-mopd-qwen3-vl-2b-source-mixture-8xgpu-colocate}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
export DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/save}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3-VL-2B-Instruct}"
TEXT_TEACHER_MODEL_NAME="${TEXT_TEACHER_MODEL_NAME:-Qwen3-4B-Instruct-2507}"
VL_TEACHER_MODEL_NAME="${VL_TEACHER_MODEL_NAME:-Qwen3-VL-4B-Instruct}"

ACTOR_GPUS="${ACTOR_GPUS:-8}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
TEACHER_GPUS="${TEACHER_GPUS:-4}"
TEACHER_NUM_GPUS_PER_ENGINE="${TEACHER_NUM_GPUS_PER_ENGINE:-1}"

mkdir -p "${SCRIPT_DIR}/log"
SOURCE_MIXTURE_CONFIG="${SOURCE_MIXTURE_CONFIG:-${SCRIPT_DIR}/source-mixture.yaml}"

TEACHER_ROUTES=$(python3 - "${MODEL_DIR}/${TEXT_TEACHER_MODEL_NAME}" "${MODEL_DIR}/${VL_TEACHER_MODEL_NAME}" <<'PY'
import json
import sys

print(json.dumps({"openai/gsm8k": sys.argv[1], "hiyouga/geometry3k": sys.argv[2]}))
PY
)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/${STUDENT_MODEL_NAME}"
   --megatron-to-hf-mode bridge
   --load "${SAVE_DIR}/${EXP_NAME}"
   --save "${SAVE_DIR}/${EXP_NAME}"
   --save-interval "${SAVE_INTERVAL:-5}"
   --max-actor-ckpt-to-keep 2
)

ROLLOUT_ARGS=(
   --rollout-source-mixture "${SOURCE_MIXTURE_CONFIG}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --use-streaming-dataset

   --multimodal-keys '{"image":"images"}'

   --rm-type dummy
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 32
   --n-samples-per-prompt 1
   --rollout-max-prompt-len 1024
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 32
)

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-only-reward
   --opd-kl-coef 0.3
   --opd-token-selection student_sampled
   --opd-teacher-key data_source
   --opd-teacher-routes "${TEACHER_ROUTES}"
   --teacher-num-gpus-per-engine "${TEACHER_NUM_GPUS_PER_ENGINE}"
   --teacher-sglang-mem-fraction-static "${TEACHER_MEM_FRACTION:-0.5}"
   --teacher-sglang-chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-16384}"
   --teacher-sglang-max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-256}"
   --teacher-sglang-disable-cuda-graph
   --opd-log-prob-top-k 64
   --opd-log-prob-min-clamp -10.0
   --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-1200}"
   --opd-teacher-image-key images
   --use-rollout-logprobs
   --rollout-stop-token-ids "${ROLLOUT_STOP_TOKEN_ID:-128247}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --accumulate-allreduce-grads-in-fp32
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

EVAL_ARGS=(
   --skip-eval-before-train
   --eval-global-batch-size 32
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

# Student colocate: TP=1, DP=8 (actor 8 GPU). GBS=32, 32/8=4 per rank ✓
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 3072
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
)

RESOURCE_JSON="{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}"

python3 -m relax.entrypoints.train \
    --resource "${RESOURCE_JSON}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --offload \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
