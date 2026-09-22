#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Multi-Teacher OPD (MOPD), 2-node 16 GPU colocate:
#   student : Qwen3.5-35B-A3B,  actor(16) + rollout(8) colocate, TP=4 PP=2 EP=8
#   teacher : one per data_source, shares the actor GPU pool (colocate)
#     dapo-math-17k       -> Qwen3.6-27B  (text, 4 GPU = 2 replicas TP=2)
#     multimodal-open-r1  -> Qwen3.5-27B  (VL,   4 GPU = 2 replicas TP=2)
#
# Colocate GPU layout (2 nodes × 8 GPU, one shared placement group):
#   training : student actor uses ALL 16 GPUs (TP=4 PP=2 EP=8, DP=2)
#   rollout  : GPU 0-7 student rollout (2 engines TP=4) | GPU 8-15 teachers
#   teachers : GPU 8-9 + 10-11 text Qwen3.6-27B | GPU 12-13 + 14-15 VL Qwen3.5-27B
#
# Why replicas instead of one wide engine per role: each engine has a single
# scheduler with a bounded per-iteration prefill budget (chunked-prefill-size),
# so a step's requests queue up behind it; splitting the same GPUs into more
# engines gives independent schedulers and divides per-engine concurrency.
# Requests round-robin across a teacher's replicas (_pick_teacher_url).
# chunked-prefill-size is raised to 16384 (sglang's max_prefill_tokens ceiling)
# for student and teachers alike: long multi-image prompts overflow the default
# budget, which pins prefill at one sequence per iteration and needs many more
# passes per request.
# Constraint (enforced): rollout_gpus(8) + teacher_gpus(8) == actor_gpus(16).
# Teachers live inside the actor placement group and offload/onload in lock-step
# with training. Training the student on all 16 GPUs (TP=4 PP=2) is what fixes
# the 8-GPU grad-norm OOM — no dedicated teacher node needed.
#
# Per-sample rm_type is embedded in extra_info by prepare_data_35b.py;
# no global --rm-type is needed (fallback: "dapo").
#
# Usage:
#   bash run-mopd-qwen35-35ba3b-16xgpu.sh

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
export RELAX_OPD_PREEXPANDED_PATCH=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=8192
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}RELAX_OPD_PREEXPANDED_PATCH,SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK,SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/recipes/mopd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mopd-qwen35-35ba3b-16xgpu-${now}}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-35B-A3B}"
TEXT_TEACHER_MODEL_NAME="${TEXT_TEACHER_MODEL_NAME:-Qwen3.6-27B}"
VL_TEACHER_MODEL_NAME="${VL_TEACHER_MODEL_NAME:-Qwen3.5-27B}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/MOPD-35B/train.parquet}"
EVAL_SET="${EVAL_SET:-${PROMPT_SET%/*}/test_small.parquet}"

ACTOR_GPUS="${ACTOR_GPUS:-16}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
TEACHER_GPUS="${TEACHER_GPUS:-8}"
TEACHER_NUM_GPUS_PER_ENGINE="${TEACHER_NUM_GPUS_PER_ENGINE:-2}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/${STUDENT_MODEL_NAME}/"
   --megatron-to-hf-mode bridge
   # --save "${EXP_DIR}/save/mopd-${STUDENT_MODEL_NAME}/"
   # --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --metadata-key extra_info
   --apply-chat-template
   --rollout-shuffle

   --multimodal-keys '{"image":"images"}'

   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 128

   --log-passrate
   --use-fault-tolerance
   --use-streaming-dataset
)

TEACHER_ROUTES="{\"dapo-math-17k\":\"${MODEL_DIR}/${TEXT_TEACHER_MODEL_NAME}/\",\"multimodal-open-r1\":\"${MODEL_DIR}/${VL_TEACHER_MODEL_NAME}/\"}"

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-only-reward
   --opd-kl-coef 0.2
   --opd-per-token-clip 2.0
   --opd-teacher-key data_source
   --opd-teacher-routes "${TEACHER_ROUTES}"
   --teacher-num-gpus-per-engine "${TEACHER_NUM_GPUS_PER_ENGINE}"
   --teacher-sglang-mem-fraction-static "${TEACHER_MEM_FRACTION:-0.5}"
   --teacher-sglang-chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-16384}"
   --teacher-sglang-max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-128}"
   --teacher-sglang-disable-cuda-graph
   --opd-log-prob-min-clamp -10.0
   --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-1200}"
   --opd-teacher-image-key images
   --use-rollout-logprobs
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
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
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --accumulate-allreduce-grads-in-fp32
   --moe-router-load-balancing-type "none"
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXPERIMENT_NAME}"
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data mopd-35b "${EVAL_SET}"
   --eval-global-batch-size 128
   --n-samples-per-eval-prompt 1
   --eval-temperature 0.0
   --eval-max-response-len 8192
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --calculate-per-token-loss
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --log-probs-max-tokens-per-gpu 10240
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.6
   --sglang-chunked-prefill-size 16384
   --sglang-max-running-requests 128
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
)

PARTIAL_ROLLOUT_ARGS=(
   --partial-rollout
   --over-sampling-batch-size 24
   --mask-offpolicy-in-partial-rollout
   --partial-rollout-max-aborted-count 3
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
    "${PARTIAL_ROLLOUT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "mopd-qwen35-35ba3b-${now}.log"
