#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Multi-Teacher OPD (MOPD), single-node 8 GPU colocate:
#   student : Qwen3.5-9B, actor(8) + rollout(4) colocate
#   teacher : one per data_source, shares the actor GPU pool (4 GPUs total)
#     dapo-math-17k       -> Qwen3.5-9B-dapo-teacher-hf (text, TP=2)
#     multimodal-open-r1  -> Qwen3.5-9B-vl-teacher-hf   (VL,   TP=2)
#   data    : dapo-math-17k + multimodal-open-r1, merged by prepare_data.py
#
# Where the teachers come from: both are Qwen3.5-9B self-distillation checkpoints,
# each trained on its own domain, then converted mcore->HF (SGLang loads HF):
#   Qwen3.5-9B-dapo-teacher-hf : scripts/training/text/run-qwen35-9B-8xgpu.sh (dapo-math)
#   Qwen3.5-9B-vl-teacher-hf   : scripts/training/multimodal/run-qwen35-9B-8xgpu-openr1mm-async.sh (openr1mm)
#
# Pure distillation: --opd-only-reward zeroes the task reward; only the OPD KL
# term trains the student toward the two teachers.
# Constraint (enforced): rollout_gpus + teacher_gpus == actor_gpus (4 + 4 == 8).
#
# Usage:
#   bash run-mopd-qwen35-9b-8xgpu-colocate.sh

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
export RELAX_OPD_PREEXPANDED_PATCH=1
# The OPD teacher needs a logprob at every response position, and
# logits_processor.py:471 materializes [N, vocab] (fp32) in one shot by default.
# N reaches tens of thousands, i.e. an 8-16 GiB single allocation -- exactly where
# the teacher OOM'd. Enabling this routes through
# process_input_logprobs_by_chunk(), capping the peak at
# SGLANG_LOGITS_PROCESSER_CHUNK_SIZE (default 2048, ~1 GiB at TP=2) and thereby
# decoupling logits memory from the prefill chunk size.
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=8192
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}RELAX_OPD_PREEXPANDED_PATCH,SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK,SGLANG_LOGITS_PROCESSER_CHUNK_SIZE"

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/recipes/mopd}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mopd-qwen35-9b-8xgpu-${now}}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-9B}"
TEXT_TEACHER_MODEL_NAME="${TEXT_TEACHER_MODEL_NAME:-Qwen3.5-9B-dapo-teacher-hf}"
VL_TEACHER_MODEL_NAME="${VL_TEACHER_MODEL_NAME:-Qwen3.5-9B-vl-teacher-hf}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/MOPD/train.parquet}"
EVAL_SET="${EVAL_SET:-${PROMPT_SET%/*}/test_small.parquet}"

ACTOR_GPUS="${ACTOR_GPUS:-8}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
TEACHER_GPUS="${TEACHER_GPUS:-4}"
TEACHER_NUM_GPUS_PER_ENGINE="${TEACHER_NUM_GPUS_PER_ENGINE:-2}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/${STUDENT_MODEL_NAME}/"
   --megatron-to-hf-mode bridge
   # --save "${EXP_DIR}/save/mopd-${STUDENT_MODEL_NAME}/"
   # --save-interval 200
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --metadata-key extra_info
   --apply-chat-template

   --multimodal-keys '{"image":"images"}'
   # The default is 16384 per image and mmlongbench rows carry up to 5 images, so a
   # single sample reaches ~82K tokens. On the training side the full-vocab logits
   # are seq x 248320 x 4B / (TP4 x CP2) = 9.45 GiB, which OOMs -- and TP x CP
   # already spans all 8 GPUs, so no further sharding is available. 8192 caps the
   # worst sample at ~49K tokens (logits 5.74 GiB) and roughly halves the rollout
   # and teacher prefill volume as well.
   --image-max-token-num 8192

   --rm-type mopd
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256

   --log-passrate
   --use-fault-tolerance
   --use-streaming-dataset
)

TEACHER_ROUTES="{\"dapo-math-17k\":\"${MODEL_DIR}/${TEXT_TEACHER_MODEL_NAME}/\",\"multimodal-open-r1\":\"${MODEL_DIR}/${VL_TEACHER_MODEL_NAME}/\"}"

OPD_ARGS=(
   --use-opd
   --opd-type sglang
   --opd-only-reward
   --opd-kl-coef 1.0
   --opd-teacher-key data_source
   --opd-teacher-routes "${TEACHER_ROUTES}"
   --teacher-num-gpus-per-engine "${TEACHER_NUM_GPUS_PER_ENGINE}"
   # The teacher is prefill-only, so its peak KV equals one in-flight request
   # (45K tokens is ~1.4 GiB; the largest 5-image rows ~2.6 GiB) -- measured
   # full token usage stays at 0.00-0.02. At 0.4 the KV pool is still ~29 GiB
   # (11x the demand) and ~57 GiB is left for the logits gather peak plus ViT
   # and attention workspace.
   --teacher-sglang-mem-fraction-static "${TEACHER_MEM_FRACTION:-0.4}"
   # A single multi-image request (~45K mm tokens) far exceeds the per-pass chunk
   # budget, so PrefillAdder admits exactly one sequence per iteration: the
   # chunked branch at schedule_policy.py:1101 drains rem_chunk_tokens and breaks
   # out immediately. 4096 is too small -- ~11 passes per request, each too small
   # a matmul for a 9B. 16384 is merely the max_prefill_tokens default; raising
   # both to 32768 cuts it to ~2 passes. This relies on
   # SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK above having decoupled the logits peak
   # from the chunk size.
   --teacher-sglang-chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-32768}"
   --teacher-sglang-max-prefill-tokens "${TEACHER_MAX_PREFILL_TOKENS:-32768}"
   --teacher-sglang-max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-16}"
   --teacher-sglang-disable-cuda-graph
   --opd-token-selection student_sampled
   --opd-log-prob-min-clamp -10.0
   # A multi-image teacher logprob re-prefills the whole multi-image context
   # (~45K tokens per request); 300s times out on the slow tail and
   # _raise_if_all_failed turns that into a hard error.
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
   --use-distributed-optimizer
   --accumulate-allreduce-grads-in-fp32
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXPERIMENT_NAME}"
)

EVAL_ARGS=(
   --skip-eval-before-train
   --eval-prompt-data mopd-9b "${EVAL_SET}"
   --eval-global-batch-size 128
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
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --log-probs-max-tokens-per-gpu 40960
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7
   # The student is serialised the same way (the 8192 default is also far below a
   # single request's 45K mm tokens).
   --sglang-chunked-prefill-size 16384
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
    "${MISC_ARGS[@]}" 2>&1 | tee "mopd-qwen35-9b-${now}.log"
