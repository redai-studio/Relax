#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# EOPD (arXiv 2603.07079): sampled-token reverse-KL on-policy distillation plus an
# entropy-gated forward-KL loss on the teacher top-k distribution.
# Reproduction target: Qwen3-0.6B-Base student, Qwen3-8B teacher, MATH train (7.5k),
# MATH500 eval Avg@8 / Pass@8 (paper: OPD 50.09/73.20, EOPD 52.02/76.00).
#
# Toggles:
#   EOPD=1  entropy-gated EOPD (the main experiment in this directory)
#   EOPD=0  plain OPD baseline (ablation)
#           Must be set explicitly. A missing or malformed value aborts
#           instead of silently falling through to the baseline.
#   SMOKE=1          tiny smoke-test config (3 rollouts, no eval, no save)
#
# Data prep (once): see prepare_data.py in this directory.
#
# Launch:
#   EXP_DIR=... MODEL_DIR=... DATA_DIR=... EOPD=1 bash examples/on_policy_distillation/eopd/run-eopd-qwen3-0.6B-math-8xgpu.sh

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/eopd}"
EXP_DIR="${EXP_DIR:-/root/exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"

# EOPD selects the training mode and has NO default: an unset or malformed
# value aborts here rather than silently running the plain-OPD baseline.
EOPD="${EOPD:?set EOPD=1 for entropy-gated EOPD or EOPD=0 for the plain OPD baseline}"
case "${EOPD}" in
    0|1) ;;
    *) echo "ERROR: EOPD must be 0 or 1, got '${EOPD}'" >&2; exit 1 ;;
esac
SMOKE="${SMOKE:-0}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3-0.6B-Base}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen3-8B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_DIR}/${TEACHER_MODEL_NAME}}"

# Paper Appendix A: batch 128 prompts x 1 sample, mini-batch 32 (4 grad steps),
# lr 3e-6 cosine, temp 1.0 / top-p 1.0, max response 4096, 3 epochs on 7.5k MATH
# -> 7500/128*3 ~= 176 rollout steps.
NUM_ROLLOUT="${NUM_ROLLOUT:=176}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_RESP_LEN="${ROLLOUT_RESP_LEN:-4096}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
OPD_TOPK="${OPD_TOPK:-16}"
EOPD_TAU="${EOPD_TAU:-0.8}"
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
STUDENT_EOS_TOKEN_ID="${STUDENT_EOS_TOKEN_ID:-151643}"
TEACHER_EOS_TOKEN_ID="${TEACHER_EOS_TOKEN_ID:-151645}"
TEACHER_RESPONSE_PREFIX_TOKEN_IDS="${TEACHER_RESPONSE_PREFIX_TOKEN_IDS:-151667 271 151668 271}"
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-1}"

EXTRA_EVAL_ARGS=()
if [ "${SKIP_EVAL_BEFORE_TRAIN}" = "1" ]; then
    EXTRA_EVAL_ARGS+=(--skip-eval-before-train)
fi
if [ "${SMOKE}" = "1" ]; then
    NUM_ROLLOUT=3
    ROLLOUT_BATCH_SIZE=32
    GLOBAL_BATCH_SIZE=16
    ROLLOUT_RESP_LEN=1024
    EVAL_INTERVAL=1000000
    SAVE_INTERVAL=1000000
fi

if [ "${EOPD}" = "1" ]; then
    MODE_TAG=eopd-tau${EOPD_TAU}-k${OPD_TOPK}
else
    MODE_TAG=opd-baseline
fi
echo "=== MODE: ${MODE_TAG}  (EOPD=${EOPD} EOPD_TAU=${EOPD_TAU} OPD_TOPK=${OPD_TOPK}) ==="

EXP_NAME=${MODE_TAG}-${STUDENT_MODEL_NAME}-t-${TEACHER_MODEL_NAME}-math-${now}

SAVE_DIR=${EXP_DIR}/save/${EXP_NAME}
mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --save-interval ${SAVE_INTERVAL}
)

PROMPT_SET=${DATA_DIR}/math-eopd/math_train.jsonl
EVAL_SET=${DATA_DIR}/math-eopd/math500_test.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout              ${NUM_ROLLOUT}
   --rollout-batch-size       ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt     1
   --rollout-max-response-len ${ROLLOUT_RESP_LEN}
   --rollout-temperature      1
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --use-fault-tolerance
   --balance-data
)

EVAL_ARGS=(
   --log-passrate
   --eval-interval ${EVAL_INTERVAL}
   --eval-prompt-data math500 ${EVAL_SET}
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-temperature 1.0
   --eval-top-p 0.8
   "${EXTRA_EVAL_ARGS[@]}"
)

OPD_ARGS=(
   --use-opd
   --opd-type sglang

   --teacher-hf-checkpoint ${TEACHER_MODEL_PATH}
   --opd-student-eos-token-id ${STUDENT_EOS_TOKEN_ID}
   --opd-teacher-eos-token-id ${TEACHER_EOS_TOKEN_ID}
   --opd-teacher-response-prefix-token-ids ${TEACHER_RESPONSE_PREFIX_TOKEN_IDS}
   --warm-hf-checkpoint-page-cache

   --teacher-sglang-mem-fraction-static 0.5
   --teacher-num-gpus-per-engine 1
   --teacher-sglang-disable-cuda-graph

   --opd-kl-coef 1.0
   --opd-kl-type reverse_kl
   --opd-teacher-timeout-s 6000
   --use-rollout-logprobs
   --opd-disable-rl-reward
)
if [ "${EOPD}" = "1" ]; then
    OPD_ARGS+=(
       --opd-loss-coef 1.0
       --opd-token-selection teacher_topk
       --opd-log-prob-top-k ${OPD_TOPK}
       --opd-norm-mode norm
       --opd-fkl-entropy-gate
       --opd-fkl-entropy-threshold ${EOPD_TAU}
    )
else
    OPD_ARGS+=(
       --opd-loss-coef 0.0
       --opd-token-selection student_sampled
    )
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.2
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LEARNING_RATE}
   --lr-decay-style cosine
   --lr-decay-iters $((NUM_ROLLOUT * ROLLOUT_BATCH_SIZE / GLOBAL_BATCH_SIZE))
   --min-lr 3e-7
   --distributed-timeout-minutes 120
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
   --sglang-max-running-requests 32
   --router-request-timeout-secs 21600
   --router-queue-timeout-secs 3600
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name    ${PROJECT_NAME}
   --tb-experiment-name ${EXP_NAME}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-rope-fusion
)

mkdir -p log

if [ -z "${RAY_DASHBOARD:-}" ]; then
    if [ -n "${RAY_ADDRESS:-}" ]; then
        RAY_DASHBOARD="http://${RAY_ADDRESS%%:*}:8265"
    else
        # MASTER_ADDR is what local.sh hands to `ray start --node-ip-address`,
        # so dialing it always matches wherever the head actually bound.
        # HOST_IP is set by some platforms to a host-level address that is not
        # this container's, which makes `ray job submit` fail with a refused
        # connection even though Ray is running locally.
        RAY_DASHBOARD="http://${MASTER_ADDR:-127.0.0.1}:8265"
    fi
fi

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"actor\": [1, 8], \"rollout\": [1, 4], \"teacher\": [1, 4]}" \
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
   "${EVAL_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   2>&1 | tee log/${EXP_NAME}.log
