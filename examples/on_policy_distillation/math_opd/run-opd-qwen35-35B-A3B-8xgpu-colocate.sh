#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# STEP_1(grpo, obtain math teacher)
# EXP_DIR=/path/to/your/exp_dir bash scripts/training/text/run-qwen35-35B-A3B-8xgpu-colocate.sh
#
# STEP_2(opd)
# EXP_DIR=/path/to/your/exp_dir bash examples/on_policy_distillation/math_opd/run-opd-qwen35-35B-A3B-8xgpu-colocate.sh

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/mathopd}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

# STUDENT: the pre-RL checkpoint (standard Qwen3.5-35B-A3B).
STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-35B-A3B}"
# TEACHER: RL-trained math checkpoint(from STEP_1)
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen3.5-35B-A3B-GRPO-dapomath17k-400step}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_DIR}/${TEACHER_MODEL_NAME}}"

EXP_NAME=opd-mathopd-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-${now}
SAVE_DIR=${EXP_DIR}/save/${EXP_NAME}
mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --save-interval 2000
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dapo
   --reward-key score

   --num-rollout              ${NUM_ROLLOUT}
   --rollout-batch-size       128
   --n-samples-per-prompt     1
   --rollout-max-response-len 8192
   --rollout-temperature      1
   --global-batch-size 128
   --use-fault-tolerance
   --balance-data
)

EVAL_ARGS=(
   --log-passrate
   # --skip-eval-before-train
   --eval-interval 5
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

OPD_ARGS=(
   --use-opd
   --opd-type sglang

   --teacher-hf-checkpoint ${TEACHER_MODEL_PATH}
   --warm-hf-checkpoint-page-cache

   --teacher-sglang-mem-fraction-static 0.5
   --teacher-num-gpus-per-engine 4
   --teacher-sglang-disable-cuda-graph

   --opd-kl-coef 1.0
   --opd-loss-coef 0.0
   --opd-kl-type reverse_kl
   --opd-token-selection student_sampled

   --opd-teacher-timeout-s 6000
   --use-rollout-logprobs
   --opd-disable-rl-reward
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.3
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --calculate-per-token-loss
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.6
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
   --sglang-max-running-requests 64
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
        RAY_DASHBOARD="http://${HOST_IP:-127.0.0.1}:8265"
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
