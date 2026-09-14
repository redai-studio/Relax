#!/bin/bash

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
export RELAX_OPD_PREEXPANDED_PATCH=1

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/recipes/opd}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

OPD_PRESET="${OPD_PRESET:-student_sampled_reverse_kl_adv}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.6}"

echo "EXP_DIR: ${EXP_DIR}"
echo "MODEL_DIR: ${MODEL_DIR}"
echo "DATA_DIR: ${DATA_DIR}"
echo "OPD_PRESET: ${OPD_PRESET}"
echo "TEACHER_MEM_FRACTION: ${TEACHER_MEM_FRACTION}"

STUDENT_MODEL_NAME="${STUDENT_MODEL_NAME:-Qwen3.5-35B-A3B}"
TEACHER_MODEL_NAME="${TEACHER_MODEL_NAME:-Qwen3.5-122B-A10B}"

ROLLOUT_GPUS="${ROLLOUT_GPUS:-64}"
TEACHER_GPUS="${TEACHER_GPUS:-64}"
ACTOR_GPUS="${ACTOR_GPUS:-128}"

SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/opd-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-v0}"
mkdir -p "${SAVE_DIR}"
CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${STUDENT_MODEL_NAME}/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --load ${SAVE_DIR}
   --save-interval 50
   --max-actor-ckpt-to-keep 2
)

PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/train.relax.jsonl}"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key messages
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle

   --multimodal-keys '{"image":"images"}'
   --image-min-token-num 64                 
   --image-max-token-num ${IMG_MAX_TOKEN:-16384}              

   # pure OPSD: reward is zeroed out by --opd-disable-rl-reward, but RewardWorker does not support rm_type='none';
   --rm-type random

   --num-rollout              ${NUM_ROLLOUT}
   --rollout-batch-size       32
   --n-samples-per-prompt     8
   --rollout-max-prompt-len   ${ROLLOUT_MAX_PROMP:-16384}
   --rollout-max-response-len 1024
   --rollout-temperature      1

   --rollout-result-dir ${SAVE_DIR}/opd-traces-${now}

   --global-batch-size 256
   --use-fault-tolerance
   --use-streaming-dataset
   --balance-data
   --mm-processor-pool-size ${MM_PROCESSOR_POOL_SIZE:-32}
   --rollout-health-check-interval ${ROLLOUT_HEALTH_CHECK_INTERVAL:-60}
   --rollout-health-check-timeout ${ROLLOUT_HEALTH_CHECK_TIMEOUT:-120}
   --rollout-health-check-first-wait ${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-120}
   --rollout-health-check-max-consecutive-failures ${ROLLOUT_HEALTH_CHECK_MAX_CONSECUTIVE_FAILURES:-5}
)


OPD_ARGS=(
   --use-opd
   --opd-type sglang

   --teacher-hf-checkpoint ${MODEL_DIR}/${TEACHER_MODEL_NAME}/
   --warm-hf-checkpoint-page-cache

   --teacher-sglang-mem-fraction-static ${TEACHER_MEM_FRACTION}
   --teacher-sglang-chunked-prefill-size ${TEACHER_CHUNKED_PREFILL_SIZE:-8192}
   --teacher-sglang-max-running-requests ${TEACHER_MAX_RUNNING_REQUESTS:-64}
   --teacher-sglang-disable-cuda-graph
   --teacher-sglang-max-prefill-tokens 16384
   --teacher-num-gpus-per-engine 8
)

case "${OPD_PRESET}" in
   student_sampled_reverse_kl_adv)
      OPD_ARGS+=(
         --opd-kl-coef 1.0
         --opd-loss-coef 0.0
         --opd-kl-type reverse_kl
         --opd-token-selection student_sampled
         --use-rollout-logprobs
      )
      ;;
   teacher_topk_jsd_loss)
      OPD_ARGS+=(
         --opd-kl-coef 0.0
         --opd-loss-coef 1.0
         --opd-kl-type jsd
         --opd-jsd-alpha 0.5
         --opd-token-selection teacher_topk
         --opd-log-prob-top-k 100
      )
      ;;
   student_topk_forward_kl_loss)
      OPD_ARGS+=(
         --opd-kl-coef 0.0
         --opd-loss-coef 1.0
         --opd-kl-type forward_kl
         --opd-token-selection student_topk
         --opd-log-prob-top-k 16
      )
      ;;
   *)
      echo "Unknown OPD_PRESET: ${OPD_PRESET}" >&2
      echo "Supported OPD_PRESET values: student_sampled_reverse_kl_adv, teacher_topk_jsd_loss, student_topk_forward_kl_loss" >&2
      exit 1
      ;;
esac

OPD_ARGS+=(
   # Disable base RL outcome reward -> pure OPSD
   --opd-disable-rl-reward

   --opd-teacher-image-key    images

   --opd-is-clip              2.0
   --opd-teacher-timeout-s    6000
)

EVAL_ARGS=()

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.3
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --lr-warmup-iters 10

   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   # --use-precision-aware-optimizer

   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 16
   --expert-tensor-parallel-size 1

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${ACTOR_MAX_TOKENS_PER_GPU:-20480}
)


SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static ${STUDENT_MEM_FRACTION:-0.7}
   --sglang-load-format dummy
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name    ${PROJECT_NAME}
   --tb-experiment-name opd-128xgpu-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-colocate-${now}
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
   --resource "{\"actor\": [1, ${ACTOR_GPUS}], \"rollout\": [1, ${ROLLOUT_GPUS}], \"teacher\": [1, ${TEACHER_GPUS}]}" \
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
   2>&1 | tee log/opd-128xgpu-${STUDENT_MODEL_NAME}-teacher-${TEACHER_MODEL_NAME}-colocate-${now}.log
