#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# WebShop agentic GRPO (no distillation), Qwen3.5-35B-A3B (MoE), 8xGPU colocate.

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/agentic_grpo_webshop}"
EXP_DIR="${EXP_DIR:-/root/exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"

DATA_DIR="${DATA_DIR:-/root/webshop-relax}"

WEBSHOP_CONDA_ENV="${WEBSHOP_CONDA_ENV:-/root/miniconda3/envs/relax-opd-webshop}"
WEBSHOP_HOME="${WEBSHOP_HOME:-/root/WebShop}"
WEBSHOP_PORT="${WEBSHOP_PORT:-36001}"
WEBSHOP_MAX_TURNS="${WEBSHOP_MAX_TURNS:-15}"
WEBSHOP_HOST="${WEBSHOP_HOST:-0.0.0.0}"
WEBSHOP_ADVERTISE_HOST="${WEBSHOP_ADVERTISE_HOST:-$(python3 -c 'import ray; print(ray.util.get_node_ip_address())')}"
WEBSHOP_URL="http://${WEBSHOP_ADVERTISE_HOST}:${WEBSHOP_PORT}"

NUM_ROLLOUT="${NUM_ROLLOUT:=150}"

ROLLOUT_BATCH_SIZE=16
ROLLOUT_N_GROUPS=8
ROLLOUT_RESP_LENGTH=512
ROLLOUT_PROMPT_LENGTH=4096
EVAL_ROLLOUT_RESP_LENGTH=512

MODEL_NAME=Qwen3.5-35B-A3B

EXP_NAME=agentic-grpo-webshop-${MODEL_NAME}-${now}
SAVE_DIR=${EXP_DIR}/save/${EXP_NAME}
mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${MODEL_NAME}/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --save-interval 50
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/train.parquet
   --input-key prompt
   --label-key label
   --metadata-key extra_info
   --reward-key score
   --rollout-shuffle
   --apply-chat-template-kwargs '{"enable_thinking": true}'

   --custom-rm-path examples.on_policy_distillation.agentic_opd.webshop.reward_webshop.reward_func
   --use-agentic-rollout
   --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
   --agent-cwd "${SCRIPT_DIR}"

   --agent-env
      "WEBSHOP_PORT=${WEBSHOP_PORT}"
      "WEBSHOP_URL=${WEBSHOP_URL}"
      "WEBSHOP_MAX_TURNS=${WEBSHOP_MAX_TURNS}"

   --num-rollout              ${NUM_ROLLOUT}
   --rollout-batch-size       ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt     ${ROLLOUT_N_GROUPS}
   --rollout-max-prompt-len   ${ROLLOUT_PROMPT_LENGTH}
   --rollout-max-response-len ${ROLLOUT_RESP_LENGTH}
   --rollout-temperature      1.0
   --rollout-top-p            1.0
   --global-batch-size $((ROLLOUT_BATCH_SIZE * ROLLOUT_N_GROUPS))
   --use-fault-tolerance
   --use-streaming-dataset
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data webshop ${DATA_DIR}/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len ${EVAL_ROLLOUT_RESP_LENGTH}
   --eval-temperature 0.4
   --eval-top-p 1.0
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.3
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
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
   --tensor-model-parallel-size ${TP_SIZE:-4}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size ${EP_SIZE:-8}
   --expert-tensor-parallel-size ${ETP_SIZE:-1}
   --calculate-per-token-loss
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${ACTOR_MAX_TOKENS_PER_GPU:-4096}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}
   --sglang-mem-fraction-static ${STUDENT_MEM_FRACTION:-0.7}
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

WEBSHOP_HEALTH_URL="http://127.0.0.1:${WEBSHOP_PORT}"
if curl --noproxy '*' -sS --max-time 2 -o /dev/null "${WEBSHOP_HEALTH_URL}/health"; then
   echo "WebShop server port ${WEBSHOP_PORT} is already occupied." >&2
   exit 1
fi

WEBSHOP_SERVER_LOG="log/${EXP_NAME}-webshop-server.log"
WEBSHOP_CONDA_ENV="${WEBSHOP_CONDA_ENV}" \
WEBSHOP_HOME="${WEBSHOP_HOME}" \
WEBSHOP_HOST="${WEBSHOP_HOST}" \
WEBSHOP_PORT="${WEBSHOP_PORT}" \
bash "${SCRIPT_DIR}/run_webshop_server.sh" >"${WEBSHOP_SERVER_LOG}" 2>&1 &
WEBSHOP_SERVER_PID=$!

cleanup_webshop_server() {
   if kill -0 "${WEBSHOP_SERVER_PID}" 2>/dev/null; then
      kill "${WEBSHOP_SERVER_PID}" 2>/dev/null || true
      for _ in $(seq 1 10); do
         kill -0 "${WEBSHOP_SERVER_PID}" 2>/dev/null || break
         sleep 1
      done
      if kill -0 "${WEBSHOP_SERVER_PID}" 2>/dev/null; then
         kill -9 "${WEBSHOP_SERVER_PID}" 2>/dev/null || true
      fi
   fi
   wait "${WEBSHOP_SERVER_PID}" 2>/dev/null || true
}
trap cleanup_webshop_server EXIT

WEBSHOP_SERVER_READY_TIMEOUT_S="${WEBSHOP_SERVER_READY_TIMEOUT_S:-600}"
WEBSHOP_SERVER_READY_DEADLINE=$((SECONDS + WEBSHOP_SERVER_READY_TIMEOUT_S))
while true; do
   if ! kill -0 "${WEBSHOP_SERVER_PID}" 2>/dev/null; then
      tail -n 100 "${WEBSHOP_SERVER_LOG}" >&2 || true
      exit 1
   fi
   if curl --noproxy '*' -fsS "${WEBSHOP_HEALTH_URL}/health" 2>/dev/null | grep -q '"ready": *true'; then
      break
   fi
   if [ "${SECONDS}" -ge "${WEBSHOP_SERVER_READY_DEADLINE}" ]; then
      tail -n 100 "${WEBSHOP_SERVER_LOG}" >&2 || true
      exit 1
   fi
   sleep 3
done

if [ -z "${RAY_DASHBOARD:-}" ]; then
    if [ -n "${RAY_ADDRESS:-}" ]; then
        RAY_DASHBOARD="http://${RAY_ADDRESS%%:*}:8265"
    else
        RAY_DASHBOARD="http://${HOST_IP:-127.0.0.1}:8265"
    fi
fi

ray job submit --address="${RAY_DASHBOARD}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "{\"actor\": [1, 8], \"rollout\": [1, 8]}" \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   2>&1 | tee log/${EXP_NAME}.log
