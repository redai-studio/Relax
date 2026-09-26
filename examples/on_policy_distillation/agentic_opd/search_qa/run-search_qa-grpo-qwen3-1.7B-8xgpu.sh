#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Search-QA single-domain GRPO, Qwen3-1.7B, 8xGPU colocate.
#
# The "world" is a shared, stateless E5 faiss retrieval HTTP service; start it
# ONCE out of band (not on these 8 training GPUs) and point SEARCH_RETRIEVAL_URL
# at it. See agentic_opd/search_qa/README.md for env setup and data prep.

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
APP_DIR="${SCRIPT_DIR}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen3-1.7B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/agentic_grpo_search_qa}"
EXP_DIR="${EXP_DIR:-/root/exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
MODEL_NAME=Qwen3-1.7B

EXP_NAME="${EXP_NAME:-agentic-grpo-search_qa-${MODEL_NAME}-${now}}"
SAVE_DIR="${SAVE_DIR:-${EXP_DIR}/save/${EXP_NAME}}"
LOAD_DIR="${LOAD_DIR:-}"
START_ROLLOUT_ID="${START_ROLLOUT_ID:-}"
EVAL_ONLY="${EVAL_ONLY:-0}"

DATA_DIR="${DATA_DIR:-/root/search_qa-relax-full}"
EVAL_INTERVAL="${EVAL_INTERVAL:-150}"
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0}"
AGENTIC_EVAL_PREPARE_POOL_SIZE="${AGENTIC_EVAL_PREPARE_POOL_SIZE:-128}"
AGENTIC_PREPARE_POOL_SIZE="${AGENTIC_PREPARE_POOL_SIZE:-8}"

CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
SEARCH_CONDA_ENV="${SEARCH_CONDA_ENV:-relax-opd-search}"
# The retrieval service is shared and external; the agent is only its HTTP client.
SEARCH_RETRIEVAL_URL="${SEARCH_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
SEARCH_RETRIEVAL_AUTOSTART="${SEARCH_RETRIEVAL_AUTOSTART:-1}"
# Autostart runs in a child process; the vars above are plain (unexported) shell
# vars it can't inherit, so pass what it reads inline (retrieval env == agent env).
SEARCH_RETRIEVAL_URL="${SEARCH_RETRIEVAL_URL}" \
SEARCH_RETRIEVAL_AUTOSTART="${SEARCH_RETRIEVAL_AUTOSTART}" \
SEARCH_RETRIEVAL_CONDA_ENV="${SEARCH_CONDA_ENV}" \
CONDA_HOME="${CONDA_HOME}" \
EXP_DIR="${EXP_DIR}" \
bash "${SCRIPT_DIR}/start_retrieval_server.sh"
SEARCH_RETRIEVAL_PREFLIGHT="${SEARCH_RETRIEVAL_PREFLIGHT:-1}"
SEARCH_RETRIEVAL_PREFLIGHT_TIMEOUT="${SEARCH_RETRIEVAL_PREFLIGHT_TIMEOUT:-30}"
SEARCH_RETRIEVAL_TRUST_ENV="${SEARCH_RETRIEVAL_TRUST_ENV:-0}"
SEARCH_RETRIEVAL_TIMEOUT_SECONDS="${SEARCH_RETRIEVAL_TIMEOUT_SECONDS:-60}"
SEARCH_RETRIEVAL_MAX_RETRIES="${SEARCH_RETRIEVAL_MAX_RETRIES:-100}"
SEARCH_RETRIEVAL_RETRY_BUDGET_SECONDS="${SEARCH_RETRIEVAL_RETRY_BUDGET_SECONDS:-600}"
if [ "${SEARCH_RETRIEVAL_PREFLIGHT}" = "1" ]; then
  if ! curl --noproxy '*' -fsS --max-time "${SEARCH_RETRIEVAL_PREFLIGHT_TIMEOUT}" \
    -H 'Content-Type: application/json' \
    -d '{"queries":["health check"],"topk":1,"return_scores":true}' \
    "${SEARCH_RETRIEVAL_URL}" \
    | python3 -c 'import json, sys; payload = json.load(sys.stdin); assert isinstance(payload.get("result"), list) and payload["result"] and payload["result"][0]'; then
    echo "Search retrieval preflight failed: ${SEARCH_RETRIEVAL_URL}" >&2
    exit 1
  fi
fi
SEARCH_MAX_TURNS="${SEARCH_MAX_TURNS:-4}"
SEARCH_HISTORY_LENGTH="${SEARCH_HISTORY_LENGTH:-4}"
SEARCH_SOFT_LENGTH_PENALTY="${SEARCH_SOFT_LENGTH_PENALTY:-1}"
# The 1.7B baseline uses the default "think" tag (no <think> -> <reason> rewrite);
# only the 4B/8B baselines set thinking_tag=reason.
SEARCH_THINKING_TAG="${SEARCH_THINKING_TAG:-think}"
# Per-request timeout for one agent LLM turn; raise to ~180 to ride through the colocate weight-sync window.
SEARCH_LLM_REQUEST_TIMEOUT_SECONDS="${SEARCH_LLM_REQUEST_TIMEOUT_SECONDS:-60}"
export SEARCH_INVALID_ACTION_PENALTY_COEF="${SEARCH_INVALID_ACTION_PENALTY_COEF:-0.01}"

NUM_ROLLOUT="${NUM_ROLLOUT:=150}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"

# Larger prompt batch (128) than ALFWorld/WebShop to keep the rollout engines saturated.
if [ "${EVAL_ONLY}" = "1" ]; then
 if [ -z "${LOAD_DIR}" ]; then
  echo "EVAL_ONLY=1 requires LOAD_DIR to point at a Relax checkpoint directory." >&2
  exit 2
 fi
 NUM_ROLLOUT=1
 EVAL_INTERVAL=1
 START_ROLLOUT_ID=0
 ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
 ROLLOUT_N_GROUPS="${ROLLOUT_N_GROUPS:-2}"
 LEARNING_RATE=0
 SAVE_INTERVAL=1
 SEARCH_EXPORT_ALL_TURNS="${SEARCH_EXPORT_ALL_TURNS:-0}"
else
 ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
 ROLLOUT_N_GROUPS="${ROLLOUT_N_GROUPS:-8}"
 LEARNING_RATE="${LEARNING_RATE:-1e-6}"
 # One sample per session (final <answer> turn); exporting every turn breaks GRPO's equal-group normalization.
 SEARCH_EXPORT_ALL_TURNS="${SEARCH_EXPORT_ALL_TURNS:-0}"
fi
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * ROLLOUT_N_GROUPS))}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-}"
ROLLOUT_RESP_LENGTH=512
# Retrieved <information> blocks make the windowed prompt long (4096 vs 2048 for ALFWorld).
ROLLOUT_PROMPT_LENGTH=4096
ROLLOUT_MAX_CONTEXT_LENGTH=8192
EVAL_ROLLOUT_RESP_LENGTH=512

mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${MODEL_NAME}/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --save-interval ${SAVE_INTERVAL}
   --max-actor-ckpt-to-keep ${MAX_ACTOR_CKPT_TO_KEEP}
)

if [ -n "${LOAD_DIR}" ]; then
 CKPT_ARGS+=(--load "${LOAD_DIR}")
 if [ "${EVAL_ONLY}" = "1" ]; then
  CKPT_ARGS+=(--no-load-optim --no-load-rng)
 fi
fi

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/train.parquet
   --input-key prompt
   --label-key label
   --metadata-key extra_info
   --reward-key score
   --rollout-shuffle
   # Thinking disabled: reasoning goes in the search template's <think> block, not
   # Qwen3's native thinking channel.
   --apply-chat-template-kwargs '{"enable_thinking": false}'

   --custom-rm-path examples.on_policy_distillation.agentic_opd.search_qa.reward_search.reward_func
   --use-agentic-rollout
   --agent-command ". ${APP_DIR}/run_agent_app.sh"
   --agent-cwd "${APP_DIR}"

   --agent-env
      "CONDA_HOME=${CONDA_HOME}"
      "SEARCH_CONDA_ENV=${SEARCH_CONDA_ENV}"
      "SEARCH_RETRIEVAL_URL=${SEARCH_RETRIEVAL_URL}"
      "SEARCH_MAX_TURNS=${SEARCH_MAX_TURNS}"
"SEARCH_RETRIEVAL_TRUST_ENV=${SEARCH_RETRIEVAL_TRUST_ENV}"
"SEARCH_RETRIEVAL_TIMEOUT_SECONDS=${SEARCH_RETRIEVAL_TIMEOUT_SECONDS}"
"SEARCH_RETRIEVAL_MAX_RETRIES=${SEARCH_RETRIEVAL_MAX_RETRIES}"
"SEARCH_RETRIEVAL_RETRY_BUDGET_SECONDS=${SEARCH_RETRIEVAL_RETRY_BUDGET_SECONDS}"
      "SEARCH_HISTORY_LENGTH=${SEARCH_HISTORY_LENGTH}"
      "SEARCH_SOFT_LENGTH_PENALTY=${SEARCH_SOFT_LENGTH_PENALTY}"
      "SEARCH_THINKING_TAG=${SEARCH_THINKING_TAG}"
      "SEARCH_LLM_REQUEST_TIMEOUT_SECONDS=${SEARCH_LLM_REQUEST_TIMEOUT_SECONDS}"
      "SEARCH_EXPORT_ALL_TURNS=${SEARCH_EXPORT_ALL_TURNS}"

   --num-rollout              ${NUM_ROLLOUT}
   --rollout-batch-size       ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt     ${ROLLOUT_N_GROUPS}
   --rollout-max-prompt-len   ${ROLLOUT_PROMPT_LENGTH}
   --rollout-max-response-len ${ROLLOUT_RESP_LENGTH}
   --rollout-max-context-len  ${ROLLOUT_MAX_CONTEXT_LENGTH}
   --rollout-temperature      1.0
   --rollout-top-p            1.0
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --use-fault-tolerance
   --use-streaming-dataset
 --agentic-eval-prepare-pool-size ${AGENTIC_EVAL_PREPARE_POOL_SIZE}
--agentic-prepare-pool-size ${AGENTIC_PREPARE_POOL_SIZE}
)

if [ -n "${START_ROLLOUT_ID}" ]; then
 ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi

# Per-dataset EM columns; fall back to the combined test parquet if no split exists.
EVAL_DATA_PAIRS=()
for _src in nq triviaqa popqa hotpotqa 2wikimultihopqa musique bamboogle; do
    if [ -f "${DATA_DIR}/test_${_src}.parquet" ]; then
        EVAL_DATA_PAIRS+=("${_src}" "${DATA_DIR}/test_${_src}.parquet")
    fi
done
if [ ${#EVAL_DATA_PAIRS[@]} -eq 0 ]; then
    EVAL_DATA_PAIRS=(search_qa "${DATA_DIR}/test.parquet")
fi

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL}"
   --eval-prompt-data "${EVAL_DATA_PAIRS[@]}"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len ${EVAL_ROLLOUT_RESP_LENGTH}
   --eval-temperature "${EVAL_TEMPERATURE}"
   --eval-top-p 1.0
)

if [ "${SKIP_EVAL_BEFORE_TRAIN}" = "1" ]; then
    EVAL_ARGS+=(--skip-eval-before-train)
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   # Plain symmetric PPO clipping (no DAPO clip-higher).
   --eps-clip 0.2
   --eps-clip-high 0.2
   # Lighter KL than ALFWorld/WebShop (0.01): the search policy needs room to move
   # off the base model's non-searching prior.
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr ${LEARNING_RATE}
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.999
   --clip-grad 1.0
)
if [ -n "${LR_WARMUP_FRACTION}" ]; then
  OPTIMIZER_ARGS+=(--lr-warmup-fraction "${LR_WARMUP_FRACTION}")
fi
if [ "${EVAL_ONLY}" = "1" ]; then
  OPTIMIZER_ARGS+=(--lr-decay-iters 1)
fi

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE:-1}
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size ${EP_SIZE:-1}
   --expert-tensor-parallel-size ${ETP_SIZE:-1}
   --calculate-per-token-loss
   --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${ACTOR_MAX_TOKENS_PER_GPU:-8192}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
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
