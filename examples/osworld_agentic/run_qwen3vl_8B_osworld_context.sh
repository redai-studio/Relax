#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# OSWorld agentic RL — context_rl variant.
#
# Reproduces the context_rl recipe from hybrid-routing-agent: same ±1 outcome
# as run_qwen3vl_8B_osworld_outcome.sh, but the agent runs with a shorter
# image history and skips redundant screenshots after a successful MCP call
# (context_policy=skip_on_mcp_success). Everything else — the checkpoint,
# reward shaping, GRPO hyperparameters, colocate layout — is intentionally
# identical to the outcome variant so the two runs stay comparable.
#
# The switch is a single env var: OSWORLD_CONFIG_PATH selects the yaml the
# session driver loads (app/osworld_config_context.yaml vs the baseline
# osworld_config.yaml).
#
# Prerequisites (see the outcome script header for details):
#   1. Convert data:   python3 examples/osworld_agentic/scripts/prepare_data.py ...
#   3. Set env vars:   MODEL_DIR=... DATA_DIR=... SAVE_DIR=... bash $0

set -ex
set -o pipefail

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
[ -f "${SCRIPT_DIR}/env.sh" ] && source "${SCRIPT_DIR}/env.sh"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-8B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/osworld-agentic}"
EXP_NAME="qwen3vl-8B-osworld-context-${TIMESTAMP}"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, SAVE_DIR must be set."
    exit 1
fi
mkdir -p "${SAVE_DIR}"

find "${SCRIPT_DIR}/log/agent" -maxdepth 1 -type d -mtime +3 -exec rm -rf {} + 2>/dev/null || true

# The one behavioural knob that separates this recipe from the outcome one.
OSWORLD_CONFIG_PATH="${SCRIPT_DIR}/app/osworld_config_context.yaml"

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-VL-8B-Thinking
    --ref-load ${MODEL_DIR}/Qwen3-VL-8B-Thinking
    --warm-hf-checkpoint-page-cache
    --save ${SAVE_DIR}/Qwen3-VL-8B-osworld-context
    --megatron-to-hf-mode bridge
    --save-interval ${SAVE_INTERVAL:-200}
    --max-actor-ckpt-to-keep 1
)

TRAIN_FILES=("'${DATA_DIR}/osworld/data/train.parquet'")
EVAL_FILES=("${DATA_DIR}/osworld/data/eval.parquet")
PROMPT_SET="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "OSWORLD_CONFIG_PATH": "${OSWORLD_CONFIG_PATH}",
    "OSWORLD_CACHE_DIR": "${OSWORLD_CACHE_DIR:-/tmp/osworld_cache}",
    "OSWORLD_REPO": "${OSWORLD_REPO}",
    "OSWORLD_MCP_BONUS": "${OSWORLD_MCP_BONUS:-0}",
    "OPENAI_MODEL": "Qwen3-VL-8B-Thinking"
  }
}
EOF
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --metadata-key extra_info
    --reward-key score
    --custom-reward-post-process-path examples.osworld_agentic.app.reward_post_process.post_process_rewards
    --custom-rm-path examples.osworld_agentic.reward_osworld.reward_func
    --use-agentic-rollout
    --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
    --agent-cwd "${SCRIPT_DIR}"
    # Single --agent-env with all KVs (see outcome script for the memory 2a732e4e caveat).
    --agent-env
        "AGENT_DEBUG_LOG_DIR=${SCRIPT_DIR}/log/agent/${TIMESTAMP}"
        "OSWORLD_CONFIG_PATH=${OSWORLD_CONFIG_PATH}"
        "OSWORLD_CACHE_DIR=${OSWORLD_CACHE_DIR:-/tmp/osworld_cache}"
      "OSWORLD_REPO=${OSWORLD_REPO}"
        "OSWORLD_MCP_BONUS=${OSWORLD_MCP_BONUS:-0}"
        "OPENAI_MODEL=Qwen3-VL-8B-Thinking"
    --agent-timeout ${AGENT_TIMEOUT:-1500}
    --sglang-router-policy consistent_hashing
    --agentic-prepare-pool-size ${PREPARE_POOL_SIZE:-8}
    --num-rollout ${NUM_ROLLOUT:-2000}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-8}
    --micro-batch-size 1
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
    --rollout-max-context-len 32768
    --rollout-max-response-len 4096
    --rollout-max-prompt-len 28672
    --rollout-temperature 1
    --global-batch-size ${GLOBAL_BATCH_SIZE:-64}
    --rollout-shuffle
    --use-streaming-dataset
)

EVAL_ARGS=(
    --skip-eval-before-train
    --eval-interval ${EVAL_INTERVAL:-500}
    --eval-prompt-data osworld_mcp "${EVAL_FILES[@]}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 4096
    --eval-top-p 0.7
    --agentic-eval-prepare-pool-size ${EVAL_PREPARE_POOL_SIZE:-16}
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.02
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --no-rope-fusion
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine ${GPUS_PER_ENGINE:-2}
    --sglang-mem-fraction-static 0.7
)

LOG_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
)

MEGATRON_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-32768}
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-dynamic-batch-size
)

RAY_RESOURCE_ARGS=(
    --resource "{\"actor\": [1, ${GPUS:-8}], \"rollout\": [1, ${GPUS:-8}]}"
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
)

mkdir -p logs

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    -- python3 relax/entrypoints/train.py \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    2>&1 | tee logs/${EXP_NAME}.log
