#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# OSWorld agentic RL — outcome_only_rl (baseline recipe).
#
# Reproduces the pure ±1 outcome variant from hybrid-routing-agent.
# For context_rl (skip_on_mcp_success + 2-image history), see the
# sibling run_qwen3vl_8B_osworld_context.sh — the two share everything
# except the OSWORLD_CONFIG_PATH env var passed to the agent.
#
# Prerequisites:
#   1. Convert data:   python3 examples/osworld_agentic/scripts/prepare_data.py \
#                        --input-dir /path/to/evaluation_examples \
#                        --output-dir ${DATA_DIR}/osworld/data
#   3. Set env vars:   MODEL_DIR=/path/to/models DATA_DIR=/path/to/data \
#                      SAVE_DIR=/path/to/save bash $0

set -e
set -o pipefail

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ "${OSWORLD_ENV_LOADED:-0}" != "1" ] && [ -f "${SCRIPT_DIR}/env.sh" ]; then
    source "${SCRIPT_DIR}/env.sh"
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-vl-8B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/osworld-agentic}"
EXP_NAME="qwen3vl-8B-osworld-outcome-${TIMESTAMP}"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, SAVE_DIR must be set."
    exit 1
fi
mkdir -p "${SAVE_DIR}"

# Housekeeping: prune stale per-run agent logs (>3 days).
if [ "${OSWORLD_PRUNE_OLD_AGENT_LOGS:-1}" = "1" ]; then
    find "${SCRIPT_DIR}/log/agent" -maxdepth 1 -type d -mtime +3 -exec rm -rf {} + 2>/dev/null || true
fi

# Model and resume checkpoint arguments. Explicit source is separate from new output.
CKPT_REF_LOAD="${MODEL_DIR}/Qwen3-VL-8B-Thinking"
if [ -n "${OSWORLD_RESUME_DIR:-}" ]; then
  if [ "${RESUME_FROM_SAVE:-0}" = "1" ]; then
    echo "Choose OSWORLD_RESUME_DIR or RESUME_FROM_SAVE, not both" >&2
    exit 1
  fi
  if [ ! -f "${OSWORLD_RESUME_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "Missing resume checkpoint tracker: ${OSWORLD_RESUME_DIR}" >&2
    exit 1
  fi
  # This recipe updates ref from actor after every completed training step.
  CKPT_REF_LOAD="${OSWORLD_RESUME_DIR}"
fi

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-VL-8B-Thinking"
    --ref-load "${CKPT_REF_LOAD}"
    --warm-hf-checkpoint-page-cache
    --save "${SAVE_DIR}/Qwen3-VL-8B-osworld-outcome"
    --megatron-to-hf-mode bridge
    --save-interval ${SAVE_INTERVAL:-200}
    --max-actor-ckpt-to-keep ${MAX_CKPT_KEEP:-1}
)

# Resume support: when RESUME_FROM_SAVE=1 and a checkpoint already exists under --save,
# pass --load so a relaunch after a crash continues from the last saved step instead of
# restarting from the base model. Default off, so existing runs are unaffected.
if [ -n "${OSWORLD_RESUME_DIR:-}" ]; then
  CKPT_ARGS+=(--load "${OSWORLD_RESUME_DIR}")
  echo "[resume] actor and ref from ${OSWORLD_RESUME_DIR}; save to ${SAVE_DIR}/Qwen3-VL-8B-osworld-outcome"
elif [ "${RESUME_FROM_SAVE:-0}" = "1" ] && [ -f "${SAVE_DIR}/Qwen3-VL-8B-osworld-outcome/latest_checkpointed_iteration.txt" ]; then
    CKPT_ARGS+=(--load "${SAVE_DIR}/Qwen3-VL-8B-osworld-outcome")
    echo "[resume] --load ${SAVE_DIR}/Qwen3-VL-8B-osworld-outcome (iter $(cat ${SAVE_DIR}/Qwen3-VL-8B-osworld-outcome/latest_checkpointed_iteration.txt))"
fi
# NCCL collective timeout. Default 30 min; a hung training step wastes that long before the
# job exits. Lower it so a crash surfaces fast and the run can be relaunched from checkpoint.
CKPT_ARGS+=(--distributed-timeout-minutes ${DIST_TIMEOUT_MIN:-30})

# Data layout produced by scripts/prepare_data.py:
#   ${DATA_DIR}/osworld/data/train.parquet     (74 tasks; outcome_only_rl split)
#   ${DATA_DIR}/osworld/data/eval.parquet      (309 tasks; OSWorld-MCP full)
TRAIN_FILES=("'${OSWORLD_TRAIN_DATA:-${DATA_DIR}/osworld/data/train.parquet}'")
EVAL_FILES=("${DATA_DIR}/osworld/data/eval.parquet")
PROMPT_SET="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

# Runtime env — propagated to every Ray worker so the session subprocess can
# find the OSWorld config + cache dirs.
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "OSWORLD_CONFIG_PATH": "${OSWORLD_CONFIG_PATH:-${SCRIPT_DIR}/app/osworld_config.yaml}",
    "OSWORLD_CACHE_DIR": "${OSWORLD_CACHE_DIR:-/tmp/osworld_cache}",
    "OSWORLD_REPO": "${OSWORLD_REPO}",
    "OSWORLD_MCP_BONUS": "${OSWORLD_MCP_BONUS:-0}",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "${RAY_OVERRIDE_JOB_RUNTIME_ENV:-0}",
    "RELAX_PROPAGATE_ENV_VARS": "${RELAX_PROPAGATE_ENV_VARS:-}",
    "OSWORLD_PER_TRAJ_NORM": "${OSWORLD_PER_TRAJ_NORM:-1}",
    "OSWORLD_MAX_STEPS_REF": "${OSWORLD_MAX_STEPS_REF:-50}",
    "OSWORLD_LENGTH_PENALTY_COEF": "${OSWORLD_LENGTH_PENALTY_COEF:-0.05}",
    "OSWORLD_CAP_PENALTY": "${OSWORLD_CAP_PENALTY:-0.2}",
    "OSWORLD_ACC_LO": "${OSWORLD_ACC_LO:-0.05}",
    "OSWORLD_ACC_HI": "${OSWORLD_ACC_HI:-0.95}",
    "OSWORLD_ADV_CLIP": "${OSWORLD_ADV_CLIP:-0}",
    "OSWORLD_REWARD_DUMP": "${OSWORLD_REWARD_DUMP:-}",
    "OSWORLD_RUN_TAG": "${OSWORLD_RUN_TAG:-}",
    "OPENAI_MODEL": "Qwen3-VL-8B-Thinking"
  }
}
EOF
)

# Explicitly forward recipe variables to the job driver, then to nested actors.
RUNTIME_ENV_JSON=$(RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON}" python3 -c '
import json, os
runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
for name in os.environ.get("RELAX_PROPAGATE_ENV_VARS", "").split(","):
    name = name.strip()
    if name and name in os.environ:
        runtime_env["env_vars"][name] = os.environ[name]
print(json.dumps(runtime_env))
')

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
    # ALL --agent-env KVs must live under a SINGLE flag — argparse uses
    # `nargs="+"`, not `action="append"`, so repeating the flag silently
    # drops all but the last group. (Memory: 2a732e4e.)
    --agent-env
        "OSWORLD_LOCAL_TORCH_BOOTSTRAP=${OSWORLD_LOCAL_TORCH_BOOTSTRAP:-0}"
        "AGENT_DEBUG_LOG_DIR=${SCRIPT_DIR}/log/agent/${TIMESTAMP}"
        "OSWORLD_CONFIG_PATH=${OSWORLD_CONFIG_PATH:-${SCRIPT_DIR}/app/osworld_config.yaml}"
        "OSWORLD_CACHE_DIR=${OSWORLD_CACHE_DIR:-/tmp/osworld_cache}"
      "OSWORLD_REPO=${OSWORLD_REPO}"
        "OSWORLD_MCP_BONUS=${OSWORLD_MCP_BONUS:-0}"
        "OPENAI_MODEL=Qwen3-VL-8B-Thinking"
        "OSWORLD_MAX_TURNS=${OSWORLD_MAX_TURNS:-50}"
        "OSWORLD_RUN_TAG=${OSWORLD_RUN_TAG:-}"
        "OSWORLD_SHOT_DUMP_DIR=${OSWORLD_SHOT_DUMP_DIR:-}"
        "OSWORLD_SHOT_EVERY_N_TASKS=${OSWORLD_SHOT_EVERY_N_TASKS:-35}"
    # OSWorld sessions are long (up to 50 turns × ~5-15s each ≈ 5-15 min).
    # Give Relax enough headroom that a healthy session never SIGTERMs mid-way.
    --agent-timeout ${AGENT_TIMEOUT:-1500}
    # Long-and-identical system prompts hash into the same SGLang worker under
    # cache_aware (memory c6351b41) — pin to consistent_hashing so agentic
    # sessions spread over engines by session_id, not prefix.
    --sglang-router-policy consistent_hashing
    # Batched VM cold starts prevent the docker-boot storm at rollout 0.
    --agentic-prepare-pool-size ${PREPARE_POOL_SIZE:-8}
    --num-rollout ${NUM_ROLLOUT:-2000}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-8}
    --micro-batch-size 1
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
    --rollout-max-context-len 32768
    --rollout-max-response-len 8192      # upstream max_gen_length: 8192
    --rollout-max-prompt-len ${MAX_PROMPT_LEN:-16384}       # 24576 + 8192 = 32768 context budget
    --rollout-temperature 1
    --global-batch-size ${GLOBAL_BATCH_SIZE:-64}
)

# Opt in to one complete, ordered task set per synchronous rollout.
# The example entry point selects custom hooks after framework argument parsing.
TRAIN_ENTRYPOINT=(relax/entrypoints/train.py)
if [ "${OSWORLD_FIXED_TASK_BATCH:-0}" = "1" ]; then
    TRAIN_ENTRYPOINT=(-m examples.osworld_agentic.train_fixed_tasks)
  if [ "${OSWORLD_NATIVE_MINIBATCH3:-0}" = "1" ]; then
    TRAIN_ENTRYPOINT=(-m examples.osworld_agentic.train_fixed12_minibatch3)
    ROLLOUT_ARGS+=(--num-steps-per-rollout 3)
  fi
else
    ROLLOUT_ARGS+=(--rollout-shuffle --use-streaming-dataset)
fi

EVAL_ARGS=(
    $( [ "${EVAL_BEFORE_TRAIN:-0}" = "1" ] || echo --skip-eval-before-train )   # EVAL_BEFORE_TRAIN=1 -> run held-out eval at step 0
    --eval-interval ${EVAL_INTERVAL:-500}
    --eval-prompt-data osworld_mcp "${EVAL_FILES[@]}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 8192
    --eval-top-p 0.7
    --agentic-eval-prepare-pool-size ${EVAL_PREPARE_POOL_SIZE:-16}
)

if [ "${OSWORLD_DISABLE_EVAL:-0}" = "1" ]; then
  EVAL_ARGS=(--skip-eval-before-train)
fi

# GRPO — matches the upstream 8B recipe: β=0.02 KL, ε=0.2, entropy=0.
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss                     # 2026-09-07 v3: without this the KL term never enters loss (loss.py:1086)
    --kl-loss-coef 0.02
    # kl_reference: "old" -- ref model tracks the actor each rollout step
    # (default None = ref frozen at base, which blocks a narrow signal from accumulating).
    --ref-update-interval 1
    # update_per_step: 4 (outcome_only_rl overrides the base config's 2)
    --num-epoch ${NUM_EPOCH:-4}
    --kl-loss-type k3                    # use_kl_estimator_k3: true (was low_var_kl)
    --entropy-coef ${ENTROPY_COEF:-0.00}
    --eps-clip 0.2
    --eps-clip-high 0.2                 # upstream eps: 0.20 (symmetric; was 0.28)
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr ${OSWORLD_LEARNING_RATE:-5e-6}
    --lr-decay-style ${LR_DECAY:-cosine}
    --weight-decay ${WEIGHT_DECAY:-0.0}
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --no-rope-fusion
)

if [ -n "${OSWORLD_LR_DECAY_ITERS:-}" ]; then
    OPTIMIZER_ARGS+=(--lr-decay-iters "${OSWORLD_LR_DECAY_ITERS}")
fi
if [ "${OSWORLD_OVERRIDE_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
   OPTIMIZER_ARGS+=(--override-opt_param-scheduler)
fi

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine ${GPUS_PER_ENGINE:-2}
    --sglang-mem-fraction-static 0.7
)
# extra --sglang-* flags from the launcher (e.g. "--sglang-tokenizer-worker-num 4"); word-split on purpose
if [ -n "${SGLANG_EXTRA_ARGS:-}" ]; then SGLANG_ARGS+=(${SGLANG_EXTRA_ARGS}); fi

LOG_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
)

MEGATRON_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE:-2}
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-32768}
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-dynamic-batch-size
)

# Seqlen-balanced placement of rollout rows across DP ranks (relax --balance-data).
# Without it rows are placed sequentially: first half -> dp0, second half -> dp1.
# Rollout rows arrive in completion order, so the long straggler trajectories all
# land on the last DP rank (measured 731 vs 2387 samples on rollout 1); the light
# rank finishes its step early and waits inside the DP gradient reduce-scatter
# past the NCCL timeout. Every rollout-1 training step failed this way (9/9).
if [ "${BALANCE_DATA:-1}" = "1" ]; then
    MEGATRON_ARGS+=(--balance-data)
fi

# ---- recompute (激活重算) 开关 ----
# full/uniform/1 = 每层都重算, 最省显存也最慢 (actor_train = 4x forward, step0 实测 3.95x)。
# selective = 只重算注意力里最便宜的部分; none = 完全不重算。
# selective 不能同时传 --recompute-method / --recompute-num-layers (Megatron 会 assert)。
RG="${RECOMPUTE_GRANULARITY:-full}"
if [ "$RG" = "selective" ]; then
    MEGATRON_ARGS+=(--recompute-granularity selective)
elif [ "$RG" = "none" ]; then
    :
else
    MEGATRON_ARGS+=(--recompute-granularity "$RG")
    MEGATRON_ARGS+=(--recompute-method "${RECOMPUTE_METHOD:-uniform}")
    MEGATRON_ARGS+=(--recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}")
fi

# ---- true on-policy: 跳过独立的 old_log_probs 前向 ----
# 前提: rollout_batch_size * n_samples == global_batch_size 且 --max-staleness 0。
# 见 relax/utils/arguments.py:328 与 backends/megatron/actor.py:1590。
if [ "${TRUE_ON_POLICY:-0}" = "1" ]; then
    MEGATRON_ARGS+=(--true-on-policy-mode)
fi

RAY_RESOURCE_ARGS=(
    --resource "{\"actor\": [1, ${GPUS:-8}], \"rollout\": [1, ${GPUS:-8}]}"
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
)

mkdir -p logs

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    -- python3 "${TRAIN_ENTRYPOINT[@]}" \
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
