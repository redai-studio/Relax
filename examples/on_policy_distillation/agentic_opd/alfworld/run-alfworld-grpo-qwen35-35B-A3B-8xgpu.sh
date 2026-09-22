#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# ALFWorld agentic GRPO (no distillation), Qwen3.5-35B-A3B (MoE), 8xGPU colocate.
# Rollout: 2P2D, TP2/EP1 per engine (4 P GPUs + 4 D GPUs); actor remains TP4/EP8.

set -ex
set -o pipefail

export NCCL_NVLS_ENABLE=0
now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/agentic_grpo_alfworld}"
EXP_DIR="${EXP_DIR:-/root/exps}"
EXP_NAME=agentic-grpo-alfworld-2p2d-tp2-${now}
SAVE_DIR=${EXP_DIR}/save/${EXP_NAME}

MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-/root/alfworld-relax}"

CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
ALFWORLD_CONDA_ENV="${ALFWORLD_CONDA_ENV:-relax-opd-alfworld}"
ALFWORLD_VENV="${ALFWORLD_VENV:-}"
ALFWORLD_DATA="${ALFWORLD_DATA:-/root/alfworld}"

NUM_ROLLOUT="${NUM_ROLLOUT:=20}"

# Multi-turn budget, sized from measured 2-step runs on this corpus
# (Qwen3.5-35B-A3B, 8xH20, 128 sessions/step):
#   * each turn costs ~540 prompt tokens (the ~33-entry admissible-action list
#     dominates, and it is re-sent every turn) + ~230 generated  => ~770/turn;
#   * a full MAX_TURNS=40 episode therefore needs ~31K, and long <think> turns
#     push past that -- measured trajectories pinned the ceiling exactly.
# Undersizing either budget does not just clip text: app/agent.py ends the
# episode, so the trajectory trains as a failure the policy never caused.
ALFWORLD_MAX_TURNS="${ALFWORLD_MAX_TURNS:=40}"
ALFWORLD_HISTORY_LENGTH="${ALFWORLD_HISTORY_LENGTH:=2}"

# 32 prompts x 8 samples = 256 concurrent agent sessions. Measured against 16:
# +137% train tokens/step for only +58% step time => throughput 3459 -> 5183 tok/s,
# MFU 24.7% -> 35.5%. Rollout is the bottleneck (train_wait ~56% of the step) and
# the KV pool sits at ~18%, so the engine absorbs the extra sessions for free.
# Ceiling to watch: 256 sessions cost ~880 GB host RAM (~3.2 GB per agent process),
# so doubling again would not fit in 1.9 TB.
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=32}"
ROLLOUT_N_GROUPS="${ROLLOUT_N_GROUPS:=8}"
# Per-TURN generation cap (not a whole-episode budget). Mean use is only ~230
# tokens/turn, but the <think> tail matters: measured A/B at 128 sessions,
# 1024 -> 2048 cut length-aborts 40.6% -> 21.9% and lifted success 0.43 -> 0.55.
ROLLOUT_RESP_LENGTH="${ROLLOUT_RESP_LENGTH:=2048}"
ROLLOUT_PROMPT_LENGTH="${ROLLOUT_PROMPT_LENGTH:=2048}"
# 32768 is a deliberate memory trade, not the length the task wants. Measured over
# 12 rollouts of a 17-step run: trajectory mean falls 17.7K -> 12.2K as the policy
# improves, but the max stays 29.7-39.2K and 9 of those 12 rollouts exceeded 32768.
# So this WILL clip the long tail again -- and that tail is the hard task types
# (pick_clean 0.645 / pick_heat 0.696 success), where a clip trains as a failure the
# policy never caused. 40960 clipped nothing (context_exhausted measured 0.0) but the
# training step then sat ~256MB from OOM at the DP gradient boundary and died at
# step 17. Raise this back toward 40960 whenever the memory side gets a real fix
# (CP=2, or chunked logits for the RL path).
ROLLOUT_MAX_CONTEXT_LENGTH="${ROLLOUT_MAX_CONTEXT_LENGTH:=32768}"
EVAL_ROLLOUT_RESP_LENGTH="${EVAL_ROLLOUT_RESP_LENGTH:=${ROLLOUT_RESP_LENGTH}}"
# Resident prompt groups. Concurrent agent subprocesses = this * ROLLOUT_N_GROUPS.
AGENTIC_CONCURRENCY="${AGENTIC_CONCURRENCY:=${ROLLOUT_BATCH_SIZE}}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:=8}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:=2}"
SGLANG_CONFIG="${SGLANG_CONFIG:-${SCRIPT_DIR}/sglang-2p2d-tp2-qwen35-35B-A3B.yaml}"
if [ "${ROLLOUT_NUM_GPUS}" -ne 8 ] || [ "${ROLLOUT_NUM_GPUS_PER_ENGINE}" -ne 2 ]; then
    echo "This PD recipe requires 8 rollout GPUs and 2 GPUs per engine." >&2
    exit 1
fi
# There are two engines in EACH phase. Size each D engine for half the resident
# sessions with 25% routing margin, not one quarter across all four P/D engines.
NUM_DECODE_ENGINES=2
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:=$((AGENTIC_CONCURRENCY * ROLLOUT_N_GROUPS * 5 / (NUM_DECODE_ENGINES * 4)))}"

# These must reach the Ray engine actors, then their SGLang HTTP processes.
# HTTP idle keepalive is distinct from the PD transfer waiting timeout.
export SGLANG_TIMEOUT_KEEP_ALIVE="${SGLANG_TIMEOUT_KEEP_ALIVE:-600}"
# The installed wheel uses RDMA; this value selects GPU metadata handling,
# not an available NVLink transport allocator. Keep memory saver enabled.
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL="${SGLANG_MOONCAKE_CUSTOM_MEM_POOL:-INTRA_NODE_NVLINK}"
RUNTIME_ENV_JSON=$(python3 - <<'PYENV'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env_vars = runtime.setdefault("env_vars", {})
for name in ("SGLANG_TIMEOUT_KEEP_ALIVE", "SGLANG_MOONCAKE_CUSTOM_MEM_POOL"):
    env_vars[name] = os.environ[name]
print(json.dumps(runtime))
PYENV
)
export RUNTIME_ENV_JSON

MODEL_NAME=Qwen3.5-35B-A3B


mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/${MODEL_NAME}/
   --ref-load ${MODEL_DIR}/${MODEL_NAME}/
   --megatron-to-hf-mode bridge
   --save ${SAVE_DIR}
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data ${DATA_DIR}/train.parquet
   --input-key prompt
   --label-key label
   --metadata-key extra_info
   --reward-key score
   --rollout-shuffle
   --apply-chat-template-kwargs '{"enable_thinking": true}'

   --custom-rm-path examples.on_policy_distillation.agentic_opd.alfworld.reward_alfworld.reward_func
   --use-agentic-rollout
   --agent-command ". ${SCRIPT_DIR}/run_agent_app.sh"
   --agent-cwd "${SCRIPT_DIR}"

   --agent-env
      "CONDA_HOME=${CONDA_HOME}"
      "ALFWORLD_CONDA_ENV=${ALFWORLD_CONDA_ENV}"
      "ALFWORLD_VENV=${ALFWORLD_VENV}"
      "ALFWORLD_DATA=${ALFWORLD_DATA}"
      "ALFWORLD_MAX_TURNS=${ALFWORLD_MAX_TURNS}"
      "ALFWORLD_HISTORY_LENGTH=${ALFWORLD_HISTORY_LENGTH}"
      # Same variable as --rollout-max-response-len below, so the two cannot
      # drift: app/agent.py compares it against usage.completion_tokens to
      # split "finish_length" from "context_exhausted".
      "ALFWORLD_MAX_RESPONSE_LEN=${ROLLOUT_RESP_LENGTH}"

   --num-rollout              ${NUM_ROLLOUT}
   --rollout-batch-size       ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt     ${ROLLOUT_N_GROUPS}
   --rollout-max-prompt-len   ${ROLLOUT_PROMPT_LENGTH}
   --rollout-max-response-len ${ROLLOUT_RESP_LENGTH}
   --rollout-max-context-len  ${ROLLOUT_MAX_CONTEXT_LENGTH}
   --agentic-concurrency      ${AGENTIC_CONCURRENCY}
   --rollout-temperature      1.0
   --rollout-top-p            1.0
   --global-batch-size $((ROLLOUT_BATCH_SIZE * ROLLOUT_N_GROUPS))
   --use-fault-tolerance
   --use-streaming-dataset
)


EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data alfworld_unseen ${DATA_DIR}/test.parquet
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
   # A whole multi-turn trajectory is ONE training sample, so the per-GPU token
   # budget must hold the longest one: max_tokens_per_gpu * CP >= max context len
   # (enforced in relax/utils/arguments.py). Sizing it off the response length
   # instead silently OOMs on the first long episode.
   --max-tokens-per-gpu ${ACTOR_MAX_TOKENS_PER_GPU:-${ROLLOUT_MAX_CONTEXT_LENGTH}}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
   --sglang-config "${SGLANG_CONFIG}"
   # SGLang reserves input/output slots internally (6 tokens in this build).
   # Keep the real RL/training budget above at 32768; only enlarge the service
   # envelope so a legal near-limit request is not rejected as an infra error.
   --sglang-context-length $((ROLLOUT_MAX_CONTEXT_LENGTH + 64))
   --sglang-router-policy consistent_hashing
   --sglang-router-prefill-policy consistent_hashing
   --sglang-router-decode-policy consistent_hashing
   --sglang-load-format dummy
   --sglang-enable-weights-cpu-backup
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
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
