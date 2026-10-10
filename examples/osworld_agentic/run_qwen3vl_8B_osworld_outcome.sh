#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3-VL-8B outcome training on an existing Ray cluster.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[ ! -f "${SCRIPT_DIR}/env.local.sh" ] || source "${SCRIPT_DIR}/env.local.sh"
source "${SCRIPT_DIR}/env.sh"
export RELAX="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${RELAX}:${MEGATRON}:${PYTHONPATH:-}"
cd "${RELAX}"
source "${RELAX}/scripts/models/qwen3-vl-8B.sh"

# Experiment settings. LR decay is measured in optimizer updates and stays
# independent of NUM_ROLLOUT: 45 * 3 = 135 by default.
ROLLOUT_BATCH_SIZE=12
N_SAMPLES_PER_PROMPT=8
OSWORLD_UPDATES_PER_ROLLOUT="${OSWORLD_UPDATES_PER_ROLLOUT:-3}"
case "${OSWORLD_UPDATES_PER_ROLLOUT}" in
    1|2|3|6) ;;
    *) echo "OSWORLD_UPDATES_PER_ROLLOUT must be 1, 2, 3 or 6 (whole groups across DP=2)." >&2; exit 1 ;;
esac
OSWORLD_LEARNING_RATE="${OSWORLD_LEARNING_RATE:-1e-6}"
NUM_ROLLOUT="${NUM_ROLLOUT:-10}"
OSWORLD_LR_DECAY_ITERS="${OSWORLD_LR_DECAY_ITERS:-$((45 * OSWORLD_UPDATES_PER_ROLLOUT))}"

SAVE_PATH="${SAVE_DIR}/Qwen3-VL-8B-osworld-outcome"
if [ -e "${SAVE_PATH}/latest_checkpointed_iteration.txt" ]; then
    echo "Choose a fresh SAVE_DIR; an existing checkpoint must not be overwritten." >&2
    exit 1
fi
TRAIN_DATA="${DATA_DIR}/osworld/train.parquet"
if [ ! -s "${TRAIN_DATA}" ]; then
    python3 "${SCRIPT_DIR}/scripts/prepare_data.py" \
        --input-dir "${OSWORLD_REPO}/evaluation_examples" \
        --output-dir "${DATA_DIR}/osworld" \
        --train-manifest "${SCRIPT_DIR}/scripts/train_tasks.txt" \
        --eval-manifest "${SCRIPT_DIR}/scripts/train_tasks.txt"
fi

export OSWORLD_RUN_TAG="osworld-outcome-$(date +%Y%m%d-%H%M%S)"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" MASTER_PORT="${MASTER_PORT:-23456}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
# Fixed settings from the working 8-GPU recipe. Construct JSON once and reuse
# its keys for nested Ray actors. Credentials are read from env.local.sh by
# the session process, never serialized into the logged runtime environment.
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env = runtime.setdefault("env_vars", {})
for name in (
    "PYTHONPATH", "MASTER_ADDR", "MASTER_PORT", "RAY_ADDRESS",
    "OSWORLD_REPO", "OSWORLD_CACHE_DIR", "MCP_SRC_ROOT", "OSWORLD_RUN_TAG",
):
    env[name] = os.environ[name]
env.update({
    "PYTHONUNBUFFERED": "1", "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1", "RELAX_EVAL_ON_EPOCH_BOUNDARY": "0",
    "OSWORLD_DISABLE_MCP": "0", "OSWORLD_FORCE_MCP": "1",
    "OSWORLD_MAX_CONCURRENT_GROUPS": "20",
    "OSWORLD_MAX_STEPS_REF": "30", "OSWORLD_PER_TRAJ_NORM": "0",
    "OSWORLD_REWARD_DUMP": os.path.join(os.environ["SAVE_DIR"], "reward_dump.jsonl"),
    "MALLOC_MMAP_THRESHOLD_": "131072", "MALLOC_ARENA_MAX": "2",
    "RELAX_TRANSFER_BATCH_GROUPS": "9", "RELAX_MM_RELEASE_TRIM": "1",
    "RELAX_MM_RELEASE_LOG": "1", "RELAX_MM_PIXEL_DTYPE": "bf16",
    "RELAX_MM_IMAGE_POOL": "1", "RELAX_MM_RELEASE_AFTER_TRANSFER": "1",
    "SGLANG_USE_CUDA_IPC_TRANSPORT": "1", "SGLANG_MM_FEATURE_CACHE_MB": "4096",
    "SGLANG_VLM_CACHE_SIZE_MB": "4096", "SGLANG_MM_THREAD_WORKERS": "8",
    "RAY_prestart_worker_first_driver": "0", "RAY_ROTATION_MAX_BYTES": "67108864",
    "RAY_ROTATION_BACKUP_COUNT": "1",
})
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(sorted(set(env) | {"RELAX_PROPAGATE_ENV_VARS"}))
print(json.dumps(runtime))
PY
)

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-VL-8B-Thinking"
    --ref-load "${MODEL_DIR}/Qwen3-VL-8B-Thinking"
    --warm-hf-checkpoint-page-cache
    --save "${SAVE_PATH}"
    --megatron-to-hf-mode bridge
    --save-interval 1
    --max-actor-ckpt-to-keep 3
    --distributed-timeout-minutes 15
)
ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt --label-key label --metadata-key extra_info --reward-key score
    --use-agentic-rollout
    --agent-command "bash \"${SCRIPT_DIR}/run_agent_app.sh\""
    --agent-cwd "${SCRIPT_DIR}"
    --agent-env "OPENAI_MODEL=Qwen3-VL-8B-Thinking"
        "AGENT_DEBUG_LOG_DIR=${SCRIPT_DIR}/log/agent/${OSWORLD_RUN_TAG}"
    --agent-timeout 21600
    --custom-reward-post-process-path examples.osworld_agentic.app.reward_post_process.post_process_rewards
    --custom-rm-path examples.osworld_agentic.reward_osworld.reward_func
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --num-steps-per-rollout "${OSWORLD_UPDATES_PER_ROLLOUT}"
    --micro-batch-size 1
    --rollout-max-context-len 32768
    --rollout-max-response-len 8192
    --rollout-max-prompt-len 16384
    --rollout-temperature 1
    --skip-eval-before-train
)
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss --kl-loss-coef 0.02 --kl-loss-type k3
    --ref-update-interval 1
    --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.2
    --use-tis
)
OPTIMIZER_ARGS=(
    --optimizer adam --lr "${OSWORLD_LEARNING_RATE}"
    --lr-decay-style cosine --lr-decay-iters "${OSWORLD_LR_DECAY_ITERS}"
    --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.98
    --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer --no-rope-fusion
)
MEGATRON_ARGS=(
    --tensor-model-parallel-size 4 --sequence-parallel
    --pipeline-model-parallel-size 1 --context-parallel-size 1
    --attention-dropout 0 --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32
    --attention-backend flash
    --balance-data
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
    --sglang-router-policy consistent_hashing
)

mkdir -p "${SAVE_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    -- python3 "${RELAX}/relax/entrypoints/train.py" \
    --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
    --max-staleness 0 --num-data-storage-units 1 --colocate \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${MEGATRON_ARGS[@]}" "${SGLANG_ARGS[@]}" \
    --use-clearml --use-metrics-service \
    --tb-project-name Relax/dev/osworld-agentic --tb-experiment-name "${OSWORLD_RUN_TAG}" \
    2>&1 | tee "${SAVE_DIR}/${OSWORLD_RUN_TAG}.log"
