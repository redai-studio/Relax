#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3-VL-8B OSWorld GRPO on 8 or 16 auto-detected colocated GPUs (TP4 train / TP2 rollout).
# Usage (configure env.local.sh first):
#   bash examples/osworld_agentic/scripts/run-qwen3vl-8B-osworld-8xgpu.sh
# MODEL_DIR: initial models; DATA_DIR: datasets; SAVE_DIR: persistent diagnostics.
# Checkpoint saving is intentionally disabled per this experiment's requirements.
# Each launch initializes from HF; this recipe does not resume optimizer state.
set -euo pipefail
# Never trace private configuration or credentials into submission logs.
set +x

RELAX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="${RELAX_ROOT}/examples/osworld_agentic"
# Keep launch-time roots authoritative when loading an older private config.
declare -A launch_roots=()
for root_name in MODEL_DIR DATA_DIR SAVE_DIR OSWORLD_AGENT_PYTHON; do
    if [[ -v "${root_name}" ]]; then
        launch_roots["${root_name}"]="${!root_name}"
    fi
done
if [ -f "${SCRIPT_DIR}/env.local.sh" ]; then
    source "${SCRIPT_DIR}/env.local.sh"
fi
for root_name in "${!launch_roots[@]}"; do
    export "${root_name}=${launch_roots[${root_name}]}"
done
unset launch_roots
: "${OSWORLD_REPO:?Set OSWORLD_REPO in env.local.sh}"
export OSWORLD_CACHE_DIR="${OSWORLD_CACHE_DIR:-${OSWORLD_REPO}/cache}"
export MCP_SRC_ROOT="${MCP_SRC_ROOT:-${OSWORLD_REPO}}"
export MEGATRON=/root/Megatron-LM
# Reject the example placeholders before touching the runtime or preparing data.
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent output root}"
for root_name in MODEL_DIR DATA_DIR SAVE_DIR; do
    case "${!root_name}" in
        /path/to|/path/to/*)
            echo "Set ${root_name} in env.local.sh or the environment." >&2
            exit 1
            ;;
    esac
done
cache_default_agent_env=0
if [ -z "${OSWORLD_AGENT_PYTHON:-}" ]; then
    cache_default_agent_env=1
    bash "${SCRIPT_DIR}/scripts/prepare_agent_env.sh"
    export OSWORLD_AGENT_PYTHON="${DATA_DIR}/osworld-agent-venv/bin/python"
fi
if ! command -v "${OSWORLD_AGENT_PYTHON}" >/dev/null 2>&1; then
    echo "OSWORLD_AGENT_PYTHON is not executable: ${OSWORLD_AGENT_PYTHON}" >&2
    exit 1
fi
export RELAX="${RELAX_ROOT}"
export PYTHONPATH="${RELAX}:${MEGATRON}:${PYTHONPATH:-}"
cd "${RELAX}"
# Prefer the existing cluster; only bootstrap locally when ray status fails.
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    set +u
    if ray status >/dev/null 2>&1; then
        source "${RELAX}/scripts/entrypoint/ray-job.sh"
    else
        source "${RELAX}/scripts/entrypoint/local.sh"
    fi
    set +x
    set -u
fi
# Query total resources, not currently available GPUs or the local CUDA count.
# ray.init(address=...) connects to the cluster and never starts a new one.
cluster_info=$(python3 - <<'PY_RAY'
import os

import ray

context = ray.init(address=os.environ["RAY_ADDRESS"], log_to_driver=False)
try:
    gpu_count = ray.cluster_resources().get("GPU", 0)
    if gpu_count not in (8, 16):
        raise SystemExit(f"OSWorld requires 8 or 16 cluster GPUs; found {gpu_count}")
    gpu_nodes = [n["Resources"]["GPU"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    if len(set(gpu_nodes)) != 1 or gpu_nodes[0] % 4:
        raise SystemExit(f"OSWorld requires uniform GPU nodes divisible by TP4; found {gpu_nodes}")
    dashboard = context.address_info["webui_url"]
    if not dashboard:
        raise SystemExit("Ray cluster has no Jobs dashboard address")
    if not dashboard.startswith(("http://", "https://")):
        dashboard = "http://" + dashboard
    print(int(gpu_count), int(gpu_nodes[0]), dashboard, ray.get_runtime_context().gcs_address)
finally:
    ray.shutdown()
PY_RAY
)
read -r CLUSTER_GPUS GPUS_PER_NODE CLUSTER_JOBS_ADDRESS CLUSTER_GCS_ADDRESS <<< "${cluster_info}"
export RAY_ADDRESS="${CLUSTER_GCS_ADDRESS}"
export RAY_JOBS_ADDRESS="${RAY_JOBS_ADDRESS:-${RAY_JOB_ADDRESS:-${CLUSTER_JOBS_ADDRESS}}}"
echo "OSWorld: ${CLUSTER_GPUS} cluster GPUs, TP4 / DP$((CLUSTER_GPUS / 4)), Jobs ${RAY_JOBS_ADDRESS}"
if [ "${cache_default_agent_env}" -eq 1 ]; then
    agent_python=$(python3 "${SCRIPT_DIR}/scripts/cache_agent_env.py" "${DATA_DIR}/osworld-agent-venv")
    export OSWORLD_AGENT_PYTHON="${agent_python}"
fi
unset cache_default_agent_env
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
source "${RELAX}/scripts/models/qwen3-vl-8B.sh"

export EXP_NAME="${EXP_NAME:-qwen3vl-8b-osworld-${CLUSTER_GPUS}xgpu}"
PROJECT_NAME="${PROJECT_NAME:-Relax/osworld-agentic}"
now=$(date "+%Y%m%d-%H%M%S")
export OSWORLD_RUN_TAG="osworld-outcome-${now}"
TRAIN_DATA="${DATA_DIR}/osworld/train.parquet"
if [ ! -s "${TRAIN_DATA}" ]; then
    python3 "${SCRIPT_DIR}/scripts/prepare_data.py" \
        --input-dir "${OSWORLD_REPO}/evaluation_examples/examples" \
        --output-dir "${DATA_DIR}/osworld" \
        --train-manifest "${SCRIPT_DIR}/scripts/train_tasks.txt" \
        --eval-manifest "${SCRIPT_DIR}/scripts/train_tasks.txt"
fi

export TENSORBOARD_DIR="${SAVE_DIR}/${EXP_NAME}/tensorboard_log/${OSWORLD_RUN_TAG}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" MASTER_PORT="${MASTER_PORT:-23456}"
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
    "OSWORLD_AGENT_PYTHON", "OSWORLD_REPO", "OSWORLD_CACHE_DIR", "MCP_SRC_ROOT", "OSWORLD_RUN_TAG", "TENSORBOARD_DIR",
):
    env[name] = os.environ[name]
env.update({
    "OSWORLD_DISABLE_MCP": "0", "OSWORLD_FORCE_MCP": "1",
    "OSWORLD_MAX_CONCURRENT_GROUPS": "20",
    "OSWORLD_MAX_STEPS_REF": "30", "OSWORLD_PER_TRAJ_NORM": "0",
    "OSWORLD_REWARD_DUMP": os.path.join(
        os.environ["SAVE_DIR"], os.environ["EXP_NAME"],
        "rollout_result", os.environ["OSWORLD_RUN_TAG"], "reward_dump.jsonl",
    ),
    "MALLOC_MMAP_THRESHOLD_": "131072", "MALLOC_ARENA_MAX": "2",
    # GPU nodes may expose different RDMA devices; all ranks need the same transport.
    "NCCL_NET": "Socket", "NCCL_IB_DISABLE": "1",
    "SGLANG_USE_CUDA_IPC_TRANSPORT": "0", "SGLANG_MM_BUFFER_SIZE_MB": "0",
    "SGLANG_MM_FEATURE_CACHE_MB": "4096",
    "SGLANG_VLM_CACHE_SIZE_MB": "4096", "SGLANG_MM_THREAD_WORKERS": "8",
    "RAY_prestart_worker_first_driver": "0", "RAY_ROTATION_MAX_BYTES": "67108864",
    "RAY_ROTATION_BACKUP_COUNT": "1",
})
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(sorted(set(env) | {"RELAX_PROPAGATE_ENV_VARS"}))
print(json.dumps(runtime))
PY
)

# No --load/--save: curve-only experiment, deliberately no checkpoint resume.
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-VL-8B-Thinking"
    --ref-load "${MODEL_DIR}/Qwen3-VL-8B-Thinking"
    --warm-hf-checkpoint-page-cache
    --megatron-to-hf-mode bridge
)
ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key label
    --metadata-key extra_info
    --reward-key score
    --use-agentic-rollout
    --multimodal-keys '{}'
    --rollout-result-dir "${SAVE_DIR}/${EXP_NAME}/rollout_result/${OSWORLD_RUN_TAG}"
    --agent-command "bash \"${SCRIPT_DIR}/run_agent_app.sh\""
    --agent-cwd "${SCRIPT_DIR}"
    --agent-env "OPENAI_MODEL=Qwen3-VL-8B-Thinking"
        "AGENT_DEBUG_LOG_DIR=${SCRIPT_DIR}/log/agent/${OSWORLD_RUN_TAG}"
    --agent-timeout 21600
    --agentic-custom-advantage-path examples.osworld_agentic.app.reward_post_process.compute_advantages
    --custom-rollout-log-function-path examples.osworld_agentic.app.metrics.log_rollout_data
    # Active rollout and prelaunch share 12 group permits (up to 96 sessions).
    --agentic-concurrency 12
    --agentic-prelaunch
    # 12 task groups x 8 trajectories; two updates of 48 trajectories.
    --num-rollout 200
    --rollout-batch-size 12
    --n-samples-per-prompt 8
    --num-steps-per-rollout 2
    --micro-batch-size 1
    --rollout-max-context-len 16384
    --rollout-max-response-len 8192
    --rollout-max-prompt-len 16383
    --rollout-temperature 1
    --skip-eval-before-train
)
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type k3
    --ref-update-interval 1
    --entropy-coef 0
    --eps-clip 0.2
    --eps-clip-high 0.2
    --use-tis
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-6
    --lr-decay-style constant
    --lr-decay-iters 400
    --weight-decay 0
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
)
PERF_ARGS=(
    --freeze-vision-model
    --distributed-timeout-minutes 15
    --use-dynamic-batch-size
    --max-tokens-per-gpu 24576
    --log-probs-max-tokens-per-gpu 24576
    --no-rope-fusion
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --attention-dropout 0
    --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --balance-data
    --recompute-granularity selective
    --recompute-modules core_attn
)
SGLANG_ARGS=(
    # Avoid the CUDA IPC / GPU feature-hash path that stalled the previous run.
    --sglang-mm-feature-transport cpu
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
    --sglang-router-policy consistent_hashing
)

WANDB_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
RESOURCE_ARGS=(
    --num-gpus-per-node "${GPUS_PER_NODE}"
    # Colocation shares the detected physical GPUs; role counts are not additive.
    --resource "{\"actor\": [1, ${CLUSTER_GPUS}], \"rollout\": [1, ${CLUSTER_GPUS}]}"
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
)

mkdir -p "${SCRIPT_DIR}/log" "${SAVE_DIR}/${EXP_NAME}/rollout_result/${OSWORLD_RUN_TAG}"
# Blocking by default; with RAY_NO_WAIT=1, tee captures submission output only.
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOBS_ADDRESS:-${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
