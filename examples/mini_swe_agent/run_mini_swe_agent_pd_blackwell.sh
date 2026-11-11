#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5 Mini SWE on eight H20G GPUs.
# Fully async: four training GPUs and TP2 Prefill + TP2 Decode.
# Launch through scripts/entrypoint/ray-job.sh; paths are supplied by the caller.
# Use the Relax CUDA 13 image with its default PD Router patch.

set -eo pipefail

if ((BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1))); then
   echo "This example requires Bash 5.1 or newer." >&2
   exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" || -z "${RUNTIME_ENV_JSON:-}" ]]; then
   echo "Launch through scripts/entrypoint/ray-job.sh on a single-node, 8-GPU Ray cluster." >&2
   exit 1
fi

# Model and run paths.
: "${MODEL_DIR:?Set MODEL_DIR to the parent of Qwen3.5-35B-A3B}"
: "${R2E_DATA_PATH:?Set R2E_DATA_PATH to the prepared parquet data directory}"
: "${R2E_SIF_DIR:?Set R2E_SIF_DIR to the prepared SIF directory}"
: "${EXP_ROOT:?Set EXP_ROOT to a new directory for this run}"
: "${MINI_SWE_AGENT_VENV:?Set MINI_SWE_AGENT_VENV to the mini-swe-agent venv}"
[[ -x "${MINI_SWE_AGENT_VENV}/bin/python" ]] || { echo "Missing mini-swe-agent venv" >&2; exit 1; }
[[ -f "${MODEL_DIR}/Qwen3.5-35B-A3B/config.json" ]] || { echo "Missing Qwen3.5-35B-A3B model" >&2; exit 1; }
source "${RELAX_DIR}/scripts/models/qwen35-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/mini_swe_agent_pd_blackwell}"
RUN_ID="${RUN_ID:-mini-swe-blackwell-async-$(date +%Y%m%d-%H%M%S)}"
NUM_ROLLOUT="${NUM_ROLLOUT:-20}"

# Sampling and streaming: one optimizer update per complete rollout partition.
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
NUM_ITERS_PER_TRAIN_UPDATE="${NUM_ITERS_PER_TRAIN_UPDATE:-8}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
MAX_STALENESS="${MAX_STALENESS:-1}"
# TP4 keeps one complete 128K context in each packed microbatch.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-131072}"
LOG_PROBS_MAX_TOKENS_PER_GPU="${LOG_PROBS_MAX_TOKENS_PER_GPU:-${MAX_TOKENS_PER_GPU}}"
for value in "${GLOBAL_BATCH_SIZE}" "${NUM_ITERS_PER_TRAIN_UPDATE}" "${MAX_CONTEXT_LEN}" \
   "${MAX_TOKENS_PER_GPU}" "${LOG_PROBS_MAX_TOKENS_PER_GPU}"; do
   [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Batch and token budgets must be positive integers" >&2; exit 1; }
done
((GLOBAL_BATCH_SIZE % (N_SAMPLES_PER_PROMPT * NUM_ITERS_PER_TRAIN_UPDATE) == 0)) || {
   echo "GBS must contain complete eight-member Groups in every streaming batch" >&2; exit 1;
}
((MAX_TOKENS_PER_GPU >= MAX_CONTEXT_LEN && LOG_PROBS_MAX_TOKENS_PER_GPU >= MAX_CONTEXT_LEN)) || {
   echo "Both token budgets must fit the complete context" >&2; exit 1;
}
((MAX_TOKENS_PER_GPU % 8192 == 0 && LOG_PROBS_MAX_TOKENS_PER_GPU % 8192 == 0)) || {
   echo "Token budgets must be divisible by the TP4 padding size (8192)" >&2; exit 1;
}
[[ "${MAX_STALENESS}" =~ ^[0-9]+$ ]] || { echo "MAX_STALENESS must be nonnegative" >&2; exit 1; }
ROLLOUT_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / N_SAMPLES_PER_PROMPT))
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * 3 / 2))}"
[[ "${OVER_SAMPLING_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] && \
   ((OVER_SAMPLING_BATCH_SIZE >= ROLLOUT_BATCH_SIZE)) || {
   echo "OVER_SAMPLING_BATCH_SIZE must be at least ROLLOUT_BATCH_SIZE" >&2; exit 1;
}

# Mini SWE server and physical Session capacity.
export AGENT_SERVER_HOST="${MASTER_ADDR:-127.0.0.1}"
export AGENT_SERVER_PORT="${AGENT_SERVER_PORT:-8765}"
export AGENT_SERVER_URL="http://${AGENT_SERVER_HOST}:${AGENT_SERVER_PORT}"
export AGENT_SERVER_WORK_DIR="${EXP_ROOT}/agent_server"
export AGENT_SERVER_TRAIN_CONCURRENCY="${AGENT_SERVER_TRAIN_CONCURRENCY:-128}"
export AGENT_SERVER_EVAL_CONCURRENCY="${AGENT_SERVER_EVAL_CONCURRENCY:-64}"
export AGENT_SERVER_REWARD_CONCURRENCY="${AGENT_SERVER_REWARD_CONCURRENCY:-24}"
export AGENT_SERVER_MAX_TURNS="${AGENT_SERVER_MAX_TURNS:-100}"
export AGENT_SERVER_RESUMABLE_GENERATION=1
export AGENT_SERVER_SEED="${AGENT_SERVER_SEED:-42}"
export AGENT_SERVER_SHUFFLE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export MINI_SWE_AGENT_VENV R2E_DATA_PATH R2E_SIF_DIR RELAX_DIR
[[ "${AGENT_SERVER_TRAIN_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] && \
   ((AGENT_SERVER_TRAIN_CONCURRENCY >= N_SAMPLES_PER_PROMPT)) || {
   echo "Session capacity must fit one complete eight-member Group" >&2; exit 1;
}
if [[ "${ENABLE_EVAL:-0}" == 1 ]] && ((AGENT_SERVER_EVAL_CONCURRENCY < 50)); then
   echo "Enabled Eval requires at least 50 dedicated Eval slots." >&2
   exit 1
fi
mkdir -p "${EXP_ROOT}" "${AGENT_SERVER_WORK_DIR}"
DUMMY_DATA="${EXP_ROOT}/relax_dummy.jsonl"
seq 50 | sed 's/.*/{"input":[{"role":"user","content":""}]}/' > "${DUMMY_DATA}"

# Ray Jobs do not inherit the submitter's environment. The shared PD driver
# owns server startup and cleanup; this script supplies the training arguments.
RUNTIME_ENV_JSON=$(python3 - <<'ENV'
from importlib.util import find_spec
import json
import os
from pathlib import Path
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
# Resolve libraries from this Python environment, without importing CUDA modules.
libraries = []
for module in ("nvidia.nvshmem", "torch"):
    try:
        spec = find_spec(module)
    except ModuleNotFoundError:
        spec = None
    locations = spec.submodule_search_locations if spec is not None else ()
    library = next(
        (Path(path) / "lib" for path in locations or () if (Path(path) / "lib").is_dir()), None
    )
    if library is None:
        raise SystemExit(f"Missing {module} runtime libraries in the current Python environment")
    libraries.append(str(library))
paths = libraries + env.get("LD_LIBRARY_PATH", "").split(":")
env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(path for path in paths if path))
keys = ["MINI_SWE_AGENT_VENV", "R2E_DATA_PATH", "R2E_SIF_DIR", "LITELLM_LOCAL_MODEL_COST_MAP"]
keys += [key for key in os.environ if key.startswith("AGENT_SERVER_")]
env.update({key: os.environ[key] for key in keys})
# Load the installed Rust cache extension; use this host's IPv4 RoCEv2 GID.
env.update(SGLANG_RUST_BUILD_MODE="never", MC_GID_INDEX="3")
env["RAY_JOB_STOP_WAIT_TIME_S"] = "180"
env["PYTHONPATH"] = os.environ["RELAX_DIR"] + ":" + env.get("PYTHONPATH", "")
print(json.dumps(runtime))
ENV
)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.5-35B-A3B"
   --ref-load "${MODEL_DIR}/Qwen3.5-35B-A3B"
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data "${DUMMY_DATA}"
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --num-steps-per-rollout 1
   --rollout-max-response-len "${MAX_RESPONSE_LEN}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-temperature 1
   --rollout-seed 42
   --balance-data
   --use-fault-tolerance
)

EVAL_ARGS=()
if [[ "${ENABLE_EVAL:-0}" == 1 ]]; then
   EVAL_ARGS=(
      --eval-interval 100
      --eval-prompt-data r2e_eval "${DUMMY_DATA}"
      --n-samples-per-eval-prompt 1
      --agentic-eval-concurrency 50
   )
fi

AGENTIC_ARGS=(
   --use-agentic-rollout
   --agentic-concurrency "${OVER_SAMPLING_BATCH_SIZE}"
   --agentic-session-capacity-per-node "${AGENT_SERVER_TRAIN_CONCURRENCY}"
   --agent-command "bash ${SCRIPT_DIR}/agent_client.sh"
   --agent-cwd "${SCRIPT_DIR}"
   --agent-resource-owner-path relax.agentic.runner.resources.ReceiptResourceOwner
   --agent-timeout 7200
   --agent-env "AGENT_SERVER_URL=${AGENT_SERVER_URL}"
   --agentic-tool-call-parser qwen3_coder
   --agentic-reasoning-parser qwen3
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --seq-length "${MAX_CONTEXT_LEN}"
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-max-tokens-per-gpu "${LOG_PROBS_MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-1024}"
   --data-pad-size-multiplier 2048
   --calculate-per-token-loss
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --overlap-grad-reduce
   --overlap-param-gather
   --no-rope-fusion
   # Full 128K trajectories still need recomputation on four training GPUs.
   --recompute-granularity full
   --recompute-method block
   --recompute-num-layers 32
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
   # Full offload keeps native FP32 parameters on the CPU Adam update path.
   --optimizer-cpu-offload
   --optimizer-offload-fraction 1.0
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-config "${SCRIPT_DIR}/pd-hicache-cpu-1p1d-tp2-blackwell.yaml"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.90}"
   --sglang-load-format dummy
   --sglang-disaggregation-transfer-backend mooncake
   # RoCE HCAs attached to this host's four rollout GPUs; exclude InfiniBand.
   --sglang-disaggregation-ib-device mlx5_gdr_4,mlx5_gdr_5,mlx5_gdr_6,mlx5_gdr_7
   --sglang-router-prefill-policy consistent_hashing
   --sglang-router-decode-policy consistent_hashing
   # Planned aborts in this single P/D pair must not trip the router circuit breaker.
   --sglang-router-disable-circuit-breaker
   --sglang-context-length "${MAX_CONTEXT_LEN}"
)

LOG_ARGS=(
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${RUN_ID}"
)
if [[ "${ENABLE_CLEARML:-1}" == 1 ]]; then
   LOG_ARGS+=(--use-clearml)
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --train-env-vars '{"CUDA_DEVICE_MAX_CONNECTIONS":"1","PYTORCH_ALLOC_CONF":"expandable_segments:True"}'
)

RAY_RESOURCE_ARGS=(
   --resource '{"actor": [1, 4], "rollout": [1, 4], "advantages": [1, 0]}'
   --fully-async
   --max-staleness "${MAX_STALENESS}"
   --num-iters-per-train-update "${NUM_ITERS_PER_TRAIN_UPDATE}"
   --num-data-storage-units 1
)

ray job submit ${RAY_NO_WAIT:+--no-wait} --address "${RAY_JOB_ADDRESS:-http://127.0.0.1:8265}" \
   --submission-id "${RUN_ID}" --runtime-env-json "${RUNTIME_ENV_JSON}" \
   -- bash "${SCRIPT_DIR}/run_mini_swe_agent_pd.sh" --ray-driver \
   "${RAY_RESOURCE_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${AGENTIC_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${LOG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "$@" 2>&1 | tee "${EXP_ROOT}/submit.log"

echo "Job: ${RUN_ID}"
echo "Logs: ray job logs --address ${RAY_JOB_ADDRESS:-http://127.0.0.1:8265} ${RUN_ID}"
