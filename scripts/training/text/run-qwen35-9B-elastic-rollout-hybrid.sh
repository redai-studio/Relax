#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5-9B DAPO-Math hybrid: actor TP4/DP2, rollout engines 2–8 on 16 GPUs.
# MODEL_DIR (root containing Qwen3.5-9B) and DATA_DIR default to EXP_DIR; run:
# bash scripts/training/text/run-qwen35-9B-elastic-rollout-hybrid.sh
# Default: 100 rollouts. CE_ENABLE_AFFINITY=0 disables affinity on tidal workers.
# DRY_RUN=1 previews arguments without Ray/GPU setup or output writes.
set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export RELAX="${RELAX_ROOT}"
EXP_DIR="${EXP_DIR:-/tmp/relax-ce}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "${RELAX}/scripts/entrypoint/ray-job.sh" "${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")" "$@"
fi
source "${RELAX_ROOT}/scripts/models/qwen35-9B.sh"
PROJECT_NAME="${PROJECT_NAME:-Relax/CE/run-qwen35-9B-elastic-rollout-hybrid}"
EXP_NAME="${EXP_NAME:-qwen35-9B-elastic-hybrid}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
now=$(date "+%Y%m%d-%H%M%S")
export SUBMISSION_ID="${SUBMISSION_ID:-${EXP_NAME}-${now}-$$}"

# Elastic workers import this checkout from the Ray working-directory upload.
WORKING_DIR="${WORKING_DIR:-${RELAX_ROOT}}"
# One socket NCCL channel avoids AF_UNIX transport failures in the tidal pool.
SOCKET_NCCL_CHANNELS="${RELAX_SOCKET_NCCL_CHANNELS:-1}"
ENV_ARGS=(
    "RDMA_DISABLE=${RDMA_DISABLE:-1}"
    "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"
    "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}"
    "NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}"
    "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}"
    "NCCL_MIN_NCHANNELS=${SOCKET_NCCL_CHANNELS}"
    "NCCL_MAX_NCHANNELS=${SOCKET_NCCL_CHANNELS}"
    "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}"
    "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${NCCL_SOCKET_IFNAME:-eth0}}"
)
export "${ENV_ARGS[@]}"

# Preserve the standard entrypoint's library paths, cache settings and hooks.
RUNTIME_ENV_JSON=$(python3 - "${ENV_ARGS[@]}" <<'PYRUNTIME'
import json
import os
import sys

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
runtime.setdefault("env_vars", {}).update(item.split("=", 1) for item in sys.argv[1:])
print(json.dumps(runtime, separators=(",", ":")))
PYRUNTIME
)
export RUNTIME_ENV_JSON

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3.5-9B"
    --ref-load "${MODEL_DIR}/Qwen3.5-9B"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

PROMPT_SET="${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rollout-seed 42
    --rm-type dapo
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 8192
    --rollout-temperature 1
    --global-batch-size 256
    --use-fault-tolerance
)

EVAL_ARGS=(
    --log-passrate
    --skip-eval-before-train
    --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 8192
    --eval-top-p 0.7
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 10240
    --no-rope-fusion
)

GRPO_ARGS=(
    --advantage-estimator grpo
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
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.8
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
    --scale-out-timeout 1800
    --scale-out-partial-success-policy keep_partial
    --scale-in-drain-timeout 60
    --scale-weight-sync-precheck
)

TRACKING_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --seed 42
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# Explicit opt-out for recovery-only runs on all-tidal Ray clusters.
if [[ "${CE_ENABLE_AFFINITY:-1}" == "0" ]]; then
    MISC_ARGS+=(--no-enable-affinity)
fi

TRAIN_ARGS=(
    --resource '{"actor": [1, 8], "rollout": [1, 2]}'
    --max-staleness 2
    --num-data-storage-units 1
    --num-iters-per-train-update 32
    --hybrid
    --use-health-check
    --autoscaler-config "${CE_AUTOSCALER_CONFIG:-${SCRIPT_DIR}/configs/qwen35-9b-elastic-autoscaler.yaml}"
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${TRACKING_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${EVAL_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "$@"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${RUNTIME_ENV_JSON}" python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" <<'PYDRY'
import json
import sys

print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYDRY
    exit 0
fi

cd "${RELAX_ROOT}"
mkdir -p "${EXP_DIR:-${RELAX_ROOT}}/log"
RAY_JOB_ADDRESS_ARGS=()
if [[ -n "${RAY_JOBS_ADDRESS:-}" ]]; then
    RAY_JOB_ADDRESS_ARGS=(--address "${RAY_JOBS_ADDRESS}")
elif [[ -n "${HOST_IP:-}" ]]; then
    RAY_JOB_ADDRESS_ARGS=(--address "http://${HOST_IP}:8265")
fi
ray job submit ${RAY_NO_WAIT:+--no-wait} "${RAY_JOB_ADDRESS_ARGS[@]}" \
    --submission-id "${SUBMISSION_ID}" \
    --working-dir "${WORKING_DIR}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${EXP_DIR:-${RELAX_ROOT}}/log/${EXP_NAME}-${now}.log"
