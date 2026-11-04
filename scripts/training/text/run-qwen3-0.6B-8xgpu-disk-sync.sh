#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Isolated synchronous RL validation: four trainer GPUs and four rollout GPUs.
# Launch through scripts/entrypoint/ray-job.sh with MODEL_DIR, DATA_DIR, SAVE_DIR.
# DISK_SYNC_CONFIG enables disk publication; omit it for the existing NCCL path.
# No ordinary training/optimizer checkpoint is saved in this bounded validation.

set -euo pipefail
export RELAX_SKIP_CLUSTER_CLEANUP=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/ray-job.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"
: "${MODEL_DIR:?Set MODEL_DIR to the root containing Qwen3-0.6B}"
: "${DATA_DIR:?Set DATA_DIR to the root containing dapo-math-17k}"
: "${SAVE_DIR:?Set SAVE_DIR to the task-owned run directory}"
: "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the verified Jobs HTTP endpoint}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/delta-weight-sync}"
EXP_NAME="${EXP_NAME:-qwen3-0.6b-dapo-8xgpu-disk-sync}"
now=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "${SAVE_DIR}/logs" "${SAVE_DIR}/tmp" "${SAVE_DIR}/cache"

# Carry every cache/temp destination into remote workers, not just the submitter.
RUNTIME_ENV_JSON=$(SAVE_DIR="${SAVE_DIR}" python - <<'PY'
import json
import os
from pathlib import Path
env = json.loads(os.environ["RUNTIME_ENV_JSON"])
root = Path(os.environ["SAVE_DIR"])
overrides = {
    "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(root / "tmp"),
    "XDG_CACHE_HOME": str(root / "cache"), "HF_HOME": str(root / "cache" / "hf"),
    "TRITON_CACHE_DIR": str(root / "cache" / "triton"),
    "TORCHINDUCTOR_CACHE_DIR": str(root / "cache" / "inductor"),
    "TORCH_EXTENSIONS_DIR": str(root / "cache" / "extensions"),
    "CUDA_CACHE_PATH": str(root / "cache" / "cuda"),
    "NUMBA_CACHE_DIR": str(root / "cache" / "numba"),
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "WANDB_DIR": str(root / "logs"), "CLEARML_CACHE_DIR": str(root / "cache" / "clearml"),
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
}
env.setdefault("env_vars", {}).update(overrides)
print(json.dumps(env))
PY
)
CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B"
   --megatron-to-hf-mode bridge
)
if [ -n "${DISK_SYNC_CONFIG:-}" ]; then
    CKPT_ARGS+=(--disk-weight-sync-config "${DISK_SYNC_CONFIG}")
fi
ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.v2.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout "${NUM_ROLLOUT:-5}"
   --rollout-batch-size 4
   --n-samples-per-prompt 2
   --rollout-max-response-len 256
   --rollout-temperature 0.8
   --global-batch-size 8
   --skip-eval-before-train
   --seed 42
)
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --log-probs-max-tokens-per-gpu 4096
   --calculate-per-token-loss
)
GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.35
   --sglang-disable-cuda-graph
)
LOG_ARGS=(
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --address="${RAY_JOBS_ADDRESS}" \
   --submission-id="${SUBMISSION_ID:-${EXP_NAME}-${now}}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 4], "rollout": [1, 4]}' \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --num-gpus-per-node 4 \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${LOG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "${SAVE_DIR}/logs/${EXP_NAME}-${now}-submit.log"
