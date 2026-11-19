#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Isolated synchronous RL validation: Qwen3.5-35B-A3B: eight training GPUs and four TP2 engines on separate nodes.
# Launch through scripts/entrypoint/ray-job.sh with MODEL_DIR, DATA_DIR, SAVE_DIR.
# Compare against NCCL's Socket transport; sharded TCP/delta needs separate validation.
# No ordinary training/optimizer checkpoint is saved in this bounded validation.

set -euo pipefail
export RELAX_SKIP_CLUSTER_CLEANUP=1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/ray-job.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-35B-A3B.sh"
: "${MODEL_DIR:?Set MODEL_DIR to the root containing the chosen Qwen3.5-35B checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to the root containing dapo-math-17k}"
: "${SAVE_DIR:?Set SAVE_DIR to the task-owned run directory}"
: "${RAY_JOBS_ADDRESS:?Set RAY_JOBS_ADDRESS to the verified Jobs HTTP endpoint}"
: "${WEIGHT_SYNC_CUDA_ARCHS:?Set WEIGHT_SYNC_CUDA_ARCHS after verifying the allocated GPUs}"
export WEIGHT_SYNC_CUDA_ARCHS
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/delta-weight-sync}"
EXP_NAME="${EXP_NAME:-qwen35-35b-a3b-dapo-16xgpu-weight-sync}"
now=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "${SAVE_DIR}/logs" "${SAVE_DIR}/tmp" "${SAVE_DIR}/cache"
SAVE_DIR="$(cd -- "${SAVE_DIR}" && pwd -P)"

# Carry every cache/temp destination into remote workers, not just the submitter.
RUNTIME_ENV_JSON=$(SAVE_DIR="${SAVE_DIR}" python - <<'PY'
import json
import os
from pathlib import Path
env = json.loads(os.environ["RUNTIME_ENV_JSON"])
env.pop("working_dir", None)
root = Path(os.environ["SAVE_DIR"]).resolve()
overrides = {
    "TORCH_CUDA_ARCH_LIST": os.environ["WEIGHT_SYNC_CUDA_ARCHS"],
    "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(root / "tmp"),
    "XDG_CACHE_HOME": str(root / "cache"), "HF_HOME": str(root / "cache" / "hf"),
    "TRITON_CACHE_DIR": str(root / "cache" / "triton"),
    "TRITON_CACHE_MANAGER": "task_delta_weight_sync.process_cache:ProcessCacheManager",
    "TRITON_DUMP_DIR": str(root / "cache" / "triton-dump"),
    "TRITON_OVERRIDE_DIR": str(root / "cache" / "triton-override"),
    "TORCHINDUCTOR_CACHE_DIR": str(root / "cache" / "inductor"),
    "TORCH_EXTENSIONS_DIR": str(root / "cache" / "extensions"),
    "CUDA_CACHE_PATH": str(root / "cache" / "cuda"),
    "NUMBA_CACHE_DIR": str(root / "cache" / "numba"),
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    # Trial allocations cannot rely on NVLS multicast support merely
    # because nvidia-smi reports NVLink connectivity.
    "NCCL_NVLS_ENABLE": "0",
    # The acceptance control is NCCL over TCP. Keep intra-node P2P/SHM enabled.
    "NCCL_NET": "Socket",
    "NCCL_IB_DISABLE": "1",
    "NCCL_DEBUG_SUBSYS": "INIT,NET",
    "DWS_COLLECT_METRICS": "1",
    "WANDB_DIR": str(root / "logs"), "CLEARML_CACHE_DIR": str(root / "cache" / "clearml"),
    "ETTR_LOG_FILE": str(root / "logs" / "ettr.jsonl"),
    "ETTR_TRACE_LOG_DIR": str(root / "logs" / "ettr"),
    "TENSORBOARD_DIR": str(root / "logs" / "tensorboard"),
    "FLASHINFER_WORKSPACE_BASE": str(root / "cache" / "flashinfer-workspace"),
    "SGLANG_CACHE_DIR": str(root / "cache" / "sglang"),
    "SGLANG_JIT_CACHE_DIR": str(root / "cache" / "sglang" / "jit"),
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
}
env.setdefault("env_vars", {}).update(overrides)
# Controller builds the detached Serve/TrainGroup environment from env.yaml.
# Forward the task-owned cache paths and runtime controls across that boundary.
propagate = env["env_vars"].get("RELAX_PROPAGATE_ENV_VARS", os.environ.get("RELAX_PROPAGATE_ENV_VARS", ""))
env["env_vars"]["RELAX_PROPAGATE_ENV_VARS"] = ",".join(sorted(set(filter(None, propagate.split(","))) | set(overrides)))
print(json.dumps(env))
PY
)
CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME:-Qwen3.5-35B-A3B}"
   --megatron-to-hf-mode bridge
)
ROLLOUT_ARGS=(
   --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.v2.jsonl"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout "${NUM_ROLLOUT:-10}"
   --rollout-batch-size 2
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 1024
   --rollout-max-response-len 1024
   --rollout-max-context-len 2048
   --rollout-temperature 1.0
   --global-batch-size 16
   --micro-batch-size 1
   --seq-length 2048
   --skip-eval-before-train
   --seed 42
)
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --log-probs-max-tokens-per-gpu 2048
   --calculate-per-token-loss
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-distributed-optimizer
)
GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
)
OPTIMIZER_ARGS=(
   # Match the standard 35B GRPO optimizer recipe, including CPU optimizer state offload.
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --no-rope-fusion
   --moe-router-load-balancing-type none
   --moe-aux-loss-coeff 0.0
)
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.65
   --sglang-dtype bfloat16
   --sglang-moe-runner-backend triton
   --sglang-disable-shared-experts-fusion
   --sglang-disable-cuda-graph
   --sglang-enforce-disable-flashinfer-allreduce-fusion
)
if [[ "${CUSTOM_MEGATRON_INIT_PATH:-}" = "task_delta_weight_sync.measure_sharded.install" || "${CUSTOM_MEGATRON_INIT_PATH:-}" = "task_delta_weight_sync.recovery_probe.install" || "${CUSTOM_MEGATRON_INIT_PATH:-}" = "task_delta_weight_sync.measure_sharded_normal.install" ]]; then
    SGLANG_ARGS+=(--sglang-custom-weight-loader task_delta_weight_sync.sharded_gpu_audit.load_audit)
fi
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
if [ -n "${CUSTOM_MEGATRON_INIT_PATH:-}" ]; then
    MISC_ARGS+=(--custom-megatron-init-path "${CUSTOM_MEGATRON_INIT_PATH}")
fi
if [ -n "${MEMORY_SYNC_CONFIG:-}" ]; then
    MISC_ARGS+=(--memory-weight-sync-config "${MEMORY_SYNC_CONFIG}")
fi
ray job submit ${RAY_NO_WAIT:+--no-wait} \
   --address="${RAY_JOBS_ADDRESS}" \
   --submission-id="${SUBMISSION_ID:-${EXP_NAME}-${now}}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- env -C "${SAVE_DIR}" python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   --num-gpus-per-node 8 \
   --train-env-vars '{"NCCL_NVLS_ENABLE":"0","NCCL_DEBUG":"INFO"}' \
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
