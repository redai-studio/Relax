#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full-model text RL on DAPO-Math-17K for 80 updates.
# Use TP2/PP2/CP2/EP64 with decode CUDA graphs disabled.
# Four 32-GPU DP-attention rollout engines, CPU Adam and full recompute:
# selective moe/mlp ran out of memory during the 15th colocated update.
# Per-node host RAM must cover CPU optimizer state, offload backups and Engram.
# This is NOT a validated 128-H800 memory/performance guarantee.
#
# Use an existing 16-node x 8-GPU Ray cluster:
#   MODEL_DIR=/path/to/models DATA_DIR=/path/to/data SAVE_DIR=/path/to/checkpoints \
#     bash scripts/entrypoint/ray-job.sh \
#       scripts/training/text/run-deepseek-v4.1-flash-dapo-math-128xgpu.sh
# Direct bash uses the same existing-cluster entrypoint; it does not create a cluster.
# DRY_RUN=1 prints arguments without Ray or ML imports.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/DeepSeek-V4.1-bf16}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl}"
PROJECT_NAME="${PROJECT_NAME:-Relax/dev/deepseek-v41-dapo-math}"
EXP_NAME="${EXP_NAME:-deepseek-v41-flash-dapo-math-gpu128-bshd}"
NUM_ROLLOUT="${NUM_ROLLOUT:-80}"
LOG_DIR="${LOG_DIR:-log}"
source "${RELAX_ROOT}/scripts/models/deepseek-v4.1-flash.sh"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${HF_CKPT}"
    --megatron-to-hf-mode bridge
    # --load "${SAVE_DIR}/${EXP_NAME}/checkpoints"
    --save "${SAVE_DIR}/${EXP_NAME}/checkpoints"
    --save-interval 100
    --max-actor-ckpt-to-keep 2
)
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --apply-chat-template-kwargs '{"thinking_mode":"thinking"}'
    --rollout-shuffle
    --rollout-seed 42
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size 16
    --n-samples-per-prompt 8
    --rollout-max-response-len 2048
    --rollout-temperature 1
    --rollout-top-p 1
    --rollout-top-k -1
    --global-batch-size 128
    --num-steps-per-rollout 1
    --num-iters-per-train-update 1
    --balance-data
    --use-rollout-routing-replay
)

PERF_ARGS=(
    --qkv-format bshd
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --context-parallel-size 2
    --allgather-cp
    --calculate-per-token-loss
    --expert-model-parallel-size 64
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --micro-batch-size 1
    --max-tokens-per-gpu 2048
    --data-pad-size-multiplier 128
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --recompute-loss-function
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --loss-type policy_loss
    --kl-coef 0
    --kl-loss-coef 0
    --kl-loss-type low_var_kl
    --entropy-coef 0
    --eps-clip 0.2
    --eps-clip-high 0.28
    # No TIS, global advantage normalization, or use-rollout-logprobs override.
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --adam-eps 1e-8
    --clip-grad 1
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --no-rope-fusion
)

SGLANG_ARGS=(
    --sglang-external-model-package relax.backends.sglang.deepseekv41
    # BF16 experts do not fit an 8-GPU H800 engine. TP32/DP4 keeps attention TP8.
    --rollout-num-gpus-per-engine 32
    --sglang-tp-size 32
    --sglang-dp-size 4
    --sglang-ep-size 32
    --sglang-enable-dp-attention
    --sglang-load-format auto
    --sglang-attention-backend dsv4
    --sglang-moe-runner-backend auto
    --sglang-mem-fraction-static 0.7
    --sglang-max-running-requests 128
    --sglang-disable-cuda-graph
    --sglang-disable-radix-cache
    --sglang-watchdog-timeout 3600
)

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --bf16
    --transformer-impl transformer_engine
    --seed 1234
    --deterministic-mode
    --attention-dropout 0
    --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    # Keep 1 GiB spare for non-Torch work; a 3 GiB margin rejects valid H800 MoE allocations.
    --train-memory-margin-bytes 1073741824
    --update-weight-buffer-size 1073741824
    --trust-remote-code
    --use-health-check
)

TRAIN_ARGS=(
    --resource '{"actor": [1, 128], "rollout": [1, 128]}'
    --num-gpus-per-node 8
    --num-data-storage-units 16
    --max-staleness 0
    --colocate
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${PERF_ARGS[@]}"
    "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${SGLANG_ARGS[@]}" "${METRICS_ARGS[@]}" "${MISC_ARGS[@]}"
    "$@"
)

if [ "${DRY_RUN:-0}" != 1 ] && [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    : "${RAY_ADDRESS:?Set RAY_ADDRESS for the existing 128-GPU Ray cluster}"
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
runtime.setdefault("env_vars", {}).update({
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "0",
    "NCCL_CUMEM_ENABLE": "1",
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
    # The 32-GPU engines span hosts; shared memfd tables cannot cross hosts.
    "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "per_rank",
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_DISABLE_MULTIMEM_AG": "1",
    "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
    "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
    "SGLANG_HEALTH_CHECK_TIMEOUT": "900",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
})
for name in ("CUDA_HOME", "CLEARML_CONFIG_FILE", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "NO_VCS_VERSION"):
    if os.environ.get(name):
        runtime["env_vars"][name] = os.environ[name]
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n%s\n' "${RUNTIME_ENV_JSON}"
    exit 0
fi

RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-${RAY_DASHBOARD_ADDRESS:-${RAY_ADDRESS:-http://127.0.0.1:8265}}}"
if [[ "${RAY_JOB_ADDRESS}" != http://* && "${RAY_JOB_ADDRESS}" != https://* ]]; then
    echo "Set RAY_JOB_ADDRESS or RAY_DASHBOARD_ADDRESS to the existing cluster's Ray Jobs HTTP address." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${EXP_NAME}-${now}.log"
