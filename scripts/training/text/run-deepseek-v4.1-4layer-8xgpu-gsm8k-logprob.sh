#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Four-layer V4.1 GSM8K GRPO/log-prob comparison, colocated on 8 H800 GPUs.
# TP2/SP + PP2 + CP2 + EP4/ETP1, full/uniform/1 recompute, flex/DeepEP,
# CPU optimizer offload, train/rollout offload and MoE routing replay (R3).
# Use a prepared BF16 checkpoint with its frozen FP8 Engram table and tokenizer.
# PROMPT_DATA must contain prepared GSM8K messages/label columns.
# BSHD keeps fixed microbatches; THD enables multi-sample dynamic packing.
#
# Online (two updates by default), with MODEL_DIR/DATA_DIR/SAVE_DIR supplied:
#   QKV_FORMAT=bshd bash scripts/entrypoint/ray-job.sh \
#     scripts/training/text/run-deepseek-v4.1-4layer-8xgpu-gsm8k-logprob.sh
# Dumps go to SAVE_DIR/EXP_NAME/debug/{rollout_data,train_data}.
# Replay the same dump from the same initial weights, using a fresh EXP_NAME:
#   QKV_FORMAT=thd EXP_NAME=v41-gsm8k-replay-thd bash scripts/entrypoint/ray-job.sh \
#     scripts/training/text/run-deepseek-v4.1-4layer-8xgpu-gsm8k-logprob.sh \
#     --load-debug-rollout-data '/path/to/saved/rollout_data/{rollout_id}.pt'
# That existing flag selects train-only: no SGLang or online weight-update check.
# Repeat with bshd and another fresh EXP_NAME for the layout comparison.
# The dump must include routing-replay data from an R3-enabled online run.
# Reuse SAVE_DIR/EXP_NAME only to resume that experiment's checkpoint.
# DRY_RUN=1 prints argv/runtime env without starting Ray or importing ML packages.
# Direct bash uses local.sh; existing clusters should use ray-job.sh above.
# SKIP_SAVING=1 disables checkpoint loading/saving for fixed-rollout comparisons.
# MOE_DISPATCHER=alltoall and DETERMINISTIC_MODE=0 select the performance configuration.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
HF_CKPT="${HF_CKPT:-${MODEL_DIR}/DeepSeek-V4.1-4layer}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/gsm8k/train.parquet}"
QKV_FORMAT="${QKV_FORMAT:-thd}"
MOE_DISPATCHER="${MOE_DISPATCHER:-deepep}"
case "${QKV_FORMAT}" in
    bshd|thd) ;;
    *) echo "QKV_FORMAT must be bshd or thd" >&2; exit 1 ;;
esac
EXP_NAME="${EXP_NAME:-deepseek-v41-4layer-gsm8k-${QKV_FORMAT}-tp2-pp2-cp2-ep4-deepep-gpu8}"
LOG_DIR="${LOG_DIR:-${SAVE_DIR}/${EXP_NAME}/log}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/deepseek-v41-gsm8k-logprob}"
USE_CLEARML="${USE_CLEARML:-1}"
source "${RELAX_ROOT}/scripts/models/deepseek-v4.1-4layer.sh"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load "${HF_CKPT}"
    --megatron-to-hf-mode bridge
    --dump-details "${SAVE_DIR}/${EXP_NAME}/debug"
)
if [ "${SKIP_SAVING:-0}" != 1 ]; then
    CKPT_ARGS+=(
        --load "${SAVE_DIR}/${EXP_NAME}/checkpoints"
        --save "${SAVE_DIR}/${EXP_NAME}/checkpoints"
        --save-interval "${SAVE_INTERVAL:-2}"
        --max-actor-ckpt-to-keep 2
    )
fi
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key messages
    --label-key label
    --apply-chat-template
    --apply-chat-template-kwargs '{}'
    --rollout-shuffle
    --rollout-seed 42
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-prompt-len 3840
    --rollout-max-response-len 256
    --rollout-max-context-len 4096
    --rollout-temperature 0.8
    --rollout-top-p 1
    --rollout-top-k -1
    --global-batch-size "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))"
    --num-steps-per-rollout 1
    --num-iters-per-train-update 1
    --balance-data
    --use-rollout-routing-replay
)
PERF_ARGS=(
    --qkv-format "${QKV_FORMAT}"
    --seq-length 4096
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --context-parallel-size 2
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --allgather-cp
    # At TP2/CP2: BSHD aligns each sample to 256 tokens; THD aligns the
    # complete packed stream to 128 tokens without padding between samples.
    --data-pad-size-multiplier 32
    --calculate-per-token-loss
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --micro-batch-size 1
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --log-probs-max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --recompute-loss-function
    --empty-unused-memory-level 1
)
case "${MOE_DISPATCHER}" in
    deepep) PERF_ARGS+=(--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep) ;;
    alltoall) PERF_ARGS+=(--moe-token-dispatcher-type alltoall) ;;
    *) echo "MOE_DISPATCHER must be deepep or alltoall" >&2; exit 1 ;;
esac
if [ "${QKV_FORMAT}" = thd ]; then
    PERF_ARGS+=(
        --use-dynamic-batch-size
        --cp-partition-mode contiguous
    )
fi

GRPO_ARGS=(
    --advantage-estimator grpo
    --loss-type policy_loss
    --kl-coef 0.00
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
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
    --no-pin-cpu-grads
    --no-pin-cpu-params
)
SGLANG_ARGS=(
    --sglang-external-model-package relax.backends.sglang.deepseekv41
    # One BF16 TP8 / attention-DP1 / EP8 engine, matching the four-layer baseline.
    --rollout-num-gpus-per-engine 8
    --sglang-tp-size 8
    --sglang-dp-size 1
    --sglang-ep-size 8
    --sglang-load-format auto
    --sglang-dtype bfloat16
    --sglang-context-length 4096
    --sglang-attention-backend dsv4
    --sglang-moe-runner-backend auto
    --sglang-mem-fraction-static 0.6
    --sglang-max-running-requests 128
    --sglang-disable-cuda-graph
    --sglang-disable-radix-cache
    --sglang-watchdog-timeout 3600
)
METRICS_ARGS=()
if [ "${USE_CLEARML}" = 1 ]; then
    METRICS_ARGS+=(
        --use-clearml
        --use-metrics-service
        --tb-project-name "${PROJECT_NAME}"
        --tb-experiment-name "${EXP_NAME}-${now}"
    )
fi
MISC_ARGS=(
    --bf16
    --transformer-impl transformer_engine
    --seed 1234
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --trust-remote-code
    --use-health-check
    --update-weight-buffer-size 1073741824
    --train-memory-margin-bytes 1073741824
)
if [ "${DETERMINISTIC_MODE:-1}" = 1 ]; then
    MISC_ARGS+=(--deterministic-mode)
fi
TRAIN_ARGS=(
    --resource '{"actor": [1, 8], "rollout": [1, 8]}'
    --num-gpus-per-node 8
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
    --offload-train
    --offload-rollout
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "$@"
)

if [ "${DRY_RUN:-0}" != 1 ] && [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
runtime.setdefault("env_vars", {}).update({
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_CUMEM_ENABLE": "1",
    "NCCL_NVLS_ENABLE": "0",
    # Bound host staging for full CPU-optimizer synchronous checkpoints.
    "MEGATRON_SYNC_SAVE_BOUNDED_STAGING": os.environ.get("MEGATRON_SYNC_SAVE_BOUNDED_STAGING", "1"),
    "MEGATRON_SYNC_SAVE_STAGE_BYTES": os.environ.get("MEGATRON_SYNC_SAVE_STAGE_BYTES", "0"),
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE": "1",
    "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT": "shared",
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_DISABLE_MULTIMEM_AG": "1",
    "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
    "SGLANG_HEALTH_CHECK_TIMEOUT": "900",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
})
if os.environ.get("DETERMINISTIC_MODE", "1") == "1":
    runtime["env_vars"].update({
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })
for name in ("CUDA_HOME", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "CLEARML_CONFIG_FILE", "NO_VCS_VERSION"):
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
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-${RAY_DASHBOARD_ADDRESS:-${RAY_ADDRESS:-}}}"
if [ -z "${RAY_JOB_ADDRESS}" ] && [ "${RELAX_ENTRYPOINT_MODE:-}" = local ]; then
    RAY_JOB_ADDRESS="http://127.0.0.1:8265"
fi
if [[ "${RAY_JOB_ADDRESS}" != http://* && "${RAY_JOB_ADDRESS}" != https://* ]]; then
    echo "Set RAY_JOB_ADDRESS or RAY_DASHBOARD_ADDRESS to the Ray Jobs HTTP address, not GCS." >&2
    exit 1
fi
mkdir -p "${LOG_DIR}"
ray job submit \
    ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_JOB_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/${EXP_NAME}-${now}.log"
