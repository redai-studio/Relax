#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full Kimi-K3.1-Pro SFT on multimodal-open-r1-8k-verified, based on the K3 recipe.
# GPU profile: 16 x 8 GPUs, 280G Blackwell per GPU.
# TP4/PP4/CP8/EP32/ETP1; PP layers 21/24/24/24, ordinary 1F1B, no VPP or MTP.
# Vision is frozen by default; FREEZE_VISION_TOWER=0 enables joint training.
#
# Set MODEL_DIR to the Kimi-K3.1-Pro HF directory, DATA_DIR to the converted
# dataset directory (train.parquet with messages/images), and SAVE_DIR to a
# persistent shared checkpoint root. PROJECT_NAME and EXP_NAME are overridable.
# Checkpoints use SAVE_DIR/EXP_NAME for both load and save, including optimizer
# state by default; save every 10 steps and retain one checkpoint. Keep these
# paths fixed to resume; change EXP_NAME for a new experiment.
#
# K3.1's HF config selects the local bridge and frozen NoPE indexer automatically.
# DSA uses Core's TileLang kernels; FLA_TILELANG independently controls KDA.
# SFT has no SGLang rollout. GPU/TP/CP numerics still need cluster validation.
# Full optimizer offload and checkpoints require substantial host RAM/storage;
# OPTIMIZER_OFFLOAD_FRACTION follows the K3 recipe and defaults to 1.0.
#
# Inspect arguments without touching Ray (all three directories must be set):
#   DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3.1-pro-openr1mm-128xblackwell.sh
# Submit on an existing 128-GPU Ray cluster, blocking by default:
#   export RAY_DASHBOARD_ADDRESS="http://<ray-head-host>:<dashboard-port>"
#   bash scripts/entrypoint/ray-job.sh \
#       examples/models/kimi-k3/scripts/run-kimi-k3.1-pro-openr1mm-128xblackwell.sh
# Short save check: NUM_STEPS=2 SAVE_INTERVAL=2 (zero-based iteration 1).

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/kimi-k3.1-pro.sh"

: "${MODEL_DIR:?Set MODEL_DIR to the shared Kimi-K3.1-Pro HF model directory}"
: "${DATA_DIR:?Set DATA_DIR to the shared OpenR1-MM dataset directory}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent shared checkpoint root}"
PROMPT_DATA="['${DATA_DIR}/train.parquet']"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/kimi-k3.1-pro-openr1mm}"
EXP_NAME="${EXP_NAME:-kimi-k3.1-pro-full-sft-openr1mm-blackwell-280g-gpu128-chat}"
NUM_STEPS="${NUM_STEPS:-}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-1}"
now=$(date "+%Y-%m-%d-%H-%M-%S")

# Clear inherited text-only rendering so SFT uses multimodal chat rendering.
unset RELAX_SFT_RAW_TEXT_CONCAT

# Preserve the K3 recipe transport when DeepEP is explicitly enabled.
# The default all-to-all dispatcher does not use NVSHMEM.
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-libfabric}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --trust-remote-code
    --ref-load "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-10}"
    --max-actor-ckpt-to-keep 1
    --num-epoch "${NUM_EPOCH:-3}"
)
# Leave epoch-based training unchanged unless an explicit step limit is set.
if [ -n "${NUM_STEPS}" ]; then
    CKPT_ARGS+=(--num-rollout "${NUM_STEPS}")
fi
# Save optimizer state by default so the stable load/save directory can resume.
# SAVE_OPTIMIZER=0 opts into model-only checkpoints, which cannot fully resume.
if [ "${SAVE_OPTIMIZER:-1}" != 1 ]; then
    CKPT_ARGS+=(--no-save-optim)
fi

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key messages
    --multimodal-keys '{"image":"images"}'
    --image-max-token-num 1024
    --global-batch-size "${GLOBAL_BATCH_SIZE:-512}"
    # dataloader.shuffle: true in the recipe.
    --rollout-shuffle
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --sft-oversize-strategy skip
    --sft-invalid-multimodal-strategy skip
    --balance-data
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-prefetch-buffer-size 256
    --sft-tq-timeout-minutes 60
)

# MoE token dispatcher. Default = plain all-to-all (no NVSHMEM, EFA-safe).
if [ "${USE_DEEPEP:-0}" = 1 ]; then
    DISPATCHER_ARGS=(
        --moe-flex-dispatcher-backend deepep
        --moe-token-dispatcher-type flex
    )
else
    DISPATCHER_ARGS=(
        --moe-token-dispatcher-type alltoall
    )
fi

# Trainable vision uses per-layer activation checkpointing; vision DP requires a frozen tower.
VISION_ARGS=()
if [ "${FREEZE_VISION_TOWER}" = 1 ]; then
    VISION_ARGS=(
        --vision-dp-when-tp
        --freeze-vision-model
    )
fi

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    "${VISION_ARGS[@]}"
    --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 21
    --decoder-last-pipeline-num-layers 24
    # Reuse the K3 128-GPU layout; validate K3.1 DSA on the target cluster.
    --context-parallel-size "${CP_SIZE:-8}"
    --calculate-per-token-loss
    # 32 GPUs per PP stage (TP4*CP8*DP1) -> EP32, 896/32 = 28 experts/rank.
    --expert-model-parallel-size "${EP_SIZE:-32}"
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"

    --disable-weights-backuper
    --cross-entropy-loss-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-256}"

    "${DISPATCHER_ARGS[@]}"
)

# Matches the recipe's AdamW block: lr 1e-5, weight_decay 0.0, betas [0.9, 0.95].
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --lr-decay-style cosine
    --min-lr 1e-6
    --weight-decay "${WEIGHT_DECAY:-0.0}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
)

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --use-tensorboard
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    # rng.seed: 1234 in the recipe.
    --seed "${SEED:-1234}"
)

TRAIN_ARGS=(
    --train-env-vars "{\"FLA_TILELANG\":\"${FLA_TILELANG:-0}\",\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\"}"
    --resource '{"sft": [1, 0], "actor": [1, 128]}'
    --sft-max-in-flight-steps 1
    --num-data-storage-units 16
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "$@"
)

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

: "${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the Ray Jobs HTTP address}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi

# Keep the raw-text rendering mode consistent in Ray workers.
if [ -n "${RUNTIME_ENV_JSON:-}" ]; then
    RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
env.pop("RELAX_SFT_RAW_TEXT_CONCAT", None)
propagate = env.get("RELAX_PROPAGATE_ENV_VARS", "").split(",")
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(
    dict.fromkeys(name for name in propagate if name and name != "RELAX_SFT_RAW_TEXT_CONCAT")
)
print(json.dumps(runtime))
PY
    )
    export RUNTIME_ENV_JSON
fi

mkdir -p "${SCRIPT_DIR}/log"
ray job submit \
    ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_DASHBOARD_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
