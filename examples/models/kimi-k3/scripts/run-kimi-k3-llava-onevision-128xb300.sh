#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full Kimi K3 multimodal SFT on LLaVA-OneVision-1.5-Instruct-Data.
# 16 x 8 B300 GPUs, BF16 from the MXFP4 checkpoint. This uses the Kimi K3
# chat rendering (K3 XTML encoder) and vision processor.
# TP4/PP4/CP8/EP32/ETP1: DP1, expert-DP1; ordinary 1F1B, no VPP or MTP.
# PP layer counts: 21 + 24 + 24 + 24 = 93; stage 0 also hosts the vision tower.
#
# One million cached-source samples, streaming JSONL with embedded image data URIs.
# Prepare from the downloaded cache (no network):
#   python -m scripts.tools.prepare_llavaonevision sample \
#       --data-dir /shared/data/onevision --count 1000000
# Submit using scripts/entrypoint/ray-job.sh; DRY_RUN=1 only prints arguments.
# Default: 1000 steps, all assistant turns, frozen vision tower.
# Save every 500 steps; retain one checkpoint without optimizer state.
# Save validation: NUM_STEPS=2 SAVE_INTERVAL=2 stops after two training steps
# and saves iteration 1 (training step indices are zero-based).

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/kimi-k3.sh"

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:-}}"
: "${HF_CHECKPOINT:?Set MODEL_DIR or HF_CHECKPOINT to the shared HF model directory}"
LLAVA_DATA_DIR="${LLAVA_DATA_DIR:-${DATA_DIR:-}}"
: "${LLAVA_DATA_DIR:?Set DATA_DIR or LLAVA_DATA_DIR to the prepared dataset directory}"
PROMPT_DATA="${PROMPT_DATA:-${LLAVA_DATA_DIR}/train}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/kimi-k3-llava-onevision}"
EXP_NAME="${EXP_NAME:-kimi-k3-full-sft-llava-onevision-1m-b300-gpu128-chat}"
if [ "${DRY_RUN:-0}" != 1 ] && [ ! -f "${LLAVA_DATA_DIR}/READY.json" ]; then
    echo "Data preparation is incomplete: ${LLAVA_DATA_DIR}/READY.json is missing." >&2
    exit 1
fi
NUM_STEPS="${NUM_STEPS:-1000}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-1}"
now=$(date "+%Y-%m-%d-%H-%M-%S")

# The source HellaSwag script enabled text-only rendering. Clear any inherited
# setting so SFT uses Kimi K3's multimodal chat renderer for training.
unset RELAX_SFT_RAW_TEXT_CONCAT

# DeepEP segfaults on this cluster's EFA fabric; NVSHMEM's libfabric transport is
# the supported path when deepep is opted back in via USE_DEEPEP=1. Harmless with
# the default all-to-all dispatcher (which never touches NVSHMEM).
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-libfabric}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --trust-remote-code
    --ref-load "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    --num-epoch "${NUM_EPOCH:-3}"
    --save-interval "${SAVE_INTERVAL:-500}"
    --max-actor-ckpt-to-keep 1
)
# Leave epoch-based training unchanged unless an explicit step limit is set.
if [ -n "${NUM_STEPS}" ]; then
    CKPT_ARGS+=(--num-rollout "${NUM_STEPS}")
fi
# Enable checkpoint saving with a stable shared directory.
if [ -n "${SAVE_DIR:-}" ]; then
    CKPT_ARGS+=(--save "${SAVE_DIR}/${EXP_NAME}" --load "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-500}")
fi
# Allow an explicit checkpoint to override the experiment's default resume path.
if [ -n "${LOAD_DIR:-}" ]; then
    CKPT_ARGS+=(--load "${LOAD_DIR}")
fi

# Long-context full-model runs save weights only unless explicitly overridden.
if [ "${SAVE_OPTIMIZER:-0}" != 1 ]; then
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
    VISION_ARGS=(--vision-dp-when-tp --freeze-vision-model)
fi

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    "${VISION_ARGS[@]}"
    --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 21
    --decoder-last-pipeline-num-layers 24
    # Preserve the K3 128-GPU layout used by the source script.
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

mkdir -p "${RELAX_ROOT}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RELAX_ROOT}/log/${EXP_NAME}-${now}.log"
