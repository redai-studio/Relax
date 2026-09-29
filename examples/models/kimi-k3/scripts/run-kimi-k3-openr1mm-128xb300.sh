#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full Kimi K3 multimodal SFT on multimodal-open-r1-8k-verified.
# 16 x 8 B300 GPUs, BF16 from the MXFP4 checkpoint. This uses the Kimi K3
# chat rendering (K3 XTML encoder) and vision processor.
# TP4/PP4/CP8/EP32/ETP1: DP1, expert-DP1; ordinary 1F1B, no VPP or MTP.
# PP layer counts: 21 + 24 + 24 + 24 = 93; stage 0 also hosts the vision tower.
#
# Hyperparameters follow the official recipe k3_hellaswag.yaml (NVIDIA
# nemo_automodel, 256 GPU FSDP2). Mapped 1:1 where the knob exists in Megatron:
#   global_batch_size 256 | seed 1234 | lr 1e-5 | weight_decay 0.0
#   seed 1234 | lr 1e-5 | weight_decay 0.0 | betas [0.9, 0.95] | clip_grad 1.0
#   dataloader shuffle: true -> --rollout-shuffle
#   adam eps 1e-8 is Megatron's default, so it is not passed explicitly.
# The converted dataset has 7424 train rows and 265 held-out test rows.
# 7424 / GBS 256 = 29 steps = exactly one pass over training data.
#
# Parallelism is NOT copied from the recipe: that is a 256-GPU FSDP2 layout
# (tp1/cp1/pp8/ep32/dp32). On 128 GPUs under Megatron, PP4 is what still yields
# 32 GPUs per pipeline stage and therefore EP32 -> 896/32 = 28 experts/rank,
# which is the lever that fits the 2.72T of routed experts. CP8 and a 16384
# token/GPU budget accommodate visual tokens and long reasoning responses.
#
# Dynamic batching uses the token budget below for image and reasoning samples.
#
# HOST RAM is the binding constraint for full-parameter SFT: FP32 master (4 B) +
# Adam m/v (8 B) + BF16 grad (2 B) ~= 14 B/param over 2.78T params ~= 39 TB,
# i.e. ~2.4 TB on each of the 16 nodes. If a node OOMs at load or step 0, lower
# OPTIMIZER_OFFLOAD_FRACTION (0.5 keeps half on the GPUs) or add nodes.
#
# Dispatcher defaults to plain all-to-all, NOT DeepEP: DeepEP's NVSHMEM init
# segfaults on this cluster's AWS EFA fabric. USE_DEEPEP=1 opts back in.
#
# Model load takes ~7-15 min (2.78T params, MXFP4->BF16 dequant); a job sitting
# in "loading" is not a hang. Step 0 is also slow (compile + NCCL init).
#
# Submit on the head of an existing 128-GPU Ray cluster:
#   bash scripts/entrypoint/ray-job.sh \
#       examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300.sh
# Inspect arguments without touching Ray:
#   DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-openr1mm-128xb300.sh
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
OPENR1MM_DATA_DIR="${OPENR1MM_DATA_DIR:-${DATA_DIR:-}}"
: "${OPENR1MM_DATA_DIR:?Set DATA_DIR or OPENR1MM_DATA_DIR to the shared dataset directory}"
PROMPT_DATA="['${OPENR1MM_DATA_DIR}/train.parquet']"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/kimi-k3-openr1mm}"
EXP_NAME="${EXP_NAME:-kimi-k3-full-sft-openr1mm-b300-gpu128-chat}"
NUM_STEPS="${NUM_STEPS:-}"
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
    --save-interval "${SAVE_INTERVAL:-100}"
    # 29 steps at GBS 256 consume the complete training split once.
    --num-epoch "${NUM_EPOCH:-3}"
)
# Leave epoch-based training unchanged unless an explicit step limit is set.
if [ -n "${NUM_STEPS}" ]; then
    CKPT_ARGS+=(--num-rollout "${NUM_STEPS}")
fi
# Distributed checkpoints of the full model are very large; opt in with SAVE_DIR.
if [ -n "${SAVE_DIR:-}" ]; then
    CKPT_ARGS+=(--save "${SAVE_DIR}/${EXP_NAME}" --load "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-100}" --max-actor-ckpt-to-keep 1)
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
