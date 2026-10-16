#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full Kimi K3 text-only full-parameter SFT on HellaSwag, with a held-out
# validation set. 16 x 8 B300 GPUs, BF16 from the MXFP4 checkpoint.
# TP4/PP4/CP8/EP32/ETP1: DP1, expert-DP1; ordinary 1F1B, no VPP or MTP.
# PP layer counts: 23 + 24 + 24 + 22 = 93.
#
# Hyperparameters follow the official recipe k3_hellaswag.yaml (NVIDIA
# nemo_automodel, 256 GPU FSDP2). Mapped 1:1 where the knob exists in Megatron:
#   global_batch_size 256 | max_steps 100 | num_epochs 1 | val_every_steps 100
#   seed 1234 | lr 1e-5 | weight_decay 0.0 | betas [0.9, 0.95] | clip_grad 1.0
#   dataloader shuffle: true -> --rollout-shuffle
#   adam eps 1e-8 is Megatron's default, so it is not passed explicitly.
# With 25600 prepared train rows and GBS 256, the default 100 steps cover
# one epoch. Supply train.jsonl and a held-out validation.jsonl.
#
# PP4 leaves 32 GPUs per pipeline stage; EP32 assigns 28 routed experts/rank.
# This recipe defaults to CP8 and DP1. CP_SIZE can override context
# parallelism; revalidate memory use and throughput after changing it.
# MAX_TOKENS_PER_GPU defaults to 16384. Dynamic batching and padding determine
# the actual microbatch count; inspect packing metrics when tuning PP utilization.
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
#       examples/models/kimi-k3/scripts/run-kimi-k3-hellaswag-128xb300.sh
# Inspect arguments without touching Ray:
#   DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-hellaswag-128xb300.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/kimi-k3.sh"

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:-}}"
: "${HF_CHECKPOINT:?Set MODEL_DIR or HF_CHECKPOINT to the shared HF model directory}"
HELLASWAG_DATA_DIR="${HELLASWAG_DATA_DIR:-${DATA_DIR:-}}"
: "${HELLASWAG_DATA_DIR:?Set DATA_DIR or HELLASWAG_DATA_DIR to the shared dataset directory}"
# Rows are {"messages": [{"role": ..., "content": ...}], "source_id": ...} --
# already the OpenAI shape SFT expects, so no --conversation-key-map is needed.
PROMPT_DATA="['${HELLASWAG_DATA_DIR}/train.jsonl']"
EVAL_DATA="${HELLASWAG_DATA_DIR}/validation.jsonl"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/kimi-k3-hellaswag}"
EXP_NAME="${EXP_NAME:-kimi-k3-full-sft-hellaswag-b300-gpu128-bridge}"
# 25600 train rows / GBS 256 = 100 steps = exactly one epoch.
NUM_STEPS="${NUM_STEPS:-100}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
now=$(date "+%Y-%m-%d-%H-%M-%S")

# DeepEP segfaults on this cluster's EFA fabric; NVSHMEM's libfabric transport is
# the supported path when deepep is opted back in via USE_DEEPEP=1. Harmless with
# the default all-to-all dispatcher (which never touches NVSHMEM).
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-libfabric}"

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --trust-remote-code
    --ref-load "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    --num-rollout "${NUM_STEPS}"
    --save-interval "${SAVE_INTERVAL:-50}"
    # num_epochs: 1 -- one pass over the 25600 rows, no repetition.
    # --num-epoch "${NUM_EPOCH:-1}"
)
# Distributed checkpoints of the full model are very large; opt in with SAVE_DIR.
if [ -n "${SAVE_DIR:-}" ]; then
    CKPT_ARGS+=(--save "${SAVE_DIR}/${EXP_NAME}" --load "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-100}" --max-actor-ckpt-to-keep 1)
fi

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    # Text-only: no --multimodal-keys, no --conversation-key-map.
    --input-key messages
    --global-batch-size "${GLOBAL_BATCH_SIZE:-256}"
    # dataloader.shuffle: true in the recipe.
    --rollout-shuffle
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --sft-oversize-strategy skip
    --balance-data
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-prefetch-buffer-size 256
    --sft-tq-timeout-minutes 60
)

# Held-out validation set (recipe: validation split, 256 rows, val_every_steps 100).
# --eval-prompt-data takes "<name> <path>" and is mutually exclusive with
# --eval-size; the eval dataset reuses --input-key and the train key handling.
EVAL_ARGS=(
    # --skip-eval-before-train
    --eval-prompt-data hellaswag "${EVAL_DATA}"
    --eval-interval "${EVAL_INTERVAL:-20}"
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

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 4
    --decoder-first-pipeline-num-layers 23
    --decoder-last-pipeline-num-layers 22
    # Default CP8; override with CP_SIZE for the target context length.
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
    --train-env-vars "{\"FLA_TILELANG\":\"${FLA_TILELANG:-0}\"}"
    --resource '{"sft": [1, 0], "actor": [1, 128]}'
    --sft-max-in-flight-steps 1
    --num-data-storage-units 16
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${EVAL_ARGS[@]}"
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

mkdir -p "${RELAX_ROOT}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${RELAX_ROOT}/log/${EXP_NAME}-${now}.log"
