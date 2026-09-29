#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full Kimi K3 multimodal full-parameter SFT: 93 language layers, 896 experts,
# full vision. 16 x 8 B300 GPUs (275040 MiB each), BF16 from the MXFP4 checkpoint.
# TP4/PP4/CP4/EP32/ETP1: DP2, expert-DP1; ordinary 1F1B, no VPP or MTP.
# PP layer counts: 23 + 24 + 24 + 22 = 93.
#
# Scaled from the validated 64-GPU LoRA run (TP4/PP4/CP4/EP16/DP1, 200/200 steps,
# ~16 s/step, grad_norm flat): the extra 64 GPUs go into EP and DP, so
# EP16 -> EP32 halves the per-rank expert weights to 896/32 = 28 experts/rank
# (same per-rank expert count as the 192-GPU script) while CP4 and the 32768-token
# packing budget stay exactly as validated.
#
# HOST RAM — this is the binding constraint for full-parameter SFT, not HBM.
# Every parameter carries an FP32 master copy (4 B) + Adam m/v (8 B) + BF16 grad
# (2 B) ~= 14 B, so 2.78T params ~= 39 TB of offloaded optimizer state, i.e.
# ~2.4 TB on each of the 16 nodes. That fits under a ~4 TB/node budget but without
# much headroom: if a node OOMs during load or step 0, lower
# OPTIMIZER_OFFLOAD_FRACTION (0.5 keeps half the optimizer state/work on the GPUs)
# or run on more nodes. LoRA instead keeps this cost negligible — see
# examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-lora-64xb300.sh.
#
# Dispatcher: defaults to the plain all-to-all dispatcher, NOT DeepEP. DeepEP's
# NVSHMEM init segfaults on this cluster's AWS EFA fabric once GDRCopy is
# unavailable. Set USE_DEEPEP=1 to opt back into deepep+flex where supported.
#
# Model load takes ~7-15 min (2.78T params, MXFP4->BF16 dequant, CPU-offloaded);
# a job sitting in "loading" is not a hang — the first `training step 0/200` log
# line is the real start signal. Step 0 itself is also slow (compile + NCCL init).
#
# Submit on the head of an existing 128-GPU Ray cluster:
#   bash scripts/entrypoint/ray-job.sh \
#       examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-128xb300.sh
# Inspect arguments without touching Ray:
#   DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-128xb300.sh

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_DIR}/kimi-k3.sh"

HF_CHECKPOINT="${HF_CHECKPOINT:-${MODEL_DIR:-}}"
: "${HF_CHECKPOINT:?Set MODEL_DIR or HF_CHECKPOINT to the shared HF model directory}"
POKEMON_DATA_DIR="${POKEMON_DATA_DIR:-${DATA_DIR:-}}"
: "${POKEMON_DATA_DIR:?Set DATA_DIR or POKEMON_DATA_DIR to the shared dataset directory}"
PROMPT_DATA="['${POKEMON_DATA_DIR}/pokemon_gpt4o_zh.parquet']"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/kimi-k3-pokemon}"
EXP_NAME="${EXP_NAME:-kimi-k3-full-sft-pokemon-zh-b300-gpu128-bridge}"
NUM_STEPS="${NUM_STEPS:-200}"
# 8192 * CP4 = 32768-token packing budget, as validated on the 64-GPU run.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
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
    # The 833-example dataset repeats; one epoch would end after only a few steps.
    --num-rollout "${NUM_STEPS}"
)
# Distributed checkpoints of the full model are very large; opt in with SAVE_DIR.
if [ -n "${SAVE_DIR:-}" ]; then
    CKPT_ARGS+=(--save "${SAVE_DIR}/${EXP_NAME}" --load "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-100}" --max-actor-ckpt-to-keep 1)
fi

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key conversations
    --multimodal-keys '{"image":"images"}'
    --conversation-key-map '{"from":"role","value":"content","human":"user","gpt":"assistant"}'
    # DP2: half the global batch per replica.
    --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
    --use-dynamic-batch-size
    # CP4 gives a MAX_TOKENS_PER_GPU*4 = 32768-token packing budget.
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --sft-oversize-strategy skip
    --balance-data
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-prefetch-buffer-size 256
    --sft-tq-timeout-minutes 60
)

# MoE token dispatcher. Default = plain all-to-all (no NVSHMEM, EFA-safe).
# USE_DEEPEP=1 restores the deepep+flex path (only safe on a fabric where
# DeepEP's NVSHMEM init does not segfault).
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
    --context-parallel-size "${CP_SIZE:-4}"
    --calculate-per-token-loss
    # 32 GPUs per PP stage (TP4*CP4*DP2) -> EP32, 896/32 = 28 experts/rank.
    --expert-model-parallel-size "${EP_SIZE:-32}"
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    # ~14 B/param of FP32 master + Adam m/v + BF16 grad over 2.78T params is
    # ~39 TB, i.e. ~2.4 TB per node across 16 nodes. Drop to 0.5 to move half the
    # optimizer state/work onto the GPUs if host RAM runs short.
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"

    --disable-weights-backuper
    --cross-entropy-loss-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-256}"

    "${DISPATCHER_ARGS[@]}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --lr-decay-style cosine
    --min-lr 1e-6
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
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
    # Pin RNG for reproducibility / parity with the 64-GPU run.
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
