#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Full Kimi K3 multimodal SFT: 93 language layers, 896 experts, full vision.
# 24 x 8 B300 GPUs (275040 MiB each), BF16 training from the MXFP4 checkpoint.
# TP4/PP6/CP2/EP32/ETP1: DP4, expert-DP1; ordinary 1F1B, no VPP or MTP.
# PP layer counts: 15 + 16 + 16 + 16 + 16 + 14 = 93.
# Requires the upgraded Blackwell image with the K3 CP/recompute patches.
#
# Submit on the head of an existing 192-GPU Ray cluster:
#   bash scripts/entrypoint/ray-job.sh \
#       examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-192xgpu-b300.sh
# Inspect arguments without touching Ray:
#   DRY_RUN=1 bash examples/models/kimi-k3/scripts/run-kimi-k3-pokemon-192xgpu-b300.sh

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
EXP_NAME="${EXP_NAME:-kimi-k3-full-sft-pokemon-zh-b300-gpu192-bridge}"
NUM_STEPS="${NUM_STEPS:-200}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-512}"
now=$(date "+%Y-%m-%d-%H-%M-%S")

CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --trust-remote-code
    --ref-load "${HF_CHECKPOINT}"
    --megatron-to-hf-mode bridge
    # The 833-example dataset repeats; one epoch would end after only four steps.
    --num-rollout "${NUM_STEPS}"
    --num-epoch 1
)
# Distributed checkpoints are large; opt in with a shared SAVE_DIR.
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
    # DP4: 48 examples per replica, enough packed microbatches to fill PP6.
    --global-batch-size "${GLOBAL_BATCH_SIZE:-192}"
    --use-dynamic-batch-size
    # CP2 gives a 1024-token packing budget, as in the reduced-model run.
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --sft-oversize-strategy skip
    --balance-data
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-prefetch-buffer-size 256
    --sft-tq-timeout-minutes 60
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 6
    --decoder-first-pipeline-num-layers 15
    --decoder-last-pipeline-num-layers 14
    --context-parallel-size 2
    --calculate-per-token-loss
    --expert-model-parallel-size 32
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
    # Full offload needs >1764 GiB host RAM per busiest node including experts'
    # FP32 master weights, Adam states and gradients, before other runtime costs.
    # With tighter host RAM, 0.5 moves half of the optimizer work/state to GPUs.
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-1.0}"

    --disable-weights-backuper
    --cross-entropy-loss-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-256}"

    --moe-flex-dispatcher-backend deepep
    --moe-token-dispatcher-type flex
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
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
)

TRAIN_ARGS=(
    --train-env-vars "{\"FLA_TILELANG\":\"${FLA_TILELANG:-0}\"}"
    --resource '{"sft": [1, 0], "actor": [1, 192]}'
    --sft-max-in-flight-steps 1
    --num-data-storage-units 24
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
