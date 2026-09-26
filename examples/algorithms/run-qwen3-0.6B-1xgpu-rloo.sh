#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B 1xGPU RLOO/GRPO A-B recipe — GSM8K train only (no eval).
#
# Colocate mode: actor and rollout time-share the same GPU.
# Switch between RLOO and GRPO via ADVANTAGE_ESTIMATOR env var for
# identical-condition A/B comparison.
#
# RLOO: leave-one-out baseline, unclipped REINFORCE loss (sync only).
#   --eps-clip is ineffective for rloo (train/pg_clipfrac is always 0).
#   Global valid-token normalization and max-staleness=0 are required.
#   Reward-side --kl-coef is unsupported; this recipe pins it to 0.
#
# train_iters = NUM_ROLLOUT × ROLLOUT_BATCH_SIZE × N_SAMPLES / GLOBAL_BATCH_SIZE
# RLOO requires ROLLOUT_BATCH_SIZE × N_SAMPLES == GLOBAL_BATCH_SIZE, so each
# rollout is consumed by exactly one optimizer update. The three batch values
# may be reduced together to fit the available hardware while preserving this
# equality (for example, 4 × 8 == 32).
# Formal experiment: 60 × 16 × 8 / 128 = 60 optimizer steps
# Smoke test:        5 × 1 × 4 / 4 = 5 optimizer steps
#
# Dataset: openai/gsm8k (7473 problems)
#   On first run the script writes a versioned, normalized copy to the writable
#   RLOO_DATA_CACHE_DIR: it keeps the answer after #### and appends an explicit
#   `\boxed{...}` output instruction required by the math reward parser.
#
# Usage:
#   # RLOO (default)
#   bash examples/algorithms/run-qwen3-0.6B-1xgpu-rloo.sh
#   # GRPO control arm
#   ADVANTAGE_ESTIMATOR=grpo bash examples/algorithms/run-qwen3-0.6B-1xgpu-rloo.sh
#   # Smoke test
#   ADVANTAGE_ESTIMATOR=rloo NUM_ROLLOUT=5 ROLLOUT_BATCH_SIZE=1 N_SAMPLES=4 GLOBAL_BATCH_SIZE=4 \
#     bash examples/algorithms/run-qwen3-0.6B-1xgpu-rloo.sh
#   # Pass extra args (e.g. seed, experiment name)
#   bash examples/algorithms/run-qwen3-0.6B-1xgpu-rloo.sh --rollout-seed 42 \
#     --tb-experiment-name task28-b1-rloo-s42
#
# Key overridable env vars:
#   ADVANTAGE_ESTIMATOR - rloo (default) or grpo
#   MODEL_DIR            - dir containing Qwen3-0.6B/
#   DATA_DIR             - dir containing gsm8k/main/train-00000-of-00001.parquet
#   NUM_ROLLOUT          - number of rollout/update steps (default: 60)
#   ROLLOUT_BATCH_SIZE   - prompts per rollout (default: 16)
#   N_SAMPLES            - responses per prompt (default: 8)
#   GLOBAL_BATCH_SIZE    - trajectories per update (default: 128)
#   RLOO_DATA_CACHE_DIR  - writable cache for normalized GSM8K data
#
# This is an A/B experiment recipe and does not configure periodic checkpoints.
# For recoverable long runs, pass --save and --save-interval explicitly.
#
# NOTE: --clip-grad 0 is set for both arms to ensure fair comparison.
#   GRPO divides by group std; RLOO multiplies by G/(G-1). The two produce
#   different advantage scales and thus different gradient-norm distributions.
#   The default --clip-grad 1.0 would clip the two arms at different rates,
#   confounding the algorithm comparison. Stability is monitored via
#   train/grad_norm and train/entropy_loss instead.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/issue-186-rloo}"
MODEL_DIR="${MODEL_DIR:-/models}"
DATA_DIR="${DATA_DIR:-/data}"

ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-rloo}"

GSM8K_RAW="${DATA_DIR}/gsm8k/main/train-00000-of-00001.parquet"
RLOO_DATA_CACHE_DIR="${RLOO_DATA_CACHE_DIR:-${RELAX_ARTIFACTS:-${RELAX_ROOT}/artifacts}/data-cache/gsm8k}"
GSM8K_CLEAN="${RLOO_DATA_CACHE_DIR}/train_rloo_boxed_v1.parquet"
mkdir -p "${RLOO_DATA_CACHE_DIR}"
if [ ! -f "${GSM8K_CLEAN}" ]; then
    GSM8K_RAW="${GSM8K_RAW}" GSM8K_CLEAN="${GSM8K_CLEAN}" python3 - <<'PY'
import os
from pathlib import Path

import pandas as pd

source = Path(os.environ["GSM8K_RAW"])
target = Path(os.environ["GSM8K_CLEAN"])
temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")

df = pd.read_parquet(source)
df["question"] = df["question"].astype(str).str.rstrip() + (
    "\n\nShow your reasoning and put the final answer in \\boxed{...}."
)
df["answer"] = df["answer"].str.split("####").str[-1].str.strip()
df.to_parquet(temporary, index=False)
os.replace(temporary, target)
print(f"Wrote {len(df)} rows to {target}")
PY
fi

NUM_ROLLOUT="${NUM_ROLLOUT:=60}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:=16}"
N_SAMPLES="${N_SAMPLES:=8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:=128}"
# train_iters = 60 × 16 × 8 / 128 = 60

CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3-0.6B
    --ref-load ${MODEL_DIR}/Qwen3-0.6B
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
    --prompt-data ${GSM8K_CLEAN}
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle

    --rm-type math
    --reward-num-workers 1
    --reward-max-concurrency 2

    --num-rollout ${NUM_ROLLOUT}
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
    --n-samples-per-prompt ${N_SAMPLES}
    --rollout-max-response-len 2048
    --rollout-temperature 1

    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --balance-data
    --use-fault-tolerance
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
    --log-probs-max-tokens-per-gpu 2048
    --log-probs-chunk-size 4

    --update-weight-buffer-size 134217728
    --train-memory-margin-bytes 536870912
    --disable-weights-backuper

    --recompute-loss-function
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

# Algorithm-specific args: RLOO uses unclipped REINFORCE (no --eps-clip);
# GRPO uses PPO-Clip with --eps-clip 0.2. Both arms use --clip-grad 0 for
# fair comparison (see script header comment).
ALGO_ARGS=(
    --advantage-estimator ${ADVANTAGE_ESTIMATOR}
    --kl-coef 0
    --entropy-coef 0.00
    --use-rollout-logprobs
    --clip-grad 0
)
if [ "${ADVANTAGE_ESTIMATOR}" = "grpo" ]; then
    ALGO_ARGS+=(--eps-clip 0.2)
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    # ~55% for training; SGLang uses 45%
    --sglang-mem-fraction-static 0.45
)

WANDB_ARGS=(
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name qwen3-0.6b-${ADVANTAGE_ESTIMATOR}-gsm8k-1xgpu-${now}
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

LOG_DIR="${RELAX_ARTIFACTS:-${RELAX_ROOT}/log}/logs"
mkdir -p "${LOG_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 1], "rollout": [1, 1]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/qwen3-0.6b-${ADVANTAGE_ESTIMATOR}-gsm8k-1xgpu-${now}.log"
