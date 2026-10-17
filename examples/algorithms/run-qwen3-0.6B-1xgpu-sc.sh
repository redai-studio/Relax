#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-0.6B / GSM8K Score Centering, one physical GPU in colocate mode.
# Requires a dedicated Ray cluster and the standard Relax training environment.
# MODEL_DIR/Qwen3-0.6B contains the HF model; DATA_DIR/gsm8k/main contains
# the original train/test-00000-of-00001.parquet files from openai/gsm8k.
# Normalized data is cached under DATA_DIR/gsm8k/sc_boxed_v1.
#
# Launch from the repository root (set the three directory variables first):
#   RAY_ADDRESS=<head>:6379 RAY_DASHBOARD_ADDRESS=http://<head>:8265 \
#     bash scripts/entrypoint/ray-job.sh \
#       examples/algorithms/run-qwen3-0.6B-1xgpu-sc.sh
# Smoke: NUM_ROLLOUT=5 SAVE_INTERVAL=2 EVAL_INTERVAL=2 EVAL_NUM_PROMPTS=16
# Resume: keep SAVE_DIR, EXP_NAME, and the training settings unchanged.
# To extend NUM_ROLLOUT, also pass --use-checkpoint-opt_param-scheduler
# to retain the saved optimizer schedule (the default LR is constant).
# A new experiment must use a different EXP_NAME or SAVE_DIR.
# EVAL_NUM_PROMPTS=0 evaluates the full test split; the default is a smoke subset.
# Defaults: 4 prompts x 8 responses = 32 trajectories / optimizer update.
# ClearML is enabled by default; USE_CLEARML=0 keeps local metrics only.
# SC_IS=none|tis|mis selects internal weighting; do not add --use-tis or OPD.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the dedicated Ray Jobs HTTP address}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/algorithms/score-centering}"
EXP_NAME="${EXP_NAME:-qwen3-0.6b-sc-gsm8k-1xgpu-colocate}"
now=$(date "+%Y%m%d-%H%M%S")
NUM_ROLLOUT="${NUM_ROLLOUT:-60}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES="${N_SAMPLES:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
EVAL_NUM_PROMPTS="${EVAL_NUM_PROMPTS:-32}"
SEED="${SEED:-42}"

# Use the same answer normalization and boxed-output prompt for both splits.
# Keep evaluation separate from the training split, including in smoke runs.
DATA_DIR="${DATA_DIR}" EVAL_NUM_PROMPTS="${EVAL_NUM_PROMPTS}" python3 - <<'PY'
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["DATA_DIR"]) / "gsm8k"
cache = root / "sc_boxed_v1"
cache.mkdir(parents=True, exist_ok=True)
eval_count = int(os.environ["EVAL_NUM_PROMPTS"])
if eval_count < 0:
    raise ValueError("EVAL_NUM_PROMPTS must be nonnegative (0 means the full test split)")
for split in ("train", "test"):
    name = "train.parquet" if split == "train" else f"test_{eval_count}.parquet"
    target = cache / name
    if target.is_file():
        continue
    frame = pd.read_parquet(root / "main" / f"{split}-00000-of-00001.parquet")
    if split == "test" and eval_count:
        frame = frame.head(eval_count).copy()
    frame["question"] = frame["question"].astype(str).str.rstrip() + (
        "\n\nShow your reasoning and put the final answer in \\boxed{...}."
    )
    frame["answer"] = frame["answer"].str.split("####").str[-1].str.strip()
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)
PY

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B"
    --ref-load "${MODEL_DIR}/Qwen3-0.6B"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-10}"
    --max-actor-ckpt-to-keep 2
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/gsm8k/sc_boxed_v1/train.parquet"
    --input-key question
    --label-key answer
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --reward-num-workers 1
    --reward-max-concurrency 2
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rollout-top-k -1
    --rollout-seed "${SEED}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
    --use-fault-tolerance
)

EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-10}"
    --eval-prompt-data gsm8k "${DATA_DIR}/gsm8k/sc_boxed_v1/test_${EVAL_NUM_PROMPTS}.parquet"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --eval-temperature 1.0
    --eval-top-p 1.0
    --eval-top-k -1
)
if [ "${SKIP_EVAL_BEFORE_TRAIN:-1}" = 1 ]; then
    EVAL_ARGS+=(--skip-eval-before-train)
fi
if [ "${LOG_PASSRATE:-1}" = 1 ]; then
    EVAL_ARGS+=(--log-passrate)
fi

SC_ARGS=(
    --loss-type policy_loss
    --advantage-estimator score_centering
    --rollout-top-logprobs-num "${ROLLOUT_TOP_LOGPROBS_NUM:-128}"
    --score-centering-is "${SC_IS:-none}"
    --kl-coef 0
    --entropy-coef 0
    --use-rollout-logprobs
    --clip-grad 1.0
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
    --log-probs-max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
    --log-probs-chunk-size 4
    --update-weight-buffer-size 134217728
    --train-memory-margin-bytes 536870912
    --disable-weights-backuper
    --recompute-loss-function
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.45}"
)

LOGGING_ARGS=(
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
case "${USE_CLEARML:-1}" in
    1) LOGGING_ARGS+=(--use-clearml) ;;
    0) ;;
    *) echo "USE_CLEARML must be 0 or 1" >&2; exit 2 ;;
esac

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend fused
    --seed "${SEED}"
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD_ADDRESS}" \
    ${RAY_SUBMISSION_ID:+--submission-id "${RAY_SUBMISSION_ID}"} \
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
    "${EVAL_ARGS[@]}" \
    "${SC_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "$@" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
