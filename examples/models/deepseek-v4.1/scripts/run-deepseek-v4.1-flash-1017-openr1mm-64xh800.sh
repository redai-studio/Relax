#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full 40-layer 1017 BF16 language model and frozen vision tower on 8 x 8 H800.
# DATA_DIR contains the stratified OpenR1 train and eval split. Train for 200
# updates with 16 prompts x 8 samples. Evaluate the base and then every 10 updates.
# bash scripts/entrypoint/ray-job.sh examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-openr1mm-64xh800.sh
set -eo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
split_dir="${DATA_DIR:?Set DATA_DIR to the dataset root}/multimodal-open-r1-8k-verified/splits/stratified-eval128-v1"
tracking_args=()
if [[ ${USE_CLEARML:-1} == 1 ]]; then
    tracking_args=(
        --use-clearml
        --use-metrics-service
        --tb-project-name "${PROJECT_NAME:-Relax/rl/openr1mm}"
        --tb-experiment-name "${EXP_NAME:-deepseek-v41-1017-40layer-openr1mm-grpo}-$(date +%Y%m%d-%H%M%S)"
    )
fi
NUM_ROLLOUT=${NUM_ROLLOUT:-200} \
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16} \
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8} \
SAVE_INTERVAL=${SAVE_INTERVAL:-50} \
DUMP_DETAILS=${DUMP_DETAILS:-0} \
PROMPT_DATA="${PROMPT_DATA:-${split_dir}/train.parquet}" \
exec bash "${script_dir}/run-deepseek-v4.1-flash-1017-openr1mm.sh" 40 \
    "${tracking_args[@]}" \
    --eval-prompt-data openr1_holdout "${EVAL_DATA:-${split_dir}/eval.parquet}" \
    --eval-interval 10 \
    --n-samples-per-eval-prompt 1 \
    --eval-temperature 0 \
    "$@"
