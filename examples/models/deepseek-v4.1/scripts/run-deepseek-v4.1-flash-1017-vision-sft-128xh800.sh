#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 40-layer frozen-vision OpenSearch-VL run on an existing 16 x 8 H800 Ray cluster.
# From the head node, set MODEL_DIR, DATA_DIR and SAVE_DIR, then run this script with bash.
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "${script_dir}/../../../.." && pwd)
recipe="${script_dir}/run-deepseek-v4.1-flash-1017-vision-sft.sh"

export EXP_NAME="${EXP_NAME:-1017-opensearch-40layer-128xh800-gbs128-200step-repro}"
export NUM_NODES=16 TP_SIZE=2 PP_SIZE=2 CP_SIZE=4 EP_SIZE=64
export SEQ_LENGTH=32768 MAX_TOKENS_PER_GPU=8192 GLOBAL_BATCH_SIZE=128
export NUM_STEPS=200 SAVE_INTERVAL=50 MAX_ACTOR_CKPT_TO_KEEP=4 EVAL_INTERVAL=10
export MOE_DISPATCHER=alltoall LR=1e-6 SEED=1234
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export RELAX_SKIP_CLUSTER_CLEANUP=1

args=(
    --use-tensorboard
    --sft-tq-timeout-minutes 90
    --moe-permute-fusion
    "$@"
)
if [[ "${DRY_RUN:-0}" == 1 || -n "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "${recipe}" "${args[@]}"
fi
exec bash "${repo}/scripts/entrypoint/ray-job.sh" "${recipe}" "${args[@]}"
