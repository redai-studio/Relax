#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full 1017 frozen-vision THD SFT on 4 x 8 H800. Train the language attention adapters.
# bash scripts/entrypoint/ray-job.sh examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-vision-lora-sft-32xh800.sh
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export NUM_NODES=4
export TP_SIZE="${TP_SIZE:-2}" PP_SIZE="${PP_SIZE:-2}" CP_SIZE="${CP_SIZE:-4}"
export EP_SIZE="${EP_SIZE:-$((32 / PP_SIZE))}" MOE_DISPATCHER=alltoall
export SEQ_LENGTH="${SEQ_LENGTH:-16384}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((SEQ_LENGTH / CP_SIZE))}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}" LR="${LR:-1e-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}" MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
export EXP_NAME="${EXP_NAME:-deepseek-v41-1017-opensearch-attention-lora-32xh800}"

exec bash "${script_dir}/run-deepseek-v4.1-flash-1017-vision-sft.sh" \
    --lora-rank "${LORA_RANK:-8}" \
    --lora-alpha "${LORA_ALPHA:-16}" \
    --lora-dropout 0.0 \
    --lora-scope language \
    --lora-target-modules linear_q_up_proj linear_proj \
    --save-lora-only \
    --sft-chunked-logits \
    --sft-logits-chunk-size 1024 \
    "$@"
