#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# 8 x 8 H800, frozen vision, language attention and independent expert LoRA.
# Set MODEL_DIR, DATA_DIR and SAVE_DIR on the Ray head, then run:
# bash scripts/entrypoint/ray-job.sh examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-kimi-lora-sft-64xh800.sh
set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# TP1 keeps the duplicated Q/KV adapter gradients local. CP32 keeps the
# production 128K context at 4096 tokens per rank. Each EP rank owns 12 experts.
export V41_KIMI_SFT_NODES=8 TP_SIZE=1 PP_SIZE=2 CP_SIZE=32 EP_SIZE=32
export V41_MXFP4_QAT=0
export EXP_NAME="${EXP_NAME:-deepseek-v41-1017-kimi-lora-sft-64xh800}"

# The attention modules correspond to wq_a, wq_b, wkv and wo_b. The FC pair
# covers w1/w3 (gate/up) and w2 (down) in both routed and shared experts.
exec bash "${script_dir}/run-deepseek-v4.1-flash-1017-kimi-sft-128xh800.sh" \
    --lora-rank "${LORA_RANK:-16}" \
    --lora-alpha "${LORA_ALPHA:-32}" \
    --lora-dropout 0.0 \
    --lora-scope language \
    --lora-merge-mode \
    --lora-target-modules linear_q_down_proj linear_q_up_proj linear_kv_proj linear_proj linear_fc1 linear_fc2 \
    --train-env-vars '{"RELAX_LORA_SHARE_EXPERT_ADAPTERS":"false"}' \
    --save-lora-only \
    "$@"
