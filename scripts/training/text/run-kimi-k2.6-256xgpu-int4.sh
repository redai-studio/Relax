#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Kimi K2.6 256xGPU text-only colocate training, INT4 QAT.
#
# Canonical "INT4 inference + BF16 training (QAT)" form — mirrors the multimodal
# launcher (run-kimi-k2.6-256xgpu-int4.sh) with text-only data and algorithm settings.
#
# How it works:
#  - SGLang inference loads the original compressed-tensors INT4 release directly.
#  - Megatron training loads a pre-cast BF16 HF directory via bridge.
#  - OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1 + OPEN_TRAINING_INT4_GROUP_SIZE=32 trip the
#    Megatron TEGroupedLinear._get_weight_tensors STE so each forward sees BF16 values
#    rounded to the INT4 grid (group_size=32). Backward is straight-through.
#  - On weight push, hf_config.quantization_config.quant_method == "compressed-tensors"
#    auto-routes through quantize_params_compressed_tensors → BF16 → INT4 repack.
#
# Model placement (TP=8 PP=8 CP=4 EP=32 ETP=1) matches the multimodal launcher.
# SGLang inference uses 16 GPUs per engine with DP-attention (dp_size=16, ep_size=16).
#
# Prerequisite (one-time):
#  Cast the original INT4 release to BF16 HF for the training side:
#    python -m relax.utils.quant_cast.convert_moe_int4_to_bf16 \
#        --model-dir ${MODEL_DIR}/Kimi-K2.6 \
#        --output-dir ${MODEL_DIR}/Kimi-K2.6_bf16
#
# Usage:
#   bash scripts/entrypoint/ray-job.sh scripts/training/text/run-kimi-k2.6-256xgpu-bf16.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/kimi-k2.6.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/kimi-k2.6-text-int4}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

# Two checkpoints — distinct roles:
#  HF_INT4 — original compressed-tensors release. Used for AutoConfig (tokenizer +
#            quant_method + group_size for QAT) and SGLang inference load.
#  HF_BF16 — pre-cast BF16 HF directory (see prerequisite in header). Used by
#            Megatron bridge to load real training weights.
HF_INT4="${HF_INT4:-${MODEL_DIR}/Kimi-K2.6/}"
HF_BF16="${HF_BF16:-${MODEL_DIR}/Kimi-K2.6_bf16/}"

CKPT_ARGS=(
   --hf-checkpoint ${HF_INT4}
   --sglang-hf-checkpoint ${HF_INT4}
   --ref-load ${HF_BF16}
   --megatron-to-hf-mode bridge
   --save ${EXP_DIR}/Kimi-K2.6_text_int4_ckpt/
   --save-interval 50
   --no-save-optim
   --no-save-rng
   --no-load-optim
   --no-load-rng
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT}
   --use-fault-tolerance
   --rollout-health-check-timeout 120

   --rm-type math

   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --skip-eval-before-train
   --log-passrate
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   # INT4 fake-QAT replaces weight tensors; TE fused wgrad requires the original tensors.
   --no-gradient-accumulation-fusion
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 8
   --context-parallel-size 4
   --calculate-per-token-loss
   --expert-model-parallel-size 32
   --expert-tensor-parallel-size 1
   --decoder-last-pipeline-num-layers 5

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --no-pin-cpu-grads
   --no-pin-cpu-params

   --no-rope-fusion
   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 16
   --sglang-mem-fraction-static 0.7
   # dp attention
   --sglang-enable-dp-attention
   --sglang-dp-size 16
   --sglang-moe-dense-tp-size 1
   --sglang-enable-dp-lm-head
   --sglang-ep-size 16

   --sglang-cuda-graph-max-bs 8
   --sglang-server-concurrency 1024
   --sglang-watchdog-timeout 3600
   --sglang-enable-nan-detection
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name Kimi-K2.6-text-256xgpu-int4-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-check-for-nan-in-loss-and-grad
   --trust-remote-code
   --update-weight-buffer-size $(( 4 * 512 * 1024 * 1024 )) \
)

RUNTIME_ENV_JSON=$(python3 -c '
import json, os
d = json.loads(os.environ["RUNTIME_ENV_JSON"])
d.setdefault("env_vars", {}).update({
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
    "OPEN_TRAINING_INT4_GROUP_SIZE": "32",
})
print(json.dumps(d))
')
export RUNTIME_ENV_JSON

mkdir -p log
mkdir -p save
stdbuf -oL -eL ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 256], "rollout": [1, 256]}'\
   --max-staleness 0 \
   --num-data-storage-units 32 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/Kimi-K2.6-text-256xgpu-int4-${now}.log
