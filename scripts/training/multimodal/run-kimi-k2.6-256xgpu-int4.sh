#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Kimi K2.6 (KimiK25ForConditionalGeneration) 256xGPU colocate multimodal training, INT4 QAT.
#
# Canonical "INT4 inference + BF16 training (QAT)" form (slime-style), validated by
# slime's scripts/low_precision/run-kimi-k2-Thinking-int4.sh. The previous all-args-to-
# INT4-dir layout (which relied on the now-removed auto-cast resolver) is preserved as
# run-kimi-k2.6-256xgpu-int4-legacy.sh for reference.
#
# How it works:
#  - SGLang inference loads the original compressed-tensors INT4 release directly. Its
#    param dict registers weight_packed/weight_scale/weight_shape triplets.
#  - Megatron training loads a pre-cast BF16 HF directory via bridge. fp32 master +
#    BF16 working weights — forward GEMM stays BF16 throughout.
#  - OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1 + OPEN_TRAINING_INT4_GROUP_SIZE=32 trip the
#    Megatron TEGroupedLinear._get_weight_tensors STE so each forward sees BF16 values
#    rounded to the INT4 grid (group_size=32). Backward is straight-through.
#  - On weight push, hf_config.quantization_config.quant_method == "compressed-tensors"
#    auto-routes through quantize_params_compressed_tensors → BF16 → INT4 repack →
#    SGLang in-place overwrites weight_packed/scale/shape buffers.
#
# Prerequisite (one-time):
#  Cast the original INT4 release to BF16 HF for the training side:
#    python -m relax.utils.quant_cast.convert_moe_int4_to_bf16 \
#        --model-dir  ${MODEL_DIR}/Kimi-K2.6 \
#        --output-dir ${MODEL_DIR}/Kimi-K2.6_bf16
#
# Model placement (TP=8 PP=8 CP=4 EP=32 ETP=1) matches the BF16 multimodal launcher;
# INT4 QAT only changes the weight-update path, not parallelism.
#
# Usage:
#   bash scripts/entrypoint/ray-job.sh scripts/training/multimodal/run-kimi-k2.6-256xgpu-int4.sh

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

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/kimi-k2.6-mm-int4}"
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
   # AutoConfig reads from here → hf_config.quantization_config.quant_method ==
   # "compressed-tensors" → push-side auto-repacks BF16 → INT4 (no env var needed).
   --hf-checkpoint ${HF_INT4}
   # SGLang loads the INT4 release directly so its param dict registers
   # weight_packed/scale/shape. MUST be the INT4 dir, NOT the BF16 cast — otherwise
   # weight pushes are dropped with "X.weight_packed not found in params_dict".
   --sglang-hf-checkpoint ${HF_INT4}
   # Megatron bridge loads BF16 HF here; the QAT STE rounds these to the INT4 grid
   # on each forward.
   --ref-load ${HF_BF16}
   --megatron-to-hf-mode bridge
   --save ${EXP_DIR}/Kimi-K2.6_mm_int4_ckpt/
   --save-interval 100
   --no-save-optim
   --no-save-rng
   --no-load-optim
   --no-load-rng
)

PROMPT_SET=${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet
SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   # --apply-chat-template-kwargs '{"thinking": false}'
   --rollout-shuffle
   --rm-type openr1mm
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 16
   --rollout-max-prompt-len 2048
   --rollout-max-response-len 4096
   --rollout-temperature 1.0
   --global-batch-size 512
   --balance-data
   --use-fault-tolerance
   --rollout-health-check-timeout 120
   --system-prompt "${SYSTEM_PROMPT}"
   --multimodal-keys '{"image":"image"}'
   --image-max-token-num 256
   --use-streaming-dataset
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
   --decoder-first-pipeline-num-layers 1
   --decoder-last-pipeline-num-layers 6
   --vision-dp-when-tp
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer

   --no-pin-cpu-grads
   --no-pin-cpu-params

   # NOTE(wuhuan): VLM training stability — disable rope fusion and MoE aux loss
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

   --sglang-load-format dummy
   # --sglang-disable-cuda-graph
   --sglang-cuda-graph-max-bs 8
   --sglang-server-concurrency 1024
   --sglang-watchdog-timeout 3600
   --sglang-enable-nan-detection
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name Kimi-K2.6-mm-256xgpu-int4-${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --trust-remote-code
   --update-weight-buffer-size $(( 4 * 512 * 1024 * 1024 )) \
)

# Inject INT4 QAT + networking env vars into Ray's runtime env. The base
# RUNTIME_ENV_JSON is assembled by scripts/entrypoint/{ray-job,local,spmd-multinode}.sh;
# merge with python rather than re-templating the whole JSON so we stay in sync with the
# entrypoint. Mirrors run-kimi-k2.6-256xgpu-bf16.sh (Gloo/NCCL pinning) plus the two
# QAT-only env vars.
RUNTIME_ENV_JSON=$(python3 -c '
import json, os
d = json.loads(os.environ["RUNTIME_ENV_JSON"])
d.setdefault("env_vars", {}).update({
    "TORCH_DIST_INIT_BARRIER": "1",
    "TORCH_NCCL_BLOCKING_WAIT": "0",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    "TORCH_DISTRIBUTED_DEFAULT_TIMEOUT": "3600",
    # DeepEP low-latency dispatch buffer cap. Default 128 collides with cuda_graph
    # capture at bs=128 (DP-attention pads x.size(0) past 128). Bumping to 256
    # covers the padded capture batch without changing dispatch numerics.
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    # INT4 QAT — fake-quantize BF16 weights to INT4 grid in forward via STE,
    # and read group_size=32 for the per-group scale layout.
    "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
    "OPEN_TRAINING_INT4_GROUP_SIZE": "32",
})
print(json.dumps(d))
')
export RUNTIME_ENV_JSON

mkdir -p log
mkdir -p save
# stdbuf -oL -eL forces line-buffered stdout/stderr through the | tee pipeline so
# the local log streams in real time. Critical when --no-wait is NOT set (ray job
# submit then blocks streaming live logs) — without stdbuf the pipe block-buffers
# 8K at a time and the tee'd file looks frozen for minutes.
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
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/Kimi-K2.6-mm-256xgpu-int4-${now}.log
