#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 4xNPU colocate (sync) GRPO + MTP joint-training script.
#
# Phase-1 RL MTP: trains the native MTP head jointly with the policy via an
# auxiliary loss (slime-style). Rollout keeps `enable_draft_weights_cpu_backup=True`
# so SGLang inference uses the base model only — no speculative decoding here.
#
# Requires the HF checkpoint to contain MTP weights (`num_nextn_predict_layers>=1`).
#
# Differences from the GPU MTP script:
#   - Removes --cross-entropy-loss-fusion / --cross-entropy-fusion-impl te
#     (TransformerEngine fused CE kernel is CUDA-only, not available on NPU).
#   - Uses NPU-specific SGLang args (--sglang-device npu, ascend attention backend).
#   - Uses NPU-specific perf args (--qkv-format bshd, --no-rope-fusion, etc.).
#
# Usage:
#   bash scripts/training/text/run_qwen35_9B_mtp_8xnpu_thd.sh

set -ex
set -o pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"
export ASCEND_COREDUMP_SIGNAL=none
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_HOST_SOCKET_PORT_RANGE=63000-63050
export HCCL_NPU_SOCKET_PORT_RANGE=64000-64050
export TMS_HOOK_MODE="preload"
export HYDRA_FULL_ERROR=1



SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local-npu.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"
# Support setting env from outside
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math-mtp}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"


CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B/
   --ref-load ${MODEL_DIR}/Qwen3.5-9B/
   --megatron-to-hf-mode bridge
   # --load ${EXP_DIR}/Qwen3.5-9B-mtp-save-0821
   --save ${EXP_DIR}/Qwen3.5-9B-mtp-save-0821
   --save-interval 100
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl
ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 50
   --eval-prompt-data aime ${EXP_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --qkv-format thd
   --max-tokens-per-gpu 10240

   --no-rope-fusion
   --no-gradient-accumulation-fusion
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

   --custom-tis-function-path relax.backends.megatron.loss.icepop_function
)

if [[ "${MTP_NUM_LAYERS:-1}" != "1" ]]; then
    echo "ERROR: MTP_NUM_LAYERS must be 1 for Qwen3.5 (checkpoint has mtp_num_hidden_layers=1)." >&2
    exit 1
fi

MTP_ARGS=(
   --mtp-num-layers ${MTP_NUM_LAYERS:-1}
   --enable-mtp-training
   --mtp-loss-scaling-factor ${MTP_LOSS_SCALING_FACTOR:-0.1}
   # NOTE: --cross-entropy-loss-fusion / --cross-entropy-fusion-impl te are
   # intentionally omitted — the TE fused CE kernel is CUDA-only. Megatron
   # will fall back to the non-fused cross-entropy path.
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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 48 64
   --sglang-max-running-requests 64
   --sglang-device npu
   --sglang-chunked-prefill-size 8192
   --sglang-max-prefill-tokens 8192
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-attention-backend ascend
   --sglang-max-mamba-cache-size 352
   --sglang-mamba-ssm-dtype bfloat16
   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-speculative-algorithm NEXTN
   --sglang-speculative-num-steps 2
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 3

)
WANDB_ARGS=(
   --use-tensorboard
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen35-9B-mtp-GRPO-4x-sync-${now}
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   --use-flash-attn
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${MTP_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee log/qwen35-9B-MATH-npu16-colocate-mtp-${now}.log
