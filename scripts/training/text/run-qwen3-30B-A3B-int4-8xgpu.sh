#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-30B-A3B INT4 fake-QAT 8xGPU colocate training script (bridge mode).
#
# INT4 fake-QAT 说明：
#   - 训练侧：MoE expert 权重经 fake-quant STE 模拟 INT4 量化误差（symmetric, group_size=128）
#     - 前向：对每个 expert weight 做 per-group symmetric INT4 fake-quant（round+clamp+dequant）
#     - 反向：STE 直通，梯度等价于 BF16 训练，Master weight 保持 BF16 高精度更新
#     - 由 Megatron 的 INT4 fake-QAT helper 在 TEGroupedLinear._get_weight_tensors() 中注入
#     - 仅覆盖 MoE expert 层（TEGroupedLinear），attention/dense 层不受影响
#   - Rollout 侧：SGLang 使用真实 INT4（compressed-tensors W4A16 asymmetric, group_size=128）推理
#     - 每个 step 结束，BF16 训练权重经 pack_layer() 量化打包为 AWQ INT4 格式后
#       通过 NCCL 同步到 rollout engine（见 quantizer_compressed_tensors.py）
#   - 前提：
#     - --hf-checkpoint 需指向 W4A16 INT4 HF checkpoint（config.json 中须含
#       quantization_config: {quant_method: compressed-tensors, ...}）
#     - relax/backends/megatron/kernels/int4_qat/ 下的 fake_int4_quant_cuda
#       须已编译安装（cd relax/backends/megatron/kernels/int4_qat && pip install -e .）
#
# Usage:
#   bash scripts/training/text/run-qwen3-30B-A3B-8xgpu-int4-bridge.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-30B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-30B-A3B-int4
   # Megatron BF16 checkpoint
   --ref-load ${MODEL_DIR}/Qwen3-30B-A3B
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --load ${EXP_DIR}/Qwen3-30B-A3B_dist
   --save ${EXP_DIR}/Qwen3-30B-A3B_dist
   --save-interval 100
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
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --global-batch-size 128
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --balance-data
   --use-fault-tolerance
   --train-iters 200
)

EVAL_ARGS=(
   --skip-eval-before-train
   --log-passrate
   --eval-interval 2000
   --eval-prompt-data aime ${DATA_DIR}/dapo-math-17k/dapo-100.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480

   # MoE dispatcher
   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --moe-router-dtype fp32

   # INT4 fake-QAT 训练侧保持 BF16（假量化 STE 由 OPEN_TRAINING_INT4_FAKE_QAT_FLAG 控制）
   --transformer-impl transformer_engine
   --bf16
   # INT4 fake-QAT replaces weight tensors; TE fused wgrad requires the original tensors.
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

   --moe-router-load-balancing-type "none"
   --moe-aux-loss-coeff 0.0
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 128)
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen3-30B-A3B-int4-r3${now}
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

_EXTRA_ENV="{
   \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
   \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"128\"
}"

export RUNTIME_ENV_JSON=$(echo "${RUNTIME_ENV_JSON}" | jq --argjson extra "${_EXTRA_ENV}" '.env_vars += $extra')

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --use-health-check \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-30B-A3B-int4-GRPO-gpu8-${now}.log
