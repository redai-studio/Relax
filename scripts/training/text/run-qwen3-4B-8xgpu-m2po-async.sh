#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU M2PO fully-async off-policy validation script.
# Reference: https://arxiv.org/abs/2510.01161 (Second-Moment Trust Policy Optimization)
#
# This is the ASYNC / off-policy counterpart of run-qwen3-4B-8xgpu-m2po.sh.
# Key differences from the colocate M2PO script:
#   --fully-async                 (actor and rollout run on independent GPU pools)
#   --resource split 4+4          (actor: 4 GPUs, rollout: 4 GPUs, advantages: 0)
#   --max-staleness ${MAX_STALENESS}  (>0 => off-policy: rollout keeps generating
#                                 with weights up to N steps behind the trainer)
#   --rollout-num-gpus-per-engine 1   (4 rollout GPUs => 4 SGLang engines)
#
# Off-policy is exactly the regime M2PO is designed for: stale samples inflate the
# importance ratio, and M2PO's second-moment trust region adapts the clip instead
# of discarding those tokens. Raise MAX_STALENESS to stress the claim harder.
#
# Usage:
#   bash scripts/training/text/run-qwen3-4B-8xgpu-m2po-async.sh
#   MAX_STALENESS=8 bash scripts/training/text/run-qwen3-4B-8xgpu-m2po-async.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/m2po-math-async}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"
# Off-policy staleness bound: 0=on-policy, >0=async off-policy. 256 is a heavy
# off-policy stress test of M2PO's core claim; lower it (e.g. 4/32) for a milder gap.
MAX_STALENESS="${MAX_STALENESS:=256}"


CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B-Instruct/
   --ref-load ${MODEL_DIR}/Qwen3-4B-Instruct/
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --load ${EXP_DIR}/Qwen3-4B-Instruct_mcore_8xgpu_m2po_async/
   --save ${EXP_DIR}/Qwen3-4B-Instruct_mcore_8xgpu_m2po_async/
   # Save every 20 steps so eviction loses at most 20 steps; --load points to the
   # same directory so Relax auto-resumes from the latest checkpoint on restart.
   --save-interval 20
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
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
   --use-fault-tolerance
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
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --calculate-per-token-loss
   --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --log-probs-max-tokens-per-gpu 30720
)

# M2PO-specific args: adaptive second-moment clipping, no fixed eps-clip
M2PO_ARGS=(
   --advantage-estimator m2po

   # Second-moment budget: controls how much harmful-token deviation triggers clipping.
   # Paper default ~0.04; smaller = tighter/more-frequent clipping, larger = more off-policy tolerance.
   --m2po-kl2-budget 0.01

   # Miniclip floors: ensure minimum clip width (mirrors DAPO asymmetric clip defaults).
   --m2po-miniclip-low 0.2
   --m2po-miniclip-high 0.28

   # KL regularization (optional; use same settings as GRPO baseline for fair comparison)
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00

   --use-tis
)

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
   --sglang-mem-fraction-static 0.8
)

TRACKING_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   # experiment name encodes algo + staleness for easy ClearML comparison
   --tb-experiment-name qwen3-4b-M2PO-async-s${MAX_STALENESS}-gpu8-${now}
   # Only report metrics relevant to M2PO validation; rollout/staleness exposes the
   # actual off-policy gap so you can correlate it with m2po_eps adaptation.
   --clearml-key-filter "train/loss,train/pg_loss,train/pg_clipfrac,train/ppo_kl,train/m2po_eps,train/entropy_loss,train/grad_norm,rollout/reward,rollout/staleness/avg,eval/aime"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p log
# Disable NVLS to avoid NVSwitch/FabricManager compatibility errors on H20
export NCCL_NVLS_ENABLE=0

ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 4], "rollout": [1, 4], "advantages": [1, 0]}'\
   --max-staleness ${MAX_STALENESS} \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 8 \
   --fully-async \
   --use-health-check \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${M2PO_ARGS[@]}" \
   "${TRACKING_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-4b-M2PO-async-s${MAX_STALENESS}-gpu8-${now}.log
