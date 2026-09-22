#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3-4B 8xGPU colocate training script demonstrating multi-instance GenRM
# (--genrm-instances): two independent judge models served by one GenRM
# deployment, routed by GenRMClient.generate(route_key=...).
#
# Mode: SPLIT-BUNDLE — rollout on 4 GPUs, GenRM on the other 4 GPUs (2 GPUs
# per judge instance). Both run in parallel; reward is fired inline per-sample
# during rollout via a custom dual-judge reward function.
#   - "quality" instance: Qwen3.8-27B, correctness judge (2 GPUs)
#   - "safety"  instance: Qwen3.5-35B-A3B, harmlessness judge (2 GPUs)
# See examples/generate_reward_model/reward_dual_genrm_quality_safety.py for
# how the two judges' scores are combined into one training reward.
#
# Usage:
#   bash examples/generate_reward_model/run-qwen3-4B-8xgpu-dual-genrm-split.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Auto-source local environment when not launched via an external entrypoint
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi

source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dual-genrm-split}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=200}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3-4B/
   --ref-load ${MODEL_DIR}/Qwen3-4B/
   --megatron-to-hf-mode bridge
)

PROMPT_SET=${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl

# ============================================
# ROLLOUT ARGS — custom dual-judge reward via --custom-rm-path.
# --rm-type dummy is a placeholder; --custom-rm-path takes priority over it
# (see relax/engine/rewards/__init__.py RewardExecutor.execute).
# ============================================
ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type dummy
   --custom-rm-path examples.generate_reward_model.reward_dual_genrm_quality_safety.reward_func
   --reward-key score

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
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

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
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
)

# Rollout engine for the 4B actor. Runs on its dedicated 4-GPU bundle (split
# from GenRM's 4 GPUs via rollout_num_gpus + genrm total num_gpus == actor
# total, see relax/utils/arguments.py's colocate+genrm split-bundle check).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)

# Two GenRM instances sharing one Serve deployment, routed by route_key. Each
# gets its own dedicated 2-GPU bundle within the 4-GPU GenRM region (split
# from rollout's 4 GPUs). num_gpus_per_engine=2 -> one TP=2 SGLang engine per
# instance (no per-instance replica splitting needed at this scale).
QUALITY_JUDGE_PATH="${QUALITY_JUDGE_PATH:-${MODEL_DIR}/Qwen3.8-27B}"
SAFETY_JUDGE_PATH="${SAFETY_JUDGE_PATH:-${MODEL_DIR}/Qwen3.5-35B-A3B}"
GENRM_ARGS=(
   --genrm-instances "{\"quality\": {\"model_path\": \"${QUALITY_JUDGE_PATH}\", \"num_gpus\": 2, \"num_gpus_per_engine\": 2, \"engine_config\": {\"max_context_len\": 10240, \"mem_fraction_static\": 0.6}, \"sampling_config\": {\"temperature\": 0.1, \"top_p\": 1.0, \"top_k\": -1, \"max_response_len\": 64, \"chat_template_kwargs\": {\"enable_thinking\": false}}}, \"safety\": {\"model_path\": \"${SAFETY_JUDGE_PATH}\", \"num_gpus\": 2, \"num_gpus_per_engine\": 2, \"engine_config\": {\"max_context_len\": 10240, \"mem_fraction_static\": 0.6}, \"sampling_config\": {\"temperature\": 0.1, \"top_p\": 1.0, \"top_k\": -1, \"max_response_len\": 32, \"chat_template_kwargs\": {\"enable_thinking\": false}}}}"
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen3-4b-GRPO-DualGenRM-split-gpu8-${now}
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
)

# ============================================
# Resource Configuration (SPLIT-BUNDLE):
# 8 GPU total. Actor runs on all 8; rollout and GenRM split into two dedicated
# 4-GPU bundles (rollout_num_gpus + genrm total num_gpus == actor_total
# triggers split-bundles colocate in relax/utils/arguments.py).
#   - rollout (4B actor) on 4 GPUs, TP=1 SGLang engine
#   - GenRM "quality" (Qwen3.8-27B)     on 2 of the remaining 4 GPUs, TP=2
#   - GenRM "safety"  (Qwen3.5-35B-A3B) on the other 2 GPUs,          TP=2
# Reward is fired inline per-sample during rollout (custom dual-judge
# reward), so both judges overlap with rollout generation.
# ============================================

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 ${SCRIPT_DIR}/../../relax/entrypoints/train.py \
   --resource '{"actor": [1, 8], "rollout": [1, 4], "genrm": [1, 4]}' \
   --colocate \
   --rollout-num-gpus 4 \
   --max-staleness 0 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${GENRM_ARGS[@]}" \
   "${MISC_ARGS[@]}"  2>&1 | tee log/qwen3-4b-GRPO-DualGenRM-split-gpu8-${now}.log
