#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-9B 8xGPU colocate (sync) PPO training script for DAPO math dataset.
#
# Adapted from run-qwen3-4B-8xgpu-ppo.sh: same PPO wiring (critic + advantages,
# GAE, KL-loss path enabled at coef=0, --use-rollout-logprobs required by
# colocate), swapped in the qwen35-9B model config and TP=4 to match the
# existing run-qwen35-9B-8xgpu.sh GRPO baseline.
#
# Notes:
#   - PPO requires critic + advantages entries in --resource. critic shares the
#     actor placement group when their resource shape matches (see
#     relax/core/placement_roles.py:20), advantages runs CPU-only ([1, 0]).
#   - Colocate mode has no independent actor_fwd/reference roles, so we must
#     use --use-rollout-logprobs and keep KL disabled (see relax/core/ppo_validation.py).
#   - --offload-rollout puts SGLang to sleep after generate, so actor+critic
#     get the full GPU while training. SGLang mem fraction stays at 0.8.
#   - Critic starts from the same HF checkpoint as the actor and is warmed up
#     for a few pure-critic steps before joint updates begin.
#   - 9B + critic is tighter than 4B PPO; recompute stays on and TP=4 matches
#     the existing 9B GRPO baseline.
#
# Usage:
#   bash scripts/training/text/run-qwen35-9B-8xgpu-ppo.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "当前时间: $now"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    export RUNTIME_ENV_JSON=$(echo "$RUNTIME_ENV_JSON" | jq --arg k "$WANDB_API_KEY" '.env_vars.WANDB_API_KEY = $k')
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dapo-math-ppo}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=1000}"
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:=1}"
CRITIC_LR_WARMUP_ITERS="${CRITIC_LR_WARMUP_ITERS:=1}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen3.5-9B
   --ref-load ${MODEL_DIR}/Qwen3.5-9B
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --load ${EXP_DIR}/Qwen3.5-9B_mcore_ppo_8xgpu/actor/
   --save ${EXP_DIR}/Qwen3.5-9B_mcore_ppo_8xgpu/actor/
   --save-interval 50
   --max-actor-ckpt-to-keep 1

   # critic must save to a separate subdir; actor and critic both call save(...) with
   # self.args.save, so a shared root causes them to overwrite each other's iter_XXXXXX/.
   --critic-load ${EXP_DIR}/Qwen3.5-9B_mcore_ppo_8xgpu/critic/
   --critic-save ${EXP_DIR}/Qwen3.5-9B_mcore_ppo_8xgpu/critic/
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
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-32}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-4}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-8192}
   # Keep rollout logprobs closer to actor logprobs (--use-rollout-logprobs is on in colocate).
   --rollout-temperature 0.8
   --global-batch-size ${GLOBAL_BATCH_SIZE:-128}
   --balance-data
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 20
   --eval-prompt-data aime ${DATA_DIR}/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   # TP=4 matches the existing 9B GRPO baseline; critic doubles training memory
   # so keep recompute on.
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --log-probs-max-tokens-per-gpu 40960

   --no-rope-fusion
)

PPO_ARGS=(
   --advantage-estimator ppo
   # GAE — slime defaults; the 0.95 λ used previously is not what slime tests.
   --gamma 1.0
   --lambd 1.0
   # actor clip
   --eps-clip 0.2
   --eps-clip-high 0.2
   --entropy-coef 0.0
   # Slime test explicitly wires up the KL-loss path even at coef=0 to keep
   # the code path identical to KL-on runs. Mirror that here.
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type k1
   --kl-coef 0.0
   # Whiten advantages across the DP group — slime's default for PPO.
   --normalize-advantages
   # critic
   --value-clip 0.5
   --critic-lr 1e-5
   --num-critic-only-steps ${NUM_CRITIC_ONLY_STEPS}
   --critic-lr-warmup-iters ${CRITIC_LR_WARMUP_ITERS}
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
   --rollout-num-gpus-per-engine 2
   # Full 0.8 fraction is safe because PPOCritic now polls
   # rollout_manager.get_status and waits for SGLang to finish offload before
   # calling wake_up (see relax/components/ppo_critic.py).
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name  ${PROJECT_NAME}
   --tb-experiment-name qwen35-9B-ppo-8x-${now}
)

if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS+=(
       --use-wandb
       --wandb-project ${PROJECT_NAME//\//-}
       --wandb-group qwen35-9B-ppo-8x-${now}
    )
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "critic": [1, 8], "rollout": [1, 8], "advantages": [1, 0]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --offload-rollout \
   --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"  2>&1 | tee log/qwen35-9B-PPO-gpu8-${now}.log
