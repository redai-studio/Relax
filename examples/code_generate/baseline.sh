#!/bin/bash

# Qwen3.5-4B, DeepCoder train, 8 GPUs colocate GRPO.
# Submit with scripts/entrypoint/ray-job.sh examples/code_generate/baseline.sh.

set -ex
set -o pipefail

now=$(date "+%Y%m%d-%H%M%S")
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the prepared dataset directory}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${SANDBOX_URL:?Set SANDBOX_URL to the SandboxFusion service URL}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-4B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/baseline/deepcoder}"
EXP_NAME="${EXP_NAME:-deepcoder-qwen35-4b-think16k-8gpu}"

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.5-4B"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
   --load "${SAVE_DIR}/${EXP_NAME}/checkpoints"
   --save "${SAVE_DIR}/${EXP_NAME}/checkpoints"
   --save-interval 50
   --max-actor-ckpt-to-keep 2
)

PROMPT_SET="${DATA_DIR}/train.jsonl"

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key input
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking":true}'
   --rollout-shuffle
   --rollout-seed 42
   --custom-rm-path examples.code_generate.app.reward.reward
   --rm-url "${SANDBOX_URL:-}"
   --reward-max-concurrency 16
   --num-epoch 1
   --num-steps-per-rollout 1
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-prompt-len 2048
   --rollout-max-response-len 16384
   --rollout-temperature 1
   # --rollout-temperature 0.6
   # --rollout-top-p 0.95
   # --rollout-top-k 20
   --global-batch-size 256
   --use-fault-tolerance
   --balance-data
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
   --max-tokens-per-gpu 18432
   --log-probs-max-tokens-per-gpu 40960
   --calculate-per-token-loss
   --use-distributed-optimizer
   --overlap-grad-reduce
   --overlap-param-gather
   --empty-unused-memory-level 1
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --loss-type policy_loss
   --kl-coef 0
   --entropy-coef 0
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --lr-warmup-iters 0
   --weight-decay 0
   --adam-beta1 0.9
   --adam-beta2 0.98
   --clip-grad 1.0

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --no-rope-fusion
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8
   --sglang-context-length 32768
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
   --rollout-result-dir "${SAVE_DIR}/${EXP_NAME}/rollout_result"
)

MISC_ARGS=(
   --attention-dropout 0
   --hidden-dropout 0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --seed 42
   --num-data-storage-units 1
   --num-iters-per-train-update 1
   --use-health-check
   --max-global-restart 0
)

RESOURCE='{"actor": [1, 8], "rollout": [1, 8]}'

mkdir -p "${SAVE_DIR}/${EXP_NAME}/logs"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource "${RESOURCE}" \
   --colocate \
   --max-staleness 0 \
   --offload-rollout \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "$@" 2>&1 | tee "${SAVE_DIR}/${EXP_NAME}/logs/train-${now}.log"
