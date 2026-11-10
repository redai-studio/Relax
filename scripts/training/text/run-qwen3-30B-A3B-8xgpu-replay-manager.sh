#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3-30B-A3B Replay Manager prototype: 8 GPUs, colocated GRPO, MoE replay.
# Provide MODEL_DIR, DATA_DIR, SAVE_DIR, RAY_ADDRESS (GCS), and
# RAY_DASHBOARD_ADDRESS (Ray Jobs HTTP URL), then run:
#   bash scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen3-30B-A3B-8xgpu-replay-manager.sh
# Defaults to refs; RELAX_REPLAY_REFERENCES=0 selects full tensors.
# Keep SAVE_DIR and EXP_NAME stable to resume; use a new EXP_NAME for a new experiment.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the dedicated Ray GCS address}"
: "${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the dedicated Ray Jobs HTTP URL}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/ray-job.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen3-30B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/rl/replay-manager}"
EXP_NAME="${EXP_NAME:-qwen3-30b-a3b-replay-manager-8xgpu-colocate}"
NUM_ROLLOUT="${NUM_ROLLOUT:-8}"
now=$(date "+%Y%m%d-%H%M%S")

export RELAX_REPLAY_REFERENCES="${RELAX_REPLAY_REFERENCES:-1}"
export RELAX_REPLAY_RUN_ID="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
: "${RUNTIME_ENV_JSON:?Launch through the existing Relax entrypoint}"
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - <<'PYENV'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
workers = runtime.setdefault("env_vars", {})
for name in (
    "RELAX_REPLAY_REFERENCES",
    "RELAX_REPLAY_RUN_ID",
    "RELAX_REPLAY_CAPACITY_MB",
    "NCCL_NVLS_ENABLE",
):
    if name in os.environ:
        workers[name] = os.environ[name]
print(json.dumps(runtime))
PYENV
)

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3-30B-A3B"
   --ref-load "${MODEL_DIR}/Qwen3-30B-A3B"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache

   --load "${SAVE_DIR}/${EXP_NAME}"
   --save "${SAVE_DIR}/${EXP_NAME}"
   --save-interval 8
   --max-actor-ckpt-to-keep 2
)

PROMPT_SET="${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --num-rollout "${NUM_ROLLOUT}"
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
   --eval-interval 4
   --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
   --log-probs-max-tokens-per-gpu 40960

   --moe-flex-dispatcher-backend deepep
   --moe-token-dispatcher-type flex
   --moe-router-dtype fp32
   --calculate-per-token-loss

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
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
   --seed 1234
   --rollout-seed 42
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

ROUTING_REPLAY_ARGS=(
   --use-rollout-routing-replay
   --use-slime-router
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_DASHBOARD_ADDRESS}" \
   --submission-id "${EXP_NAME}-${now}" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   --colocate \
   --use-health-check \
   "${ROUTING_REPLAY_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" "$@" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
