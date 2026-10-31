#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Text-only Keye DAPO Math RL, 200 rollouts; checkpoint every 20 updates.
# Save/resume the stable SAVE_DIR/EXP_NAME checkpoint; SAVE_CHECKPOINT=0 disables it.
# Usage: MODEL_DIR=/models DATA_DIR=/data SAVE_DIR=/checkpoints bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-math.sh

set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
source "${SCRIPT_DIR}/keye-vl2-30B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/rl/dapo-math}"
EXP_NAME="${EXP_NAME:-keye-vl2-30b-a3b-dapo-8xgpu-colocate}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"
case "${SAVE_CHECKPOINT}" in
    0|1) ;;
    *) echo "SAVE_CHECKPOINT must be 0 or 1" >&2; exit 2 ;;
esac
now=$(date "+%Y%m%d-%H%M%S")
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
# Local Ray control and rollout traffic must stay on the node's local network.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1${MASTER_ADDR:+,${MASTER_ADDR}}"
export no_proxy="${NO_PROXY}"
RUNTIME_ENV_JSON=$(python - "${RUNTIME_ENV_JSON}" <<'PY'
import json
import os
import sys

runtime = json.loads(sys.argv[1])
runtime.setdefault("env_vars", {}).update(NO_PROXY=os.environ["NO_PROXY"], no_proxy=os.environ["NO_PROXY"])
print(json.dumps(runtime))
PY
)

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Keye-VL-2.0-30B-A3B"
    --ref-load "${MODEL_DIR}/Keye-VL-2.0-30B-A3B"
    --megatron-to-hf-mode bridge
)
if [ "${SAVE_CHECKPOINT}" = 1 ]; then
    CKPT_ARGS+=(
        --load "${SAVE_DIR}/${EXP_NAME}"
        --save "${SAVE_DIR}/${EXP_NAME}"
        --save-interval "${SAVE_INTERVAL:-20}"
        --max-actor-ckpt-to-keep 2
        # Keep the checkpoint's LR/WD schedule when extending NUM_ROLLOUT.
        --use-checkpoint-opt_param-scheduler
    )
fi
ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo
    --reward-key score
    --num-rollout "${NUM_ROLLOUT:-200}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-16}"
    --n-samples-per-prompt "${N_SAMPLES:-8}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
    --rollout-max-response-len "${RESPONSE_LEN:-8192}"
    --rollout-temperature 1
    --balance-data
)
EVAL_ARGS=(
    --skip-eval-before-train
    --log-passrate
    --eval-interval "${EVAL_INTERVAL:-20}"
    --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 16384
    --eval-top-p 0.7
)
PERF_ARGS=(
    --no-rope-fusion
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-32768}"
    --log-probs-max-tokens-per-gpu "${LOGPROBS_MAX_TOKENS:-32768}"
    --calculate-per-token-loss
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0
)
ALGORITHM_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0
    --kl-loss-type low_var_kl
    --entropy-coef 0
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
)
SGLANG_ARGS=(
    --sglang-external-model-package relax.models.keye_vl2.sglang
    --sglang-json-model-override-args '{"language_model_only":true}'
    # Four TP2 engines share each model replica across two GPUs.
    --rollout-num-gpus-per-engine "${SGLANG_TP:-2}"
    --sglang-mem-fraction-static 0.65
    --sglang-attention-backend triton
    --sglang-cuda-graph-backend-decode full
    --sglang-disable-prefill-cuda-graph
    --sglang-cuda-graph-bs-decode 1 2 4 8 16 32
    --sglang-disable-overlap-schedule
    --sglang-disable-radix-cache
    --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS:-262144}"
    --sglang-max-running-requests "${SGLANG_CONCURRENCY:-32}"
    --sglang-chunked-prefill-size 4096
    --sglang-context-length 32768
)
MONITOR_ARGS=(
    --use-clearml
    --use-metrics-service
    --tensorboard-dir "${SAVE_DIR}/${EXP_NAME}/tensorboard"
    --rollout-result-dir "${SAVE_DIR}/${EXP_NAME}/rollout_result"
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
if [ "${SAVE_DEBUG_DATA:-0}" = 1 ]; then
    MONITOR_ARGS+=(
        --save-debug-rollout-data "${SAVE_DIR}/${EXP_NAME}/debug/rollout/{rollout_id}.pt"
        --save-debug-train-data "${SAVE_DIR}/${EXP_NAME}/debug/train/{rollout_id}_{rank}.pt"
    )
fi
MISC_ARGS=(
    --attention-dropout 0
    --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

mkdir -p "${SCRIPT_DIR}/log"
ray job submit \
    ${RAY_NO_WAIT:+--no-wait} \
    ${SUBMISSION_ID:+--submission-id "${SUBMISSION_ID}"} \
    --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${ALGORITHM_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MONITOR_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "$@" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
