#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Keye OpenR1-MM RL; 64 x 8 samples per update, TP1/PP1/CP2/EP8 training and TP2 rollout.
# Usage: MODEL_DIR=/models DATA_DIR=/data SAVE_DIR=/checkpoints bash examples/models/keye-vl2/scripts/run-keye-vl2-30B-A3B-8xgpu-openr1mm.sh

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
MODEL_ARGS+=(
    --position-embedding-type mrope
    --mrope-section 16 24 24
    --no-rope-fusion
)

PROJECT_NAME="${PROJECT_NAME:-Relax/rl/openr1mm}"
EXP_NAME="${EXP_NAME:-keye-vl2-30b-a3b-openr1mm-cp2-8xgpu-colocate}"
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
runtime.setdefault("env_vars", {}).update(
    NO_PROXY=os.environ["NO_PROXY"], no_proxy=os.environ["NO_PROXY"],
    # Vision/projector weights change after every update; encoded features must not survive it.
    SGLANG_VLM_CACHE_SIZE_MB="0",
)
print(json.dumps(runtime))
PY
)

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Keye-VL-2.0-30B-A3B"
    --ref-load "${MODEL_DIR}/Keye-VL-2.0-30B-A3B"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
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
    --prompt-data "${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --multimodal-keys '{"image":"image"}'
    --use-streaming-dataset
    --custom-prompt-path relax.models.keye_vl2.prompts.openr1_prompt
    --rm-type openr1mm
    --num-rollout "${NUM_ROLLOUT:-200}"
    # Match the Qwen3.5-35B-A3B OpenR1-MM sampling and update batch.
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-64}"
    --n-samples-per-prompt "${N_SAMPLES:-8}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-512}"
    --rollout-max-response-len "${RESPONSE_LEN:-2048}"
    --rollout-max-prompt-len 2048
    --rollout-max-context-len 4096
    --rollout-temperature 1
    --balance-data
    --use-fault-tolerance
)
EVAL_ARGS=(
    --skip-eval-before-train
    --log-passrate
)
PERF_ARGS=(
    # Keye's trainable vision wrapper requires TP1/PP1; retain zigzag CP2.
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 2
    --cp-partition-mode zigzag
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --log-probs-max-tokens-per-gpu 8192
    --calculate-per-token-loss
    # Do not use flex/DeepEP with this Keye rollout implementation: it rounds
    # each EP rank's partial expert sum to BF16 before combine, unlike rollout's
    # single FP32 accumulation. EP8 OpenR1-MM measured ~0.022 probability mismatch.
    # Keep alltoall to merge all routed experts once and match rollout arithmetic.
    --moe-token-dispatcher-type alltoall
    --moe-router-dtype fp32
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0
)
ALGORITHM_ARGS=(
    --advantage-estimator grpo
    --true-on-policy-mode
    --kl-loss-coef 0
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
    --sglang-enable-multimodal
    # Four TP2 engines, as in the text recipe; prefill/vision remains eager.
    --rollout-num-gpus-per-engine "${SGLANG_TP:-2}"
    --sglang-mem-fraction-static 0.65
    --sglang-attention-backend triton
    --sglang-cuda-graph-backend-decode full
    --sglang-disable-prefill-cuda-graph
    --sglang-cuda-graph-bs-decode 1 2 4 8 16 32 64
    --sglang-disable-overlap-schedule
    --sglang-disable-radix-cache
    --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS:-65536}"
    --sglang-max-running-requests "${SGLANG_CONCURRENCY:-64}"
    --sglang-chunked-prefill-size 4096
    --sglang-context-length 4096
)
MONITOR_ARGS=(
    --use-clearml
    --use-metrics-service
    --tensorboard-dir "${SAVE_DIR}/${EXP_NAME}/tensorboard"
    --rollout-result-dir "${SAVE_DIR}/${EXP_NAME}/rollout_result"
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
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
