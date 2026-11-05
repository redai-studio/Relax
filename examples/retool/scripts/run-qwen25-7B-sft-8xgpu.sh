#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Single node: MODEL_DIR=/path/to/hf DATA_DIR=/path/to/data SAVE_DIR=/path/to/checkpoints bash <this-script>
# Existing cluster: bash scripts/entrypoint/ray-job.sh <this-script>
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${MODEL_DIR:?Set the initial Hugging Face model directory}"
: "${DATA_DIR:?Set the prepared ReTool data directory}"
: "${SAVE_DIR:?Set a persistent checkpoint root}"
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "Missing HF model config: ${MODEL_DIR}/config.json" >&2; exit 1; }
[[ -f "${DATA_DIR}/sft/rows.jsonl" ]] || { echo "Missing prepared ReTool data: ${DATA_DIR}/sft/rows.jsonl" >&2; exit 1; }
export RELAX="${RELAX_ROOT}" RELAX_SKIP_CLUSTER_CLEANUP=1
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    # local.sh reads optional environment variables without nounset guards.
    set +u
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
    set -u
fi
source "${RELAX_ROOT}/scripts/models/qwen25-7B.sh"
PROJECT_NAME="${PROJECT_NAME:-Relax/retool}"
EXP_NAME="${EXP_NAME:-qwen25-7b-retool-sft-8xgpu}"
now=$(date +%Y%m%d-%H%M%S)
export DATA_DIR SAVE_DIR
export PYTHONPATH="${RELAX_ROOT}:${PYTHONPATH:-}"
RUNTIME_ENV_JSON=$(python3 - <<'ENV'
import json
import os
value = json.loads(os.environ["RUNTIME_ENV_JSON"])
names = ["DATA_DIR", "SAVE_DIR", "SANDBOX_FUSION_URL", "SANDBOX_IMAGE", "RELAX_PROPAGATE_ENV_VARS"]
value.setdefault("env_vars", {})["PYTHONPATH"] = os.environ["PYTHONPATH"]
value["env_vars"].update({name: os.environ[name] for name in names if name in os.environ})
print(json.dumps(value))
ENV
)
START_ROLLOUT_ID=$(python3 -m examples.retool.checkpoint "${SAVE_DIR}/${EXP_NAME}" --sft)
CKPT_ARGS=(
    --disable-s3-model-download --hf-checkpoint "${MODEL_DIR}" --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}" --save "${SAVE_DIR}/${EXP_NAME}"
    --start-rollout-id "${START_ROLLOUT_ID}"
    --use-checkpoint-opt_param-scheduler
    --save-interval 25 --max-actor-ckpt-to-keep 2 --save-hf "${SAVE_DIR}/${EXP_NAME}/hf_{rollout_id}"
)
SFT_ARGS=(
    --loss-type sft --prompt-data "${DATA_DIR}/sft/rows.jsonl" --input-key author_masked
    --custom-dataset-class examples.retool.sft_dataset.AuthorMaskedSFTDataset
    --num-rollout 50 --rollout-batch-size 80 --global-batch-size 80 --n-samples-per-prompt 1
    --num-steps-per-rollout 1 --num-iters-per-train-update 1 --seed 20261102
    --seq-length 16384 --use-dynamic-batch-size --max-tokens-per-gpu 16384 --micro-batch-size 1
    --sft-oversize-strategy keep --balance-data --sft-prefetch-num-workers 1
    --resource '{"sft":[1,0],"actor":[1,8]}' --max-staleness 1 --num-data-storage-units 1
)
PERF_ARGS=(
    --weight-decay 0 --data-parallel-sharding-strategy optim
    --cross-entropy-fusion-impl te --accumulate-allreduce-grads-in-fp32
    --recompute-loss-function --log-probs-chunk-size 256 --skip-eval-before-train
    --use-distributed-optimizer --bf16
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 --context-parallel-size 1 --sequence-parallel
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --attention-dropout 0 --hidden-dropout 0 --attention-backend flash
)
OPTIMIZER_ARGS=(
    --optimizer adam --lr 1e-6 --lr-decay-style constant
    --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1
)
TRACKING_ARGS=(
    --use-clearml --use-metrics-service --tb-project-name "${PROJECT_NAME}" --tb-experiment-name "${EXP_NAME}-${now}"
)
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" "$@" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
