#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# One native M-BEIR retrieval SFT epoch from a prepared PT Retrieve checkpoint.

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the prepared PT Retrieve HF checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to the native-sampling M-BEIR input root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the Ray Jobs HTTP address}"
# The controller must retire only its own preceding job; never purge a cluster.
export RELAX_SKIP_CLUSTER_CLEANUP=1
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_LORA_A_INIT_METHOD="${RELAX_LORA_A_INIT_METHOD:-kaiming}"
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime.setdefault("env_vars", {})["RELAX_LORA_A_INIT_METHOD"] = os.environ["RELAX_LORA_A_INIT_METHOD"]
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON
HF_CHECKPOINT="${MODEL_DIR}"
source "${RELAX_ROOT}/examples/seq_contrast_sft/scripts/model.sh"
read -r DATASET_STEPS WARMUP_STEPS GLOBAL_BATCH_SIZE < <(
    python3 - "${DATA_DIR}/mbeir/pairs.meta.json" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as stream:
    data = json.load(stream)
if data["stage"] != "prepared" or data["sampling"] != "native_runtime":
    raise SystemExit("Prepare M-BEIR pairs with prepare_mbeir.py")
queries, batch = data["raw_queries"], data["global_batch_size"]
if queries <= 0 or batch <= 0 or data["prepared_rows"] != queries:
    raise SystemExit("Invalid M-BEIR pair or batch count")
steps = math.ceil(queries / batch)
print(steps, math.ceil(steps * 0.03), batch)
PY
)
EXP_NAME="${EXP_NAME:-${LAMRA_MODEL_NAME}-mbeir-from-pt-r128-dp8}"
now=$(date "+%Y%m%d-%H%M%S")

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval 200
    --max-actor-ckpt-to-keep 2
)
SFT_ARGS=(
    --loss-type sft --task-type embedding --embedding-loss-type infonce --contrastive-temperature 0.05
    --custom-dataset-class examples.seq_contrast_sft.mbeir.MbeirPairDataset
    --prompt-data "${DATA_DIR}/mbeir/pairs.jsonl"
    --global-batch-size "${GLOBAL_BATCH_SIZE}" --n-samples-per-prompt 1
    --num-rollout "${DATASET_STEPS}" --num-steps-per-rollout 1
    --seq-length 1024 --use-dynamic-batch-size --max-tokens-per-gpu 131072
    --sft-prefetch-num-workers 32 --sft-prefetch-chunk-size 64 --sft-prefetch-buffer-size 512
    --image-min-token-num 4 --image-max-token-num 300
    --sft-oversize-strategy keep --per-rank-fetch
)
LORA_ARGS=(
    --lora-rank 128 --lora-alpha 256 --lora-dropout 0.05
    --lora-target-modules "${LAMRA_LORA_TARGETS[@]}"
    --lora-scope language --lora-merge-mode
    --freeze-vision-model --freeze-vision-projection
)
OPTIMIZER_ARGS=(
    --optimizer adam --lr 1e-4 --lr-decay-style cosine --min-lr 0
    --lr-warmup-iters "${WARMUP_STEPS}" --weight-decay 0 --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-8 --clip-grad 1
)
PERF_ARGS=(
    --bf16 --seed 42 --data-parallel-random-init
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1
    --qkv-format thd --use-gloo-process-groups
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --no-rope-fusion --attention-dropout 0 --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
)
TRACKING_ARGS=(--use-metrics-service --no-use-tensorboard)
RUN_LOG_DIR="${RELAX_RUN_LOG_DIR:-${SAVE_DIR}/logs}"
mkdir -p "${RUN_LOG_DIR}"
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="${RAY_ADDRESS}" \
    \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft": [1, 0], "actor": [1, 8]}' \
    --sft-max-in-flight-steps 2 --num-data-storage-units 8 \
    "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SFT_ARGS[@]}" "${LORA_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" "${PERF_ARGS[@]}" "${TRACKING_ARGS[@]}" "$@" \
    2>&1 | tee "${RUN_LOG_DIR}/${EXP_NAME}-${now}.log"
