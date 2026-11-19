#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# One physical GPU: Actor -> Rollout -> deferred GenRM -> OPD Teacher.
# Uses archived PR2 genrm-train.jsonl/genrm-eval.jsonl; never rewrites samples.
# MODEL_DIR must contain Qwen3-0.6B (official source: Qwen/Qwen3-0.6B).
# Usage: MODEL_DIR=... DATA_DIR=... SAVE_DIR=... bash "$0"
# PREPARE_ONLY=1 validates existing data and writes the eval configuration only.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the model root}"
: "${DATA_DIR:?Set DATA_DIR to the existing dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
PROJECT_NAME="${PROJECT_NAME:-Relax/dev/pr381-shared-inference}"
EXP_NAME="${EXP_NAME:-qwen3-0.6b-1xgpu-shared-inference}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/genrm-train.jsonl}"
EVAL_DATA="${EVAL_DATA:-${DATA_DIR}/genrm-eval.jsonl}"
EVAL_CONFIG="${SAVE_DIR}/${EXP_NAME}-eval.yaml"
now=$(date +%Y%m%d-%H%M%S)
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

# Only configuration is generated. Keep the previously used samples unchanged.
python - "${TRAIN_DATA}" "${EVAL_DATA}" "${EVAL_CONFIG}" <<'PY'
import json
from pathlib import Path
import sys

train, evaluation, config = map(Path, sys.argv[1:])
for path in (train, evaluation):
    if not path.is_file():
        raise FileNotFoundError(f"Expected an existing dataset: {path}")
    with path.open() as stream:
        first = next((line for line in stream if line.strip()), None)
    if first is None:
        raise ValueError(f"Dataset is empty: {path}")
    row = json.loads(first)
    if "prompt" not in row or "label" not in row:
        raise ValueError(f"Dataset requires prompt and label fields: {path}")
config.parent.mkdir(parents=True, exist_ok=True)
# JSON is valid YAML. Eval uses the real math verifier, not the training dummy RM.
config.write_text(json.dumps({"eval": {"datasets": [
    {"name": "validation", "path": str(evaluation), "rm_type": "dapo"}
]}}, indent=2) + "\n")
PY
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
    exit 0
fi

if [[ ! -f "${MODEL_DIR}/Qwen3-0.6B/config.json" ]]; then
    echo "Missing model config: ${MODEL_DIR}/Qwen3-0.6B/config.json" >&2
    exit 1
fi
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
source "${SCRIPT_DIR}/../../models/qwen3-0.6B.sh"

# Requires the existing Megatron sync-save-bounded-staging image patch.
# Offloaded short runs save without retaining an unbounded preload staging pool.
export MEGATRON_SYNC_SAVE_BOUNDED_STAGING="${MEGATRON_SYNC_SAVE_BOUNDED_STAGING:-1}"
export MEGATRON_SYNC_SAVE_STAGE_BYTES="${MEGATRON_SYNC_SAVE_STAGE_BYTES:-0}"
export RUNTIME_ENV_JSON="$(python - <<'PYENV'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime.setdefault("env_vars", {}).update({
    key: os.environ[key]
    for key in ("MEGATRON_SYNC_SAVE_BOUNDED_STAGING", "MEGATRON_SYNC_SAVE_STAGE_BYTES")
})
print(json.dumps(runtime))
PYENV
)"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval 1
    --max-actor-ckpt-to-keep 2
)
ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking":false}'
    --num-rollout 2
    --rollout-batch-size 2
    --n-samples-per-prompt 2
    --global-batch-size 4
    --micro-batch-size 1
    --rollout-max-prompt-len 512
    --rollout-max-response-len 128
    --rollout-max-context-len 1024
    --rollout-temperature 1
    --num-iters-per-train-update 1
)
SCORING_ARGS=(
    --rm-type dummy
    --reward-key score
    --defer-reward-to-post-process
    --custom-reward-post-process-path examples.generate_reward_model.post_process_genrm_swap.custom_reward_post_process
    --genrm-model-path "${MODEL_DIR}/Qwen3-0.6B"
    --genrm-num-gpus 1
    --genrm-num-gpus-per-engine 1
    --genrm-engine-config '{"context_length":1024,"mem_fraction_static":0.5,"disable_cuda_graph":true}'
    --genrm-sampling-config '{"temperature":0,"max_response_len":32,"chat_template_kwargs":{"enable_thinking":false}}'
    --use-opd
    --opd-type sglang
    --opd-token-selection student_sampled
    --teacher-hf-checkpoint "${MODEL_DIR}/Qwen3-0.6B"
    --teacher-num-gpus-per-engine 1
    --teacher-sglang-mem-fraction-static 0.5
    --teacher-sglang-context-length 1024
    --teacher-sglang-enable-weights-cpu-backup
    --teacher-sglang-disable-cuda-graph
    --opd-kl-coef 0.1
    --opd-loss-coef 0
)
EVAL_ARGS=(
    --skip-eval-before-train
    --eval-interval 1
    --eval-config "${EVAL_CONFIG}"
    --n-samples-per-eval-prompt 1
    --eval-max-prompt-len 512
    --eval-max-response-len 128
    --log-passrate
)
TRAIN_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 1024
    --recompute-granularity selective
    --advantage-estimator grpo
    --kl-coef 0
    --kl-loss-coef 0
    --entropy-coef 0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0
    --adam-beta1 0.9
    --adam-beta2 0.98
    --attention-dropout 0
    --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.5
    --sglang-context-length 1024
    --sglang-disable-cuda-graph
)
TRACKING_ARGS=(
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)
if [[ "${USE_CLEARML:-1}" == 1 ]]; then
    TRACKING_ARGS+=(
        --use-clearml
    )
fi
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    --submission-id "${EXP_NAME}-${now}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor":[1,1],"rollout":[1,1],"genrm":[1,1],"teacher":[1,1],"advantages":[1,0]}' \
    --num-gpus-per-node 1 \
    --actor-num-gpus-per-node 1 \
    --rollout-num-gpus 1 \
    --no-enable-affinity \
    --colocate \
    --offload \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${SCORING_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "$@" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
