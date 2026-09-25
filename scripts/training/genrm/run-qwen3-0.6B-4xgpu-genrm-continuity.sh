#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Training-continuity smoke for the GenRM elastic-scaling acceptance: the
# same real DAPO + GenRM path as run-qwen3-0.6B-4xgpu-genrm-smoke.sh, but
# the actor trains on a single GPU so one GPU stays free for the elastic
# GenRM engine.  A sidecar monitor (demos/task4_genrm/e2e_train_continuity.py)
# drives scale_out/scale_in against the running service and asserts that
# training steps, rollouts and reward scoring keep advancing through both
# scaling windows.
#
# Required environment:
#   TRAIN_VENV=/absolute/path/to/the-native-training-venv
#   MODEL_PATH=/absolute/path/to/Qwen3-0.6B
#   PROMPT_SET=/absolute/path/to/dapo-math-17k.jsonl
# Optional:
#   SAVE_DIR=/absolute/path/for-smoke-checkpoints

set -ex
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi

: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-0.6B Hugging Face checkpoint.}"
: "${PROMPT_SET:?Set PROMPT_SET to the DAPO math JSONL file.}"
: "${TRAIN_VENV:?Set TRAIN_VENV to the native Megatron training virtual environment.}"

TRAIN_PYTHON="${TRAIN_VENV}/bin/python"
if [ ! -x "${TRAIN_PYTHON}" ]; then
    echo "TRAIN_VENV has no executable bin/python: ${TRAIN_VENV}" >&2
    exit 2
fi

TRAIN_SITE="$("${TRAIN_PYTHON}" -c 'import site; print(site.getsitepackages()[0])')"
if [ ! -d "${TRAIN_SITE}" ]; then
    echo "Could not resolve training site-packages: ${TRAIN_SITE}" >&2
    exit 2
fi

export RUNTIME_ENV_JSON="$(TRAIN_SITE="${TRAIN_SITE}" "${TRAIN_PYTHON}" - <<'PY'
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
env_vars = runtime_env.setdefault("env_vars", {})
existing = env_vars.get("PYTHONPATH", "")
env_vars["PYTHONPATH"] = f"{os.environ['TRAIN_SITE']}:{existing}" if existing else os.environ["TRAIN_SITE"]
print(json.dumps(runtime_env, separators=(",", ":")))
PY
)"

source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

NOW=$(date "+%Y-%m-%d-%H:%M:%S")
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/../../../checkpoints/task4-genrm-continuity}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
    --save "${SAVE_DIR}"
    --save-interval 1
    --max-actor-ckpt-to-keep 1
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo-genrm
    --reward-key score
    --num-rollout 8
    --rollout-batch-size 4
    --n-samples-per-prompt 2
    --rollout-max-response-len 2048
    --rollout-temperature 0.7
    --global-batch-size 8
    --use-fault-tolerance
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --entropy-coef 0.0
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

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.55
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${TRAIN_PYTHON}" -m relax.entrypoints.train \
    --resource '{"actor": [1, 1], "rollout": [1, 1], "advantages": [1, 0], "genrm": [1, 1]}' \
    --fully-async \
    --num-data-storage-units 1 \
    --num-iters-per-train-update 1 \
    --skip-eval-before-train \
    --genrm-model-path "${MODEL_PATH}" \
    --genrm-num-gpus 1 \
    --genrm-num-gpus-per-engine 1 \
    --genrm-engine-config '{"max_context_len": 3072, "mem_fraction_static": 0.55}' \
    --genrm-sampling-config '{"temperature": 0.1, "top_p": 1.0, "top_k": -1, "max_response_len": 512}' \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen3-0.6b-genrm-continuity-${NOW}.log"
