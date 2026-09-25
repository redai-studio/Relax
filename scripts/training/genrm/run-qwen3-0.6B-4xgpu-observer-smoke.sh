#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Real four-GPU Relax DAPO + GRPO + GenRM smoke used as the Task 11 observer
# evidence vehicle. It is the proven native-entrypoint recipe of this machine
# (Qwen3-0.6B, actor DP=2 on two GPUs, one rollout GPU, one GenRM GPU) with an
# opt-in straggler observer bolted on through environment variables only.
#
# Why this topology: the two Megatron actor ranks are two data-parallel replicas
# of the same role, so they form a valid comparison cohort (same TP/PP/CP/chunk,
# differing only in DP). One GPU is deliberately left to the rollout engine and
# one to GenRM, so the run is a genuine end-to-end RL job rather than a
# standalone benchmark.
#
# Required environment:
#   TRAIN_VENV=/absolute/path/to/the-native-training-venv
#   MODEL_PATH=/absolute/path/to/Qwen3-0.6B
#   PROMPT_SET=/absolute/path/to/dapo-math-17k.jsonl
# Optional:
#   SAVE_DIR=/absolute/path/for-smoke-checkpoints
#   NUM_ROLLOUT=<number of rollouts>          (default 2)
#   SAVE_INTERVAL=<optimizer steps per save>  (default 1000, i.e. effectively off)
#   STRAGGLER=1                               (default 0: observer off)
#   OBSERVER_DIR=/absolute/path/for-evidence  (default under /root/autodl-tmp)
#   RELAX_STRAGGLER_WINDOW_S, RELAX_STRAGGLER_WARMUP_WINDOWS,
#   RELAX_STRAGGLER_REPORT_INTERVAL_S, RELAX_STRAGGLER_COLLECTOR_ADDR
#
# The observer is default-off: with STRAGGLER unset this script runs exactly the
# recipe it is derived from. Enabling it changes no training argument; only
# process environment variables, which is what the acceptance protocol needs in
# order to compare an OFF arm and an ON arm of the same job.
#
# Launch through the repository entrypoint, for example:
#   TRAIN_VENV=/path/to/venv MEGATRON=/path/to/Megatron-LM RELAX=$PWD \
#     MODEL_PATH=... PROMPT_SET=... RAY_NO_WAIT=1 STRAGGLER=1 \
#     bash scripts/entrypoint/ray-job.sh scripts/training/genrm/run-qwen3-0.6B-4xgpu-observer-smoke.sh

set -ex
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

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

# Ray's system interpreter owns worker processes, while Transformer Engine and
# the native extensions live in TRAIN_VENV.  Carry its site-packages into the
# submitted runtime environment rather than relying on the submitter's PATH.
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

# ── optional straggler observer ─────────────────────────────────────────────
# Default off. When STRAGGLER=1 the observer variables are added to the Ray
# runtime environment so every worker process sees them; rank 0 becomes the CPU
# collector and the other ranks ship envelopes to it over one-way TCP.
STRAGGLER="${STRAGGLER:-0}"
NOW_FOR_EVIDENCE="$(date "+%Y-%m-%d-%H:%M:%S")"
OBSERVER_DIR="${OBSERVER_DIR:-/root/autodl-tmp/relax-work/task11_evidence/real/${NOW_FOR_EVIDENCE}}"
if [ "${STRAGGLER}" = "1" ]; then
    export RELAX_STRAGGLER_ENABLE=1
    export RELAX_STRAGGLER_COLLECTOR_ADDR="${RELAX_STRAGGLER_COLLECTOR_ADDR:-127.0.0.1:29711}"
    export RELAX_STRAGGLER_OUTPUT_DIR="${OBSERVER_DIR}"
    export RELAX_STRAGGLER_WINDOW_S="${RELAX_STRAGGLER_WINDOW_S:-10.0}"
    export RELAX_STRAGGLER_WARMUP_WINDOWS="${RELAX_STRAGGLER_WARMUP_WINDOWS:-2}"
    export RELAX_STRAGGLER_REPORT_INTERVAL_S="${RELAX_STRAGGLER_REPORT_INTERVAL_S:-10.0}"
    mkdir -p "${OBSERVER_DIR}"
    export RUNTIME_ENV_JSON="$("${TRAIN_PYTHON}" - <<'PYINNER'
import json
import os

runtime_env = json.loads(os.environ["RUNTIME_ENV_JSON"])
env_vars = runtime_env.setdefault("env_vars", {})
for name in (
    "RELAX_STRAGGLER_ENABLE",
    "RELAX_STRAGGLER_COLLECTOR_ADDR",
    "RELAX_STRAGGLER_OUTPUT_DIR",
    "RELAX_STRAGGLER_WINDOW_S",
    "RELAX_STRAGGLER_WARMUP_WINDOWS",
    "RELAX_STRAGGLER_REPORT_INTERVAL_S",
):
    env_vars[name] = os.environ[name]
print(json.dumps(runtime_env, separators=(",", ":")))
PYINNER
)"
    echo "straggler observer enabled: window=${RELAX_STRAGGLER_WINDOW_S}s warmup=${RELAX_STRAGGLER_WARMUP_WINDOWS} collector=${RELAX_STRAGGLER_COLLECTOR_ADDR}"
    echo "straggler evidence directory: ${OBSERVER_DIR}"
fi

source "${MODEL_CONFIG_DIR}/qwen3-0.6B.sh"

NOW=$(date "+%Y-%m-%d-%H:%M:%S")
SAVE_DIR="${SAVE_DIR:-${SCRIPT_DIR}/../../../checkpoints/task11-observer-smoke}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-1000}"
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
    --num-rollout "${NUM_ROLLOUT:-2}"
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
    --max-tokens-per-gpu 4096
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
    --resource '{"actor": [1, 2], "rollout": [1, 1], "advantages": [1, 0], "genrm": [1, 1]}' \
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
    "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen3-0.6b-genrm-smoke-${NOW}.log"
