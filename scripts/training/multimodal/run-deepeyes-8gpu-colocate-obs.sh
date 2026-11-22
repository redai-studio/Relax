#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Forty-rollout Qwen3.5-9B colocated smoke run.
# Actor (TP4, DP2) and rollout (four TP2 engines) time-share eight GPUs.
# RBS4 * N4 / GBS8 = two optimizer updates per rollout.
# The existing 1.5B judge shares GPU memory with training (no ninth GPU).
# No partial rollout, routing replay, periodic eval, or checkpoint saving.
# DRY_RUN=1 prints arguments and checks input files without starting services.
# Run locally on eight GPUs, or through scripts/entrypoint/ray-job.sh on an existing cluster.
# Keep this launcher attached: its EXIT trap owns the local judge service.

set -e
set -o pipefail

###############################################################################
#                                 ENVIRONMENT                                 #
###############################################################################

TIMESTAMP=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
AGENT_DIR="${RELAX_ROOT}/examples/deepeyes_agentic"
export MEGATRON=/root/Megatron-LM
if [ "${DRY_RUN:-0}" != 1 ] && [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/local.sh"
fi
# This launcher owns the judge; keep ray job submit attached until training exits.
unset RAY_NO_WAIT
cd "${RELAX_ROOT}"
source "${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}/qwen35-9B.sh"

###############################################################################
#                                    DIRS                                     #
###############################################################################

PROJECT_NAME="${PROJECT_NAME:-Relax/agentic-obs}"
EXP_NAME="${EXP_NAME:-qwen35-deepeyes-8gpu-colocate-smoke}"

# Reuse the prepared model and DeepEyes datasets from MODEL_DIR/DATA_DIR.
if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, and SAVE_DIR must be set."
    echo "Example: MODEL_DIR=/path/to/models DATA_DIR=/path/to/data SAVE_DIR=/path/to/save bash $0"
    exit 1
fi
OBS_DIR="${OBS_DIR:-${SAVE_DIR}/agentic_obs/${EXP_NAME}}"
for required_file in \
    "${MODEL_DIR}/Qwen3.5-9B/config.json" \
    "${MODEL_DIR}/Qwen2.5-1.5B-Instruct/config.json" \
    "${DATA_DIR}/deepeyes-v1/data_0.1.2_visual_toolbox_v2.parquet" \
    "${DATA_DIR}/deepeyes-v1/data_v0.8_visual_toolbox_v2.parquet"; do
    if [ ! -s "${required_file}" ]; then
        echo "Missing required file: ${required_file}" >&2
        exit 1
    fi
done


###############################################################################
#                              JUDGE MODEL API                                #
###############################################################################

# The judge is started below, after argument and dry-run checks.

###############################################################################
#                                  MODEL CONFIG                               #
###############################################################################

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3.5-9B"
    --ref-load "${MODEL_DIR}/Qwen3.5-9B"
    --megatron-to-hf-mode bridge
)

###############################################################################
#                                  DATASETS                                   #
###############################################################################

TRAIN_FILES=(
    "'${DATA_DIR}/deepeyes-v1/data_0.1.2_visual_toolbox_v2.parquet@[0:64]'"
    "'${DATA_DIR}/deepeyes-v1/data_v0.8_visual_toolbox_v2.parquet@[0:64]'"
)
PROMPT_SET="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

###############################################################################
#                               ROLLOUT CONFIG                                #
###############################################################################

# Pass only one limit: NUM_EPOCH takes precedence over NUM_ROLLOUT.
if [ -n "${NUM_EPOCH:-}" ]; then
    if [[ ! "${NUM_EPOCH}" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_EPOCH must be a positive integer." >&2
        exit 1
    fi
    RUN_LENGTH_ARGS=(--num-epoch "${NUM_EPOCH}")
    RUN_LENGTH_LABEL="${NUM_EPOCH} epochs"
else
    NUM_ROLLOUT="${NUM_ROLLOUT:-40}"
    if [[ ! "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_ROLLOUT must be a positive integer." >&2
        exit 1
    fi
    RUN_LENGTH_ARGS=(--num-rollout "${NUM_ROLLOUT}")
    RUN_LENGTH_LABEL="${NUM_ROLLOUT} rollouts"
fi
printf -v AGENT_COMMAND '. %q' "${AGENT_DIR}/run_agent_app.sh"

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key reward_model
    --multimodal-keys '{"image":"images"}'
    --reward-key score
    --metadata-key extra_info
    --custom-rm-path examples.deepeyes_agentic.reward_deepeyes.reward_func
    --use-agentic-rollout
    --agent-command "${AGENT_COMMAND}"
    --agent-cwd "${AGENT_DIR}"
    "${RUN_LENGTH_ARGS[@]}"
    --rollout-batch-size 4
    --micro-batch-size 1
    --n-samples-per-prompt 4
    --rollout-max-response-len 2048
    --rollout-max-prompt-len 2048
    --rollout-max-context-len 8192
    --rollout-temperature 1
    --global-batch-size 8
    --use-fault-tolerance
    --rollout-shuffle
    --use-streaming-dataset
)


###############################################################################
#                                EVAL CONFIG                                  #
###############################################################################

EVAL_ARGS=(--skip-eval-before-train)

###############################################################################
#                              ALGORITHM CONFIG                               #
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --eps-clip-c 3
    --use-tis
)

###############################################################################
#                              OPTIMIZER CONFIG                               #
###############################################################################

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
    --no-rope-fusion
)

###############################################################################
#                               SGLANG CONFIG                                 #
###############################################################################

SGLANG_ARGS=(--sglang-mem-fraction-static 0.6)

###############################################################################
#                               LOGGING CONFIG                                #
###############################################################################

LOG_ARGS=(
    --use-clearml
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${TIMESTAMP}"
    --dump-agentic-obs "${OBS_DIR}"
)

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################

MEGATRON_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --max-tokens-per-gpu 9216
    --log-probs-max-tokens-per-gpu 8192
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-dynamic-batch-size
)

###############################################################################
#                              RESOURCE CONFIG                                #
###############################################################################

RAY_RESOURCE_ARGS=(
    --rollout-num-gpus-per-engine 2
    --resource '{"actor": [1, 8], "rollout": [1, 8]}'
    --max-staleness 0
    --num-data-storage-units 1
    --num-iters-per-train-update 1
    --colocate
    --no-enable-affinity
)

###############################################################################
#                                 LAUNCH JOB                                  #
###############################################################################

TRAIN_ARGS=(
    "${RAY_RESOURCE_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${LOG_ARGS[@]}"
    "${MEGATRON_ARGS[@]}"
    "${EVAL_ARGS[@]}"
)
printf 'Model: Qwen3.5-9B; colocate, actor TP4/DP2 + rollout TP2 sharing 8 GPUs; %s\n' "${RUN_LENGTH_LABEL}"
printf 'SAVE_DIR=%s\nOBS_DIR=%s\n' "${SAVE_DIR}" "${OBS_DIR}"
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi
if [ "${RELAX_ENTRYPOINT_MODE:-}" = "local" ]; then
    export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
fi
: "${RAY_ADDRESS:?Set RAY_ADDRESS to the existing single-node eight-GPU Ray cluster.}"
: "${RUNTIME_ENV_JSON:?Run through local.sh or scripts/entrypoint/ray-job.sh.}"
mkdir -p "${SAVE_DIR}" "${OBS_DIR}" logs
python3 - "${SAVE_DIR}/train_args.json" "${TRAIN_ARGS[@]}" <<'PY_ARGS'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:], indent=2) + "\n", encoding="utf-8")
PY_ARGS
# The original DeepEyes helper starts a TP1 judge with mem-fraction-static=0.05.
# It shares a training GPU and its EXIT trap owns cleanup. Do not submit detached.
# All model and judge traffic in this single-node run is local/private.
export NO_PROXY="*" no_proxy="*"
source "${AGENT_DIR}/sglang_judge_service.sh"
RUNTIME_ENV_JSON="$(python3 - <<'PY_ENV'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
names = ("DEEPEYES_JUDGE_API_KEY", "DEEPEYES_JUDGE_BASE_URL", "DEEPEYES_JUDGE_MODELS", "NO_PROXY", "no_proxy")
for name in names:
    env[name] = os.environ[name]
forwarded = env.get("RELAX_PROPAGATE_ENV_VARS", os.environ.get("RELAX_PROPAGATE_ENV_VARS", "")).split(",")
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys(filter(None, [*forwarded, *names])))
print(json.dumps(runtime))
PY_ENV
)"
export RUNTIME_ENV_JSON
SUBMIT_ARGS=(job submit --address="${RAY_ADDRESS}" --runtime-env-json="${RUNTIME_ENV_JSON}")
if [ -n "${OBS_RAY_SUBMISSION_ID:-}" ]; then SUBMIT_ARGS+=(--submission-id "${OBS_RAY_SUBMISSION_ID}"); fi
if [ -n "${WORKING_DIR:-}" ]; then SUBMIT_ARGS+=(--working-dir "${WORKING_DIR}"); fi
command ray "${SUBMIT_ARGS[@]}" -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SAVE_DIR}/train.log"
