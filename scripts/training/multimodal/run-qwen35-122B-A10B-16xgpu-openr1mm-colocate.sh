#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen3.5-122B-A10B full-model OpenR1MM GRPO, one Actor service on
# 2 physical nodes x 8 GPUs, colocated with one TP8 Rollout engine per node.
# Use an external Ray entrypoint. RELAX_RECIPE_CAPTURE=1 only prints arguments.

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${SCRIPT_DIR}/../../models}"
source "${MODEL_CONFIG_DIR}/qwen35-122B-A10B.sh"

WEIGHT_SYNC_MODE="${WEIGHT_SYNC_MODE:-native}"
case "${WEIGHT_SYNC_MODE}" in
    native) SYNC_ARGS=() ;;
    shard-p2p) SYNC_ARGS=(--use-shard-p2p-weight-sync) ;;
    *) echo "WEIGHT_SYNC_MODE must be native or shard-p2p" >&2; exit 2 ;;
esac

EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet}"
MODEL_PATH="${MODEL_DIR}/Qwen3.5-122B-A10B"
SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)
ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type openr1mm
    --num-rollout 10
    --rollout-batch-size 4
    --n-samples-per-prompt 2
    --rollout-max-response-len 2048
    --rollout-max-prompt-len 2048
    --rollout-temperature 1
    --global-batch-size 8
    --use-streaming-dataset
    --balance-data
    --use-fault-tolerance
    --system-prompt "${SYSTEM_PROMPT}"
    --multimodal-keys '{"image":"image"}'
    --no-rope-fusion
)
PERF_ARGS=(
    --actor-num-nodes 2
    --actor-num-gpus-per-node 8
    --num-gpus-per-node 8
    --rollout-num-gpus 16
    --tensor-model-parallel-size 8
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --calculate-per-token-loss
    --context-parallel-size 1
    --expert-model-parallel-size 16
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
)
GRPO_ARGS=(
    --advantage-estimator grpo
    --true-on-policy-mode
    --kl-loss-coef 0.00
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
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --use-rollout-routing-replay
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-ep-size 1
    --sglang-mem-fraction-static 0.8
)
MISC_ARGS=(
    --selective-offload
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)
TRAIN_ARGS=(
    python3 -m relax.entrypoints.train
    # One 16-GPU Actor service and one 16-GPU Rollout service share the same
    # placement group. --actor-num-nodes above describes physical layout.
    --resource '{"actor": [1, 16], "rollout": [1, 16]}'
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${SYNC_ARGS[@]}"
)

# This common C00 setting is identical in both arms. Other inherited runtime
# settings, including NCCL and polling, must come from the same entrypoint.
if [[ "${RELAX_RECIPE_CAPTURE:-0}" == "1" ]]; then
    python3 - "${WEIGHT_SYNC_MODE}" "${TRAIN_ARGS[@]}" <<'PY'
import json
import os
import sys

env = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
selected = {
    name: env.get("env_vars", {}).get(name)
    for name in ("NCCL_BUFFSIZE", "NCCL_MAX_CTAS", "NCCL_NVLS_ENABLE")
}
print(json.dumps({
    "mode": sys.argv[1],
    "train_argv": sys.argv[2:],
    "common_runtime_env": {"RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP": "1", **selected},
}, ensure_ascii=False, indent=2))
PY
    exit 0
fi

# Never fall through to local.sh: it can stop the current Ray cluster.
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" || -z "${RUNTIME_ENV_JSON:-}" || -z "${RAY_DASHBOARD:-}" ]]; then
    echo "An external entrypoint must set RELAX_ENTRYPOINT_MODE, RUNTIME_ENV_JSON and RAY_DASHBOARD" >&2
    exit 2
fi
if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/model.safetensors.index.json" || ! -f "${PROMPT_SET}" ]]; then
    echo "Model config/index and OpenR1MM prompt data must be present before launch" >&2
    exit 2
fi

RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

env = json.loads(os.environ["RUNTIME_ENV_JSON"])
env.setdefault("env_vars", {})["RELAX_COLOCATE_OVERLAP_WEIGHT_BACKUP"] = "1"
print(json.dumps(env))
PY
)
export RUNTIME_ENV_JSON

RAY_ARGS=(--address="${RAY_DASHBOARD}" --runtime-env-json="${RUNTIME_ENV_JSON}")
if [[ -n "${WORKING_DIR:-}" ]]; then RAY_ARGS+=(--working-dir "${WORKING_DIR}"); fi
if [[ -n "${RAY_NO_WAIT:-}" ]]; then RAY_ARGS+=(--no-wait); fi
mkdir -p log
ray job submit "${RAY_ARGS[@]}" -- "${TRAIN_ARGS[@]}" 2>&1 | tee "log/qwen35-122B-A10B-openr1mm-${WEIGHT_SYNC_MODE}-$(date +%Y%m%d-%H%M%S).log"
