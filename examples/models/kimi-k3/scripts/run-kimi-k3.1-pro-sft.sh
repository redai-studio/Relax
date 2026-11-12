#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Kimi K3.1 full-parameter SFT. Configure paths, data mode and parallelism via env.
# Submit through scripts/entrypoint/ray-job.sh; DRY_RUN=1 only prints arguments.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-${RELAX_ROOT}/scripts/models}"
source "${MODEL_CONFIG_SCRIPT:-${MODEL_CONFIG_DIR}/kimi-k3.1-pro.sh}"

: "${MODEL_DIR:?Set MODEL_DIR to the shared Kimi-K3.1-Pro HF model directory}"
: "${DATA_DIR:?Set DATA_DIR to the shared dataset directory}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent shared checkpoint root}"
PROMPT_DATA="${PROMPT_DATA:-${DATA_DIR}/${TRAIN_FILE:-train.parquet}}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/kimi-k3.1-pro}"
EXP_NAME="${EXP_NAME:-kimi-k3.1-pro-sft}"
NUM_STEPS="${NUM_ROLLOUT-${NUM_STEPS-1000}}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-1024}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-0}"
export RAW_TEXT="${RAW_TEXT:-0}"
NUM_GPUS="${NUM_GPUS:-128}"
export RELAX_SFT_EVAL_FIRST_STEP="${SFT_EVAL_FIRST_STEP:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
now=$(date "+%Y-%m-%d-%H-%M-%S")

# Clear inherited text-only rendering so SFT uses multimodal chat rendering.
if [ "${RAW_TEXT}" = 1 ]; then
    export RELAX_SFT_RAW_TEXT_CONCAT=1
else
    unset RELAX_SFT_RAW_TEXT_CONCAT
fi

# Preserve the K3 recipe transport when DeepEP is explicitly enabled.
# The default all-to-all dispatcher does not use NVSHMEM.
export NVSHMEM_REMOTE_TRANSPORT="${NVSHMEM_REMOTE_TRANSPORT:-libfabric}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --trust-remote-code
    --ref-load "${MODEL_DIR}"
    --megatron-to-hf-mode bridge
    --load "${LOAD_DIR:-${SAVE_DIR}/${EXP_NAME}}"
    --use-checkpoint-opt-param-scheduler
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-100}"
    --max-actor-ckpt-to-keep "${MAX_CKPT_TO_KEEP:-1}"
    --num-epoch "${NUM_EPOCH:-1000}"
)
# Leave epoch-based training unchanged unless an explicit step limit is set.
if [ -n "${NUM_STEPS}" ]; then
    CKPT_ARGS+=(--num-rollout "${NUM_STEPS}")
fi
# Save optimizer state by default so the stable load/save directory can resume.
# SAVE_OPTIMIZER=0 opts into model-only checkpoints, which cannot fully resume.
if [ "${SAVE_OPTIMIZER:-1}" != 1 ]; then
    CKPT_ARGS+=(--no-save-optim)
fi

SFT_ARGS=(
    --loss-type sft
    --prompt-data "${PROMPT_DATA}"
    --input-key "${INPUT_KEY:-messages}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-512}"
    # dataloader.shuffle: true in the recipe.
    --rollout-shuffle
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --sft-oversize-strategy skip
    --sft-invalid-multimodal-strategy "${INVALID_MULTIMODAL_STRATEGY:-error}"
    --balance-data
    --per-rank-fetch
    --sft-prefetch-num-workers 16
    --sft-prefetch-buffer-size 256
    --sft-tq-timeout-minutes 60
)

if [ "${RAW_TEXT}" != 1 ]; then
    if [ -z "${MULTIMODAL_KEYS:-}" ]; then
        MULTIMODAL_KEYS='{"image":"images"}'
    fi
    SFT_ARGS+=(
        --multimodal-keys "${MULTIMODAL_KEYS}"
        --image-max-token-num "${IMAGE_MAX_TOKENS:-1024}"
    )
fi
LORA_ARGS=()
if [ "${KIMI_LORA:-0}" = 1 ]; then
    CKPT_ARGS+=(--save-lora-only)
    LORA_ARGS=(
        --lora-rank "${LORA_RANK:-16}"
        --lora-alpha "${LORA_ALPHA:-32}"
        --lora-dropout "${LORA_DROPOUT:-0.0}"
        --lora-scope all
        --lora-merge-mode
        --lora-target-modules q_proj k_proj v_proj o_proj f_a_proj f_b_proj b_proj g_proj q_a_proj q_b_proj kv_a_proj_with_mqa kv_b_proj linear_fc1 linear_fc2 fc1_latent_proj fc2_latent_proj
    )
fi

# Use the independent held-out file; never split already-consumed training rows.
EVAL_ARGS=()
if [ "${EVAL_INTERVAL}" != 0 ]; then
    EVAL_ARGS=(
        --eval-prompt-data "${EVAL_NAME:-validation}" "${DATA_DIR}/${EVAL_FILE:?Set EVAL_FILE when EVAL_INTERVAL is enabled}"
        --eval-interval "${EVAL_INTERVAL}"
    )
fi

# MoE token dispatcher. Default = plain all-to-all (no NVSHMEM, EFA-safe).
if [ "${USE_DEEPEP:-0}" = 1 ]; then
    DISPATCHER_ARGS=(
        --moe-flex-dispatcher-backend deepep
        --moe-token-dispatcher-type flex
    )
else
    DISPATCHER_ARGS=(
        --moe-token-dispatcher-type alltoall
    )
fi

# Trainable vision uses per-layer activation checkpointing; vision DP requires a frozen tower.
VISION_ARGS=()
if [ "${FREEZE_VISION_TOWER}" = 1 ]; then
    VISION_ARGS=(
        --vision-dp-when-tp
        --freeze-vision-model
    )
fi

if [ "${FREEZE_VISION_PROJECTION:-${RAW_TEXT}}" = 1 ]; then
    VISION_ARGS+=(--freeze-vision-projection)
fi

PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE:-4}"
    --sequence-parallel
    "${VISION_ARGS[@]}"
    --pipeline-model-parallel-size "${PP_SIZE:-4}"
    --decoder-first-pipeline-num-layers "${PP_FIRST_LAYERS:-21}"
    --decoder-last-pipeline-num-layers "${PP_LAST_LAYERS:-24}"
    # Reuse the K3 128-GPU layout; validate K3.1 DSA on the target cluster.
    --context-parallel-size "${CP_SIZE:-8}"
    --linear-cp-mode headwise
    --cp-partition-mode zigzag
    --data-pad-size-multiplier 128
    --calculate-per-token-loss
    # 32 GPUs per PP stage (TP4*CP8*DP1) -> EP32, 896/32 = 28 experts/rank.
    --expert-model-parallel-size "${EP_SIZE:-32}"
    --expert-tensor-parallel-size "${ETP_SIZE:-1}"

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --main-grads-dtype "${MAIN_GRADS_DTYPE:-bf16}"
    --optimizer-offload-fraction "${OPTIMIZER_OFFLOAD_FRACTION:-0.8}"

    --disable-weights-backuper
    --cross-entropy-loss-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size "${SFT_LOGITS_CHUNK_SIZE:-256}"

    "${DISPATCHER_ARGS[@]}"
)

# Matches the recipe's AdamW block: lr 1e-5, weight_decay 0.0, betas [0.9, 0.95].
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --lr-decay-style cosine
    --min-lr "${MIN_LR:-1e-6}"
    --weight-decay "${WEIGHT_DECAY:-0.0}"
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
)

if [ -n "${LR_DECAY_STEPS:-}" ]; then
    OPTIMIZER_ARGS+=(--lr-decay-iters "${LR_DECAY_STEPS}")
fi

METRICS_ARGS=(
    --use-clearml
    --use-metrics-service
    --use-tensorboard
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${EXP_NAME}-${now}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend "${ATTENTION_BACKEND:-flash}"
    --dsa-kernel-backend tilelang
    # rng.seed: 1234 in the recipe.
    --seed "${SEED:-1234}"
)

# Merge recipe defaults without discarding caller-provided worker settings.
export TRAIN_ENV_VARS
TRAIN_ENV_VARS=$(python3 - <<'PYENV'
import json, os
values = {
    "FLA_TILELANG": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "RELAX_KIMI_K31_EXPANDED_DSA_BACKEND": "core", "MEGATRON_SYNC_SAVE_BOUNDED_STAGING": "1",
    "RELAX_KIMI_K3_FUSED_SITU": "1", "RELAX_KIMI_K3_FUSED_ATTN_RES": "1",
    "RELAX_KIMI_K31_DSA_SELECTION_REUSE": "0", "RELAX_KIMI_K31_DSA_FUSED_INDEXER": "0",
    "RELAX_KIMI_K31_DSA_QUERY_CHUNK_SIZE": "8192", "RELAX_KIMI_K3_EXPERT_SPLIT": "1",
}
values = {key: os.environ.get(key, default) for key, default in values.items()}
values.update(json.loads(os.environ.get("TRAIN_ENV_VARS") or "{}"))
if os.environ.get("KIMI_LORA") == "1":
    values["RELAX_LORA_SHARE_EXPERT_ADAPTERS"] = "false"
print(json.dumps(values))
PYENV
)

TRAIN_ARGS=(
    --train-env-vars "${TRAIN_ENV_VARS}"
    --resource "{\"sft\": [1, 0], \"actor\": [1, ${NUM_GPUS}]}"
    --sft-max-in-flight-steps 1
    --num-data-storage-units 16
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${LORA_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${EVAL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${METRICS_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "$@"
)

if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

: "${RAY_DASHBOARD_ADDRESS:?Set RAY_DASHBOARD_ADDRESS to the Ray Jobs HTTP address}"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
# Prepare recipe defaults even when the outer entrypoint ran before this script.
source "${RELAX_ROOT}/examples/models/kimi-k3/tools/configure_kernel_cache.sh" "$@"

# Propagate the selected ClearML config and CPU thread limits to detached workers.
# Keep multimodal rendering enabled throughout the worker hierarchy.
if [ -n "${RUNTIME_ENV_JSON:-}" ]; then
    RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
if os.environ.get("RAW_TEXT") == "1":
    env["RELAX_SFT_RAW_TEXT_CONCAT"] = "1"
else:
    env.pop("RELAX_SFT_RAW_TEXT_CONCAT", None)
if os.environ.get("KIMI_LORA") == "1":
    env["RELAX_LORA_SHARE_EXPERT_ADAPTERS"] = "false"
worker_vars = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CLEARML_CONFIG_FILE", "RELAX_SFT_EVAL_FIRST_STEP", "RELAX_SFT_RAW_TEXT_CONCAT", "RELAX_LORA_SHARE_EXPERT_ADAPTERS"
)
for name in worker_vars:
    if name in os.environ:
        env[name] = os.environ[name]
propagate = env.get("RELAX_PROPAGATE_ENV_VARS", "").split(",")
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(
    dict.fromkeys(
        name for name in [*propagate, *worker_vars] if name and (name != "RELAX_SFT_RAW_TEXT_CONCAT" or os.environ.get("RAW_TEXT") == "1")
    )
)
print(json.dumps(runtime))
PY
    )
    export RUNTIME_ENV_JSON
fi

# Populate custom HF modules before 128 ranks concurrently import the tokenizer.
# A cache-local lock also covers nodes that share HF_MODULES_CACHE.
# WARMUP_HF_CACHE=0 is only for an already populated cache or argument inspection.
if [ "${WARMUP_HF_CACHE:-1}" = 1 ]; then
    : "${RAY_ADDRESS:?Set RAY_ADDRESS to the same cluster as RAY_DASHBOARD_ADDRESS}"
    python3 - "${MODEL_DIR}" <<'PYWARM'
import json
import os
import sys

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1)
def warmup(model: str) -> str:
    import fcntl
    import socket
    from pathlib import Path

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from transformers.dynamic_module_utils import HF_MODULES_CACHE

    cache = Path(HF_MODULES_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / ".relax-kimi-tokenizer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        AutoConfig.from_pretrained(model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        AutoProcessor.from_pretrained(model, trust_remote_code=True)
        assert tokenizer.encode("Kimi K3.1 validation")
    return f"{socket.gethostname()}: {type(tokenizer).__name__} cache ready"


ray.init(address=os.environ["RAY_ADDRESS"], runtime_env=json.loads(os.environ["RUNTIME_ENV_JSON"]))
try:
    nodes = [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0]
    if not nodes:
        raise RuntimeError("No live GPU nodes for HF cache warmup")
    pending = [
        warmup.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)).remote(sys.argv[1])
        for node in nodes
    ]
    for result in ray.get(pending, timeout=300):
        print(result)
finally:
    ray.shutdown()
PYWARM
fi

mkdir -p "${SCRIPT_DIR}/log"
ray job submit \
    ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_DASHBOARD_ADDRESS}" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
