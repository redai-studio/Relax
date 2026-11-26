#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# GPU profile: 280G Blackwell per GPU.
#
# Qwen3.8-Flash-Next full-model OpenR1MM GRPO, 16/24/32/64 GPUs (280G Blackwell each), colocate.
# WORLD_SIZE is the NODE count, matching spmd-multinode.sh; NUM_GPUS must be 8.
# If WORLD_SIZE is unset or empty, count ALIVE GPU nodes in the current Ray cluster.
# Nodes GPUs  Training (TP/PP/CP/EP/ETP)  Rollout
# 2     16    4/4/1/4/1                  one TP16/DP2/EP16 engine (attention TP8)
# 3     24    4/3/1/8/1                  three TP8/EP8 engines
# 4     32    4/4/1/8/1                  four TP8/EP8 engines
# 8     64    4/4/2/8/1 (DP2)            eight TP8/EP8 engines
# The 32/64-GPU presets require multi-node training validation.
# Requires docker/Dockerfile.cu13 target relax with GPU_ARCH=blackwell on all nodes.
# Use the same Relax checkout on all nodes.
# DRY_RUN=1 WORLD_SIZE=3 MEGATRON=<image-checkout> prints argv/runtime JSON
# without starting Ray or importing GPU dependencies.
#
# Usage (export MODEL_DIR, DATA_DIR, SAVE_DIR and RAY_ADDRESS first):
#   WORLD_SIZE=3 bash scripts/entrypoint/ray-job.sh examples/models/qwen3.8-flash-next/scripts/run-qwen38-flash-next-openr1mm-grpo-blackwell.sh
# TorchJob: launch this script through spmd-multinode.sh on each pod.
# MODEL_DIR: initial Qwen3.8-Flash-Next checkpoint root.
# DATA_DIR: converted multimodal-open-r1-8k-verified dataset root.
# SAVE_DIR: persistent checkpoint root, accessible on all nodes.
# RAY_ADDRESS: existing cluster GCS address; submit from the Ray head via its local dashboard.
# Resume: keep SAVE_DIR and EXP_NAME unchanged; new experiment: change EXP_NAME.
# Default: 200 updates, save every 100 updates, retain two checkpoints.
# Two-update validation: NUM_ROLLOUT=2 (use a separate EXP_NAME).
# Ray submission blocks by default; set RAY_NO_WAIT=1 only for nonblocking submission.
# CPU_OFFLOAD: optimizer CPU offload, 1=on / 0=off; defaults to on for 16 GPUs, off for >16.
# ENABLE_MTP=1 trains the bundled MTP head and enables EAGLE rollout (top-k 1).

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"

# Validate before sourcing an entrypoint or importing GPU dependencies.
if [ "${NUM_GPUS:-8}" != 8 ]; then
    echo "This recipe requires NUM_GPUS=8 per node." >&2
    exit 1
fi
if [ -z "${WORLD_SIZE:-}" ]; then
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "Set WORLD_SIZE to the node count (2, 3, 4, or 8) for DRY_RUN." >&2
        exit 1
    fi
    # Query structured cluster status; include busy nodes, exclude CPU-only/dead nodes.
    WORLD_SIZE=$(ray list nodes --format json | jq -er '
        map(select(.state == "ALIVE" and (.resources_total.GPU // 0) > 0)) |
        if length == 0 then
            error("No ALIVE GPU nodes found in the current Ray cluster")
        elif any(.[]; .resources_total.GPU != 8) then
            error("This recipe requires exactly 8 GPUs on every GPU node")
        else length end
    ')
    echo "Detected WORLD_SIZE=${WORLD_SIZE} GPU nodes from Ray ($((WORLD_SIZE * 8)) GPUs)."
fi
PP_LAYOUT_ARGS=()
ENABLE_MTP="${ENABLE_MTP:-0}"
case "${ENABLE_MTP}" in
    0|1) ;;
    *) echo "ENABLE_MTP must be 0 or 1." >&2; exit 1 ;;
esac
ROLLOUT_TOPOLOGY_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-ep-size 8
)
case "${WORLD_SIZE}" in
    2)
        TOTAL_GPUS=16
        PP_SIZE=4
        CP_SIZE=1
        EP_SIZE=4
        DEFAULT_SAVE_INTERVAL=100
        # Two PP stages share each node: disable P2P PXN relays to avoid lazy
        # transport initialization deadlocking with another stage's CUDA wait.
        export NCCL_P2P_PXN_LEVEL=0
        ROLLOUT_TOPOLOGY_ARGS=(
            --rollout-num-gpus-per-engine 16
            --sglang-enable-dp-attention
            --sglang-dp-size 2
            --sglang-ep-size 16
        )
        ;;
    3)
        TOTAL_GPUS=24
        # Keep three independent TP8/EP8 engines. A single TP24/DP3 engine
        # cannot evenly shard 512 experts over EP24, or width 640 over MoE TP3 (EP8).
        PP_SIZE=3
        CP_SIZE=1
        EP_SIZE=8
        DEFAULT_SAVE_INTERVAL=100
        PP_LAYOUT_ARGS=(--pipeline-model-parallel-layout 'Et*16|t*16|t*16L')
        ;;
    4|8)
        TOTAL_GPUS=$((WORLD_SIZE * 8))
        PP_SIZE=4
        CP_SIZE=$((WORLD_SIZE == 4 ? 1 : 2))
        EP_SIZE=8
        DEFAULT_SAVE_INTERVAL=100
        ;;
    *)
        echo "Unsupported WORLD_SIZE=${WORLD_SIZE}: expected node count 2, 3, 4, or 8." >&2
        exit 1
        ;;
esac

MTP_ARGS=()
SGLANG_CONTEXT_LENGTH=10240
if [ "${ENABLE_MTP}" = 1 ]; then
    # EAGLE reserves max(topk * steps, draft_tokens) slots beyond prompt + response.
    SGLANG_CONTEXT_LENGTH=$((2048 + 8192 + 2))
    # Keep the first validation on the existing 24-GPU, CP1 topology.
    if [ "${WORLD_SIZE}" != 3 ]; then
        echo "This MTP recipe currently requires WORLD_SIZE=3 (24 GPUs, CP1)." >&2
        exit 1
    fi
    PP_LAYOUT_ARGS=(--pipeline-model-parallel-layout 'Et*16|t*16|t*16mL')
    MTP_ARGS=(
        --enable-mtp-training
        --mtp-num-layers 1
        --mtp-loss-scaling-factor 0.2
        --mtp-detach-paths embedding backbone lm-head
        --sglang-speculative-algorithm EAGLE
        --sglang-speculative-num-steps 1
        --sglang-speculative-eagle-topk 1
        --sglang-speculative-num-draft-tokens 2
        --sglang-mamba-radix-cache-strategy extra_buffer
    )
fi

CPU_OFFLOAD="${CPU_OFFLOAD:-$((TOTAL_GPUS == 16))}"
case "${CPU_OFFLOAD}" in
    0|1) ;;
    *)
        echo "Invalid CPU_OFFLOAD=${CPU_OFFLOAD}: expected 0 or 1." >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# Use this checkout and the Megatron source selected by the Python runtime.
export RELAX="${RELAX_ROOT}"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    : "${MEGATRON:?Set MEGATRON to the image checkout path for DRY_RUN}"
else
    MEGATRON=$(python3 -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("megatron.core").origin).resolve().parents[2])')
fi
export MEGATRON
export PYTHONPATH="${RELAX}:${MEGATRON}${PYTHONPATH:+:${PYTHONPATH}}"
: "${MODEL_DIR:?Set MODEL_DIR to the initial model root}"
: "${DATA_DIR:?Set DATA_DIR to the OpenR1MM dataset root}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent checkpoint root}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi
# Reuse the example's full 48-layer architecture and Bridge defaults.
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"

# Keep streamed PLE disabled. Host-only training PLE is not supported by the
# colocated GPU weight iterator; keep the frozen table on its owning GPU ranks.
export RELAX_QWEN38_PLE_STREAMED=0
export QWEN38_PLE_CPU_OFFLOAD=0
export SGLANG_VLM_CACHE_SIZE_MB=0
# Attention TP8 GDN projection has width 2060, incompatible with DeepGEMM TMA alignment.
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
# Scale FP32 QK scores like training; avoid rounding a pre-scaled Q to BF16.
export RELAX_QSA_MATCH_TRAINING_SCALE="${RELAX_QSA_MATCH_TRAINING_SCALE:-1}"
# Match Megatron's FP32 router. Other AutoModel parity knobs remain opt-in:
# their BF16 rounding contracts are not established for Megatron fused kernels.
export RELAX_QWEN38_FP32_ROUTER=1

MODEL_NAME="${MODEL_NAME:-Qwen3.8-Flash-Next}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/openr1mm}"
DEFAULT_EXP_NAME="qwen38-flash-next-openr1mm-grpo-${TOTAL_GPUS}xgpu-blackwell-280g-colocate-t4p${PP_SIZE}c${CP_SIZE}e${EP_SIZE}-ple-frozen-vision-frozen-tis-bs512-think-answer"
if [ "${ENABLE_MTP}" = 1 ]; then
    DEFAULT_EXP_NAME+="-mtp"
fi
EXP_NAME="${EXP_NAME:-${DEFAULT_EXP_NAME}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${DEFAULT_SAVE_INTERVAL}}"
now=$(date "+%Y%m%d-%H%M%S")

# First launch initializes from HF; subsequent launches restore training state.
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL}"
    --max-actor-ckpt-to-keep 2
)

# Match the Qwen3.5 OpenR1MM system prompt verbatim.
SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet"
    --input-key prompt
    --label-key label
    --multimodal-keys '{"image":"image"}'
    --apply-chat-template
    # Qwen3.8's medium mode adds no extra reasoning instruction; retain the <think> prefix.
    --apply-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}'
    --system-prompt "${SYSTEM_PROMPT}"
    --rollout-shuffle
    --rm-type openr1mm_accuracy_format
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size 64
    --n-samples-per-prompt 8
    --global-batch-size 512
    --rollout-max-prompt-len 2048
    --rollout-max-response-len 8192
    --rollout-max-context-len 10240
    --rollout-temperature 1.0
    --balance-data
)

# DYNAMIC_FILTER=1 enables DAPO filtering and resamples until 64 groups remain.
# Disabled by default because filtering increases rollout sampling time.
if [ "${DYNAMIC_FILTER:-0}" = 1 ]; then
    ROLLOUT_ARGS+=(
        --dynamic-sampling-filter-path relax.engine.filters.dynamic_sampling_filters.check_reward_nonzero_std
    )
fi

# Split 48 layers over PP3 (16 layers per stage) or PP4 (12 layers per stage).
PERF_ARGS=(
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size "${PP_SIZE}"
    "${PP_LAYOUT_ARGS[@]}"
    --context-parallel-size "${CP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --micro-batch-size 1
    --use-dynamic-batch-size
    # Dynamic microbatch token budget per GPU.
    --max-tokens-per-gpu 16384
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --freeze-vision-model
    --freeze-vision-projection
    # Freeze the entire PLE and QSA indexer.
    --freeze-params-name-list 'self_attention\.indexer\.' '(^|\.)per_layer_embedding\.'
    --no-rope-fusion
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --calculate-per-token-loss
    --eps-clip 0.2
    --eps-clip-high 0.28
    --entropy-coef 0.0
    --get-mismatch-metrics
    --use-tis
    --custom-tis-function-path relax.backends.megatron.loss.vanilla_tis_function
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-precision-aware-optimizer
)
if [ "${CPU_OFFLOAD}" = 1 ]; then
    OPTIMIZER_ARGS+=(
        --optimizer-cpu-offload
        --overlap-cpu-optimizer-d2h-h2d
    )
fi

SGLANG_ARGS=(
    --sglang-load-format "${SGLANG_LOAD_FORMAT:-dummy}"
    "${ROLLOUT_TOPOLOGY_ARGS[@]}"
    # Distribute requests across engines / DP workers.
    --sglang-router-policy round_robin
    --sglang-mem-fraction-static 0.8
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
    # Allow 64 full 10240-token contexts plus prefix-cache headroom.
    --sglang-max-total-tokens 1048576
    # Explicit backend enables the BF16 MoE unpack/repack weight-update lifecycle.
    --sglang-moe-runner-backend flashinfer_trtllm
    --sglang-max-running-requests 128
    --sglang-max-mamba-cache-size 512
    # Preserve visual processing: do not copy the text-only AutoModel override.
)

MISC_ARGS=(
    --bf16
    --attention-backend flash
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
)

# Credentials come from the existing environment; each run gets a new task name.
TRACKING_ARGS=(--use-metrics-service)
if [ "${USE_CLEARML:-1}" = 1 ]; then
    TRACKING_ARGS+=(
        --use-clearml
        --tb-project-name "${PROJECT_NAME}"
        --tb-experiment-name "${EXP_NAME}-${now}"
    )
fi

# Actor and rollout workers must receive the same model sources and precision controls.
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env = runtime.setdefault("env_vars", {})
for name, value in os.environ.items():
    if name.startswith(("CLEARML_", "RELAX_QWEN38_", "RELAX_GDN_", "RELAX_HC_", "RELAX_MOE_", "RELAX_QSA_")) or name in (
        "PYTHONPATH", "MEGATRON", "RELAX", "PYTHONNOUSERSITE",
        "QWEN38_PLE_CPU_OFFLOAD", "MCORE_QSA_SPARSE_BACKEND",
        "SGLANG_VLM_CACHE_SIZE_MB", "CUDA_HOME", "TRITON_F32_DEFAULT",
        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM",
        "NCCL_P2P_PXN_LEVEL",
    ):
        env[name] = value
propagated = [*env.get("RELAX_PROPAGATE_ENV_VARS", "").split(","), *env]
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys(name for name in propagated if name))
print(json.dumps(runtime))
PY
)
TRAIN_ARGS=(
    --resource "{\"actor\":[1,${TOTAL_GPUS}],\"rollout\":[1,${TOTAL_GPUS}]}"
    --colocate
    --max-staleness 0
    --actor-num-nodes "${WORLD_SIZE}"
    --actor-num-gpus-per-node 8
    --num-data-storage-units 4
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MTP_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${TRACKING_ARGS[@]}"
    "$@"
)
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${RUNTIME_ENV_JSON}" python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" <<'PYDRY'
import json
import sys
print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYDRY
    exit 0
fi
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:8265}" \
    --entrypoint-num-cpus 1 \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
