#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full 48-layer Flash-Next: Megatron TP8/PP8/CP1/EP16; colocated SGLang TP8.
# Requires 16 nodes x 8 H800 GPUs and the unified Hopper image on every node.
# Export MODEL_DIR, DATA_DIR, SAVE_DIR, RAY_ADDRESS (GCS), then:
# bash scripts/entrypoint/ray-job.sh examples/models/qwen3.8-flash-next/scripts/run-qwen38-flash-next-openr1mm-128xgpu.sh
# MODEL_DIR/Qwen3.8-Flash-Next-F0 is the initial checkpoint; MODEL_NAME may override it.
# DATA_DIR/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet
# must contain prompt, label and image columns (the existing OpenR1MM conversion).
# Resume by keeping SAVE_DIR/EXP_NAME; checkpoints every 50 steps, retain 2.
# DRY_RUN=1 MEGATRON=<image-checkout> prints argv/runtime JSON without Ray or GPU setup.
# This is a full-model recipe; local reduced-model checks do not validate 128-GPU RL.

set -eo pipefail
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
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
source "${RELAX_ROOT}/scripts/models/qwen38-flash-next.sh"
# CP1 cannot use streamed PLE. Host-only training PLE is not supported by the
# colocated GPU weight iterator; keep the frozen table on its owning GPU ranks.
export RELAX_QWEN38_PLE_STREAMED=0
export QWEN48_PLE_CPU_OFFLOAD=0
export SGLANG_VLM_CACHE_SIZE_MB=0
# Match Megatron's FP32 router. Other AutoModel parity knobs remain opt-in:
# their BF16 rounding contracts are not established for Megatron fused kernels.
export RELAX_QWEN38_FP32_ROUTER=1

MODEL_NAME="${MODEL_NAME:-Qwen3.8-Flash-Next-F0}"
PROJECT_NAME="${PROJECT_NAME:-Relax/rl/openr1mm}"
EXP_NAME="${EXP_NAME:-qwen38-flash-next-openr1mm-128xgpu-t8p8c1e16}"
now=$(date "+%Y%m%d-%H%M%S")
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/${MODEL_NAME}"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${EXP_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 2
)
ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/multimodal-open-r1-8k-verified/data/train-00000-of-00001_converted_noextract.parquet"
    --input-key prompt
    --label-key label
    --multimodal-keys '{"image":"image"}'
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking":false}'
    --system-prompt 'Solve the problem and put your final answer inside <answer> </answer> tags.'
    --rollout-shuffle
    --rm-type openr1mm
    --num-rollout "${NUM_ROLLOUT:-200}"
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --global-batch-size 256
    --rollout-max-prompt-len 2048
    --rollout-max-response-len 2048
    --rollout-max-context-len 4096
    --rollout-temperature 1.0
    --image-max-token-num 1024
    --balance-data
)
PERF_ARGS=(
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 8
    --context-parallel-size 1
    --expert-model-parallel-size 16
    --expert-tensor-parallel-size 1
    --sequence-parallel
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --micro-batch-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 4096
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --freeze-vision-model
    --freeze-vision-projection
    --freeze-params-name-list 'self_attention\.indexer\.' '(^|\.)per_layer_embedding\.'
    --no-rope-fusion
    --attention-backend flash
    --bf16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
)
GRPO_ARGS=(
    --advantage-estimator grpo
    --calculate-per-token-loss
    --eps-clip 0.2
    --eps-clip-high 0.28
    --entropy-coef 0.0
    --get-mismatch-metrics
    # Keep observations separate from TIS. Enable correction only after evaluating mismatch.
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
)
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 8
    --sglang-mem-fraction-static 0.8
    --sglang-disable-cuda-graph
    --sglang-enable-deterministic-inference
    --sglang-bf16-gemm-backend torch
    --sglang-linear-attn-prefill-backend triton
    --sglang-max-running-requests 16
    --sglang-max-mamba-cache-size 64
    # Preserve visual processing: do not copy the text-only AutoModel override.
)
TRACKING_ARGS=(--use-metrics-service)
if [ "${USE_CLEARML:-1}" = 1 ]; then
    TRACKING_ARGS+=(--use-clearml --tb-project-name "${PROJECT_NAME}" --tb-experiment-name "${EXP_NAME}-${now}")
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
        "PYTHONPATH", "MEGATRON", "RELAX", "PYTHONNOUSERSITE", "QWEN48_PLE_CPU_OFFLOAD",
        "SGLANG_VLM_CACHE_SIZE_MB", "CUDA_HOME", "TRITON_F32_DEFAULT",
    ):
        env[name] = value
propagated = [*env.get("RELAX_PROPAGATE_ENV_VARS", "").split(","), *env]
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys(name for name in propagated if name))
print(json.dumps(runtime))
PY
)
TRAIN_ARGS=(
    --resource '{"actor":[1,128],"rollout":[1,128]}'
    --colocate
    --max-staleness 0
    --actor-num-nodes 16
    --actor-num-gpus-per-node 8
    --num-data-storage-units 1
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
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
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" \
    2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
