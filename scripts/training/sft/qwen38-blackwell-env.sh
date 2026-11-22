#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Shared Ray runtime environment for the Qwen3.8 32-Blackwell SM103 SFT examples.
# Arguments are additional per-modality worker settings.
qwen38_blackwell_env() {
    local -a ENV_ARGS=(
        CLEARML_OFFLINE_MODE=0
        HF_HUB_OFFLINE=1
        "RELAX_SKIP_NONFINITE_GRAD=${RELAX_SKIP_NONFINITE_GRAD:-0}"
        MCORE_QSA_SPARSE_BACKEND=id_sparse
        TORCHINDUCTOR_COMPILE_THREADS=1
        QWEN38_PLE_CPU_OFFLOAD=1
        "RELAX_HYBRID_EP_COMMIT=${RELAX_HYBRID_EP_COMMIT}"
        "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        NCCL_NVLS_ENABLE=0
        "PYTHONPATH=${RELAX}:${MEGATRON}"
        "MEGATRON=${MEGATRON}"
        "RELAX=${RELAX}"
        PYTHONNOUSERSITE=1
        "CUDA_DEVICE_MAX_CONNECTIONS=${QWEN38_CUDA_DEVICE_MAX_CONNECTIONS}"
        "NVTE_USE_CUTLASS_GROUPED_GEMM=${NVTE_USE_CUTLASS_GROUPED_GEMM:-0}"
        TOKENIZERS_PARALLELISM=false
        RELAX_QWEN38_QSA_FUSED_ATTENTION=0
        RELAX_QWEN38_RETAIN_QSA_ATTENTION=0
        RELAX_QWEN38_QSA_FIXED_POOL=0
        RELAX_QWEN38_QSA_SPARSE_BLOCKMASK=1
        "RELAX_QWEN38_RECORD_FUNCTION_TAGS=${RELAX_QWEN38_RECORD_FUNCTION_TAGS:-1}"
        RELAX_QWEN38_QSA_LOCAL_QUERY=1
        RELAX_QWEN38_ROUTER_SORTED_REPLAY=1
        RELAX_QWEN38_QSA_ROUTED_QKV=1
        RELAX_QWEN38_QSA_RECOMPUTE_REPLAY=1
        RELAX_QWEN38_QSA_CPU_LAYOUT=0
        RELAX_QWEN38_PLE_STREAMED=0
        RELAX_QWEN38_GDN_OUTPUT_SAFE_CONFIG=1
        RELAX_QWEN38_FLA_CONV_CACHE_REUSE=1
        RELAX_QWEN38_QSA_COMMUNITY_BACKEND=mixed
        "RELAX_QWEN38_QSA_MIXED_LONG_BACKEND=${RELAX_QWEN38_QSA_MIXED_LONG_BACKEND:-grouped}"
        "RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN=${RELAX_QWEN38_QSA_MIXED_FLASH_MAX_LEN:--1}"
        "RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS=${RELAX_QWEN38_QSA_NV_MAX_PACK_TOKENS:-0}"
        "QWEN38_WORKSPACE=${RELAX}"
        "SAVE_DIR=${SAVE_DIR}"
        RELAX_SFT_RESIDENT_PARAM_OVERLAP=0
        RAY_OVERRIDE_JOB_RUNTIME_ENV=1
        "TENSORBOARD_DIR=${SAVE_DIR}/tensorboard"
        "CLEARML_CACHE_DIR=${SAVE_DIR}/clearml"
        "RELAX_PROFILE_DIR=${SAVE_DIR}/traces"
        PYTHONDONTWRITEBYTECODE=1
        RELAX_PROFILE_GZIP_LEVEL=1
        FLA_CACHE_RESULTS=1
        TRITON_CACHE_AUTOTUNING=1
        TORCHINDUCTOR_FX_GRAPH_CACHE=1
        TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE=1
        "RELAX_AUTOTUNE_LOG_DIR=${RELAX_AUTOTUNE_LOG_DIR-${SAVE_DIR}/autotune}"
        RELAX_QWEN38_HC_FUSED_MIX=1
        RELAX_SFT_SELECTIVE_LOGITS=1
        RELAX_QWEN38_QSA_TP_SELECTOR=1
        RELAX_QWEN38_QSA_FUSED_SELECTOR=0
        RELAX_QWEN38_QSA_GROUPED_BACKWARD_WARPS=4
        RELAX_QWEN38_QSA_FLEX_SPARSE_MASK=1
        NVLINK_DOMAIN_SIZE=8
        USE_MNNVL=0
        RELAX_QWEN38_HC_FUSED_COMBINE=1
        RELAX_QWEN38_PLE_FUSED_CONV=1
        RELAX_SFT_FUSED_LINEAR_CE=1
        RELAX_QWEN38_QSA_GROUPED_BACKWARD_STAGES=3
        RELAX_SFT_MTP_ACCEPTANCE_METRICS=0
        RELAX_QWEN38_QSA_GROUPED_BACKWARD_BLOCK_K=32
        RELAX_QWEN38_QSA_SELECTOR_COMPILE=1
        "$@"
        "LD_PRELOAD=${LD_PRELOAD:-}"
        RELAX_PROPAGATE_ENV_VARS=QWEN38_WORKSPACE,SAVE_DIR,RELAX_PROFILE_DIR,TENSORBOARD_DIR,CLEARML_CACHE_DIR,CLEARML_OFFLINE_MODE,PYTHONDONTWRITEBYTECODE,RELAX_HYBRID_EP_COMMIT,LD_LIBRARY_PATH,LD_PRELOAD,NVLINK_DOMAIN_SIZE,USE_MNNVL,CUDA_DEVICE_MAX_CONNECTIONS,RELAX_SKIP_NONFINITE_GRAD
        "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM=${NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM:-0}"
        "PYTORCH_CUDA_ALLOC_CONF=${QWEN38_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
        "RELAX_PROFILE_RANKS=${RELAX_PROFILE_RANKS:-0,8,16,24}"
        "RELAX_QWEN38_HC_RECOMPUTE=${RELAX_QWEN38_HC_RECOMPUTE:-0}"
        "RELAX_QWEN38_HC_OUTER_BYPASS_MAX_LOCAL_ROWS=${RELAX_QWEN38_HC_OUTER_BYPASS_MAX_LOCAL_ROWS:-0}"
        "RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN=${RELAX_QWEN38_QSA_MIXED_FLEX_MAX_LEN:-0}"
    )
    local optional_env recipe_env
    for optional_env in NCCL_MAX_CTAS NCCL_MIN_CTAS NCCL_DEBUG; do
        recipe_env="QWEN38_${optional_env}"
        if [[ -n "${!recipe_env:-}" ]]; then
            ENV_ARGS+=("${optional_env}=${!recipe_env}")
        fi
    done
    unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN
    export "${ENV_ARGS[@]}"
    TRAIN_ENV_JSON=$(python3 - "${RUNTIME_ENV_JSON:-}" "${ENV_ARGS[@]}" <<'PYENV'
import json
import sys
env = dict(item.split("=", 1) for item in sys.argv[2:])
# ray-job.sh adds the torch and NVSHMEM library directories to the submission environment.
driver = json.loads(sys.argv[1] or "{}").get("env_vars", {})
if driver.get("LD_LIBRARY_PATH"):
    env["LD_LIBRARY_PATH"] = driver["LD_LIBRARY_PATH"]
print(json.dumps(env))
PYENV
    )
    BASE_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-}"
    if [[ -z "${BASE_RUNTIME_ENV_JSON}" ]]; then BASE_RUNTIME_ENV_JSON='{}'; fi
    RUNTIME_ENV_JSON=$(python3 - "${BASE_RUNTIME_ENV_JSON}" "${TRAIN_ENV_JSON}" <<'PYRUNTIME'
import json
import sys
runtime = json.loads(sys.argv[1])
runtime.setdefault("env_vars", {}).pop("NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN", None)
runtime["env_vars"].update(json.loads(sys.argv[2]))
print(json.dumps(runtime))
PYRUNTIME
    )
    # DeepEPv2 requires cuMem on the training actors. Use the train-worker CLI setting.
    TRAIN_WORKER_ENV_JSON=$(python3 - "${TRAIN_ENV_JSON}" <<'PYWORKERS'
import json
import sys

env = json.loads(sys.argv[1])
env["NCCL_CUMEM_ENABLE"] = "1"
print(json.dumps(env))
PYWORKERS
    )
}

qwen38_write_config() {
    # Retain the effective argv and worker environment for old/new comparison.
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "${SAVE_DIR}/logs"
    python3 - "${SAVE_DIR}/logs/effective-config.json" "${RUNTIME_ENV_JSON}" "${TRAIN_ARGS[@]}" <<'PYCONFIG'
import json
import sys
from pathlib import Path

runtime = json.loads(sys.argv[2])
# Only record explicitly selected training settings, not inherited credentials.
selected = {key: value for key, value in runtime.get("env_vars", {}).items()
            if key.startswith(("RELAX_QWEN38_", "RELAX_SFT_")) or key in
            ("MCORE_QSA_SPARSE_BACKEND", "CUDA_DEVICE_MAX_CONNECTIONS", "RELAX_HYBRID_EP_COMMIT",
             "NVTE_USE_CUTLASS_GROUPED_GEMM")}
Path(sys.argv[1]).write_text(json.dumps({"argv": sys.argv[3:], "env_vars": selected}, indent=2) + "\n")
PYCONFIG
    fi

    if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "${RUNTIME_ENV_JSON}" python3 -m relax.entrypoints.train "${TRAIN_ARGS[@]}" <<'PYDRY'
import json
import sys
print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PYDRY
    exit 0
    fi
}
