#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Shared compiler-cache lifecycle helpers for Relax entrypoints.
#
# local.sh / spmd-multinode.sh restore on the current node before it joins Ray,
# then the head attaches detached publishers after the cluster is ready.
# ray-job.sh runs only on the head and asks Ray to restore/attach every GPU node.

if [ -n "${_RELAX_KERNEL_CACHE_HELPER_SOURCED:-}" ]; then
    return 0 2>/dev/null || exit 0
fi
_RELAX_KERNEL_CACHE_HELPER_SOURCED=1

_RELAX_KERNEL_CACHE_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

relax_kernel_cache_configure() {
    local run_script="${1:-}"
    shift || true
    if [ -z "${RELAX_KERNEL_CACHE_DIR:-}" ]; then
        return 0
    fi
    if [ -n "${_RELAX_KERNEL_CACHE_CONFIGURED:-}" ]; then
        return 0
    fi
    if [[ "${RELAX_KERNEL_CACHE_DIR}" != /* ]]; then
        echo "ERROR: RELAX_KERNEL_CACHE_DIR must be an absolute shared-filesystem path." >&2
        return 1
    fi
    if [ -z "${run_script}" ] || [ ! -f "${run_script}" ]; then
        echo "ERROR: kernel cache requires the concrete training script path." >&2
        return 1
    fi

    RELAX_KERNEL_CACHE_SESSION_ID="${RELAX_KERNEL_CACHE_SESSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}}"
    if [ -z "${RELAX_KERNEL_CACHE_BUILD_FINGERPRINT:-}" ]; then
        if [ "${RELAX_ENTRYPOINT_MODE:-}" = "ray-job" ]; then
            RELAX_KERNEL_CACHE_BUILD_FINGERPRINT=$(python3 -c \
                'from relax.distributed.ray.kernel_cache import compute_gpu_cluster_build_fingerprint; print(compute_gpu_cluster_build_fingerprint())')
        else
            RELAX_KERNEL_CACHE_BUILD_FINGERPRINT=$(python3 -c \
                'from relax.distributed.ray.kernel_cache import compute_build_fingerprint; print(compute_build_fingerprint())')
        fi
    fi
    if [ -z "${RELAX_KERNEL_CACHE_KEY:-}" ]; then
        RELAX_KERNEL_CACHE_KEY=$(python3 - "${run_script}" "$@" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

profile_env = (
    "MODEL_PATH", "MODEL_DIR", "TOKENIZER_PATH", "TRAIN_BACKEND", "MODEL_TYPE",
    "TP_SIZE", "PP_SIZE", "CP_SIZE", "EP_SIZE", "ETP_SIZE", "VPP_SIZE",
    "DECODER_FIRST", "DECODER_LAST", "MAX_TOKENS_PER_GPU", "DATA_PAD_SIZE_MULTIPLIER",
    "MOE_TOKEN_DISPATCHER_TYPE", "MOE_FLEX_DISPATCHER_BACKEND", "MTP_NUM_LAYERS",
    "IMAGE_MAX_TOKEN_NUM", "SFT_LOGITS_CHUNK_SIZE", "LORA_RANK", "LORA_ALPHA",
    "LORA_DROPOUT", "LORA_SCOPE", "LORA_TARGET_MODULES", "ATTENTION_BACKEND",
    "RECOMPUTE_GRANULARITY", "RECOMPUTE_METHOD", "RECOMPUTE_NUM_LAYERS",
    "USE_FLASH_ATTN", "DISABLE_JIT_FUSER", "VISION_DP_WHEN_TP",
)
run_script = Path(sys.argv[1]).resolve()
payload = {
    "run_script_sha256": hashlib.sha256(run_script.read_bytes()).hexdigest(),
    "extra_args": sys.argv[2:],
    "overrides": {name: os.environ[name] for name in profile_env if name in os.environ},
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
PY
        )
    fi
    if [ -z "${RELAX_KERNEL_CACHE_LOCAL_DIR:-}" ]; then
        RELAX_KERNEL_CACHE_LOCAL_DIR=$(python3 -c \
            'import sys; from relax.distributed.ray.kernel_cache import derive_local_cache_dir; print(derive_local_cache_dir(*sys.argv[1:4]))' \
            "${RELAX_KERNEL_CACHE_DIR}" "${RELAX_KERNEL_CACHE_KEY}" "${RELAX_KERNEL_CACHE_BUILD_FINGERPRINT}")
    fi

    export RELAX_KERNEL_CACHE_DIR RELAX_KERNEL_CACHE_LOCAL_DIR RELAX_KERNEL_CACHE_SESSION_ID
    export RELAX_KERNEL_CACHE_KEY RELAX_KERNEL_CACHE_BUILD_FINGERPRINT
    _RELAX_KERNEL_CACHE_CONFIGURED=1
}

_relax_kernel_cache_prepare() {
    local mode="$1"
    if [ -z "${RELAX_KERNEL_CACHE_DIR:-}" ]; then
        return 0
    fi
    python3 "${_RELAX_KERNEL_CACHE_SH_DIR}/../tools/kernel_cache.py" \
        --mode "${mode}" \
        --shared-dir "${RELAX_KERNEL_CACHE_DIR}" \
        --local-dir "${RELAX_KERNEL_CACHE_LOCAL_DIR}" \
        --session-id "${RELAX_KERNEL_CACHE_SESSION_ID}" \
        --cache-key "${RELAX_KERNEL_CACHE_KEY}" \
        --build-fingerprint "${RELAX_KERNEL_CACHE_BUILD_FINGERPRINT}" \
        --sync-interval-sec "${RELAX_KERNEL_CACHE_SYNC_INTERVAL_SEC:-900}" \
        --lease-timeout-sec "${RELAX_KERNEL_CACHE_LEASE_TIMEOUT_SEC:-600}" \
        --startup-timeout-sec "${RELAX_KERNEL_CACHE_STARTUP_TIMEOUT_SEC:-7200}" \
        --compression "${RELAX_KERNEL_CACHE_COMPRESSION:-none}"
}

relax_kernel_cache_prepare_local() {
    if [ -n "${RELAX_KERNEL_CACHE_DIR:-}" ]; then
        echo "=== Restoring compiler cache on the current node ==="
        _relax_kernel_cache_prepare local
    fi
}

relax_kernel_cache_start_agents() {
    local mode="${1:-cluster}"
    if [ -n "${RELAX_KERNEL_CACHE_DIR:-}" ]; then
        echo "=== Starting compiler cache agents on all GPU nodes (mode=${mode}) ==="
        _relax_kernel_cache_prepare "${mode}"
    fi
}

relax_kernel_cache_inject_runtime_env() {
    if [ -z "${RELAX_KERNEL_CACHE_DIR:-}" ]; then
        return 0
    fi
    export RUNTIME_ENV_JSON
    RUNTIME_ENV_JSON=$(python3 -c '
import json, os
d = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = d.setdefault("env_vars", {})
for name in (
    "RELAX_KERNEL_CACHE_DIR",
    "RELAX_KERNEL_CACHE_LOCAL_DIR",
    "RELAX_KERNEL_CACHE_SESSION_ID",
    "RELAX_KERNEL_CACHE_KEY",
    "RELAX_KERNEL_CACHE_BUILD_FINGERPRINT",
    "RELAX_KERNEL_CACHE_HEARTBEAT_INTERVAL_SEC",
    "RELAX_KERNEL_CACHE_EXIT_TIMEOUT_SEC",
):
    if name in os.environ:
        env[name] = os.environ[name]
print(json.dumps(d))
')
    export RUNTIME_ENV_JSON
}
