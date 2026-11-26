#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Source after entrypoint setup and before Ray submission. SAVE_DIR must be
# shared across GPU nodes. An explicitly empty RELAX_KERNEL_CACHE_DIR disables
# persistence; runs without SAVE_DIR can supply a shared cache directory.

if [ "${DRY_RUN:-0}" = 1 ]; then
    return 0
fi
if [ -z "${RELAX_KERNEL_CACHE_DIR+x}" ] && [ -n "${SAVE_DIR:-}" ]; then
    export RELAX_KERNEL_CACHE_DIR="${SAVE_DIR}/.kernel-cache"
fi
# An outer entrypoint may have already prepared this session. Reuse it rather
# than starting a second set of cache publishers for the same training job.
_KIMI_KERNEL_CACHE_READY=$(python3 - <<'PY'
import json
import os

names = (
    "RELAX_KERNEL_CACHE_DIR", "RELAX_KERNEL_CACHE_LOCAL_DIR",
    "RELAX_KERNEL_CACHE_SESSION_ID", "RELAX_KERNEL_CACHE_KEY",
    "RELAX_KERNEL_CACHE_BUILD_FINGERPRINT",
)
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"]).get("env_vars", {})
prepared = any(name in runtime for name in names)
if prepared and any(not runtime.get(name) or runtime[name] != os.environ.get(name) for name in names):
    raise SystemExit("Kimi kernel cache: runtime and shell cache identities are incomplete or inconsistent.")
print("1" if prepared else "0")
PY
)
if [ -z "${RELAX_KERNEL_CACHE_DIR:-}" ]; then
    unset _KIMI_KERNEL_CACHE_READY
    return 0
fi

source "${RELAX_ROOT}/scripts/entrypoint/kernel-cache.sh"
if [ "${_KIMI_KERNEL_CACHE_READY}" != 1 ]; then
    relax_kernel_cache_configure "${BASH_SOURCE[1]}" "$@"
    relax_kernel_cache_start_agents cluster
fi
relax_kernel_cache_inject_runtime_env
unset _KIMI_KERNEL_CACHE_READY
