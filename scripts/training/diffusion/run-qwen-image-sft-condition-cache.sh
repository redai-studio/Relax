#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Add bounded frozen-conditioning reuse and measurements to the online Qwen recipe.
# Invoke through scripts/entrypoint/ray-job.sh; DRY_RUN=1 only prints the command.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    # shellcheck source=../../entrypoint/ray-job.sh
    source "${RELAX_ROOT}/scripts/entrypoint/ray-job.sh"
fi

: "${SAVE_DIR:?Set SAVE_DIR to the experiment checkpoint root}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/sft/condition-cache}"
export EXP_NAME="${EXP_NAME:-qwen-image-sft-condition-cache}"
export RELAX_CONDITION_CACHE="${RELAX_CONDITION_CACHE:-0}"
export RELAX_CONDITION_PROFILE="${RELAX_CONDITION_PROFILE:-1}"
export RELAX_CONDITION_VERIFY="${RELAX_CONDITION_VERIFY:-0}"
export RELAX_CONDITION_VERIFY_RTOL="${RELAX_CONDITION_VERIFY_RTOL:-0.01}"
export RELAX_CONDITION_CACHE_MAX_BYTES="${RELAX_CONDITION_CACHE_MAX_BYTES:-1073741824}"
export RELAX_CONDITION_CACHE_DIR="${RELAX_CONDITION_CACHE_DIR:-}"
export RELAX_CONDITION_CACHE_RUN_DIR="${RELAX_CONDITION_CACHE_RUN_DIR:-${SAVE_DIR}/${EXP_NAME}/metrics}"
export RELAX_CONDITION_PREFETCH_DEPTH="${RELAX_CONDITION_PREFETCH_DEPTH:-0}"
export RELAX_CONDITION_PREFETCH_MAX_BYTES="${RELAX_CONDITION_PREFETCH_MAX_BYTES:-536870912}"
export RUN_DIR="${RUN_DIR:-${RELAX_CONDITION_CACHE_RUN_DIR}}"
RUNTIME_ENV_JSON=$(python3 - <<'PYENV'
import json
import math
import os

for name in ("RELAX_CONDITION_CACHE", "RELAX_CONDITION_PROFILE", "RELAX_CONDITION_VERIFY"):
    if os.environ[name] not in ("0", "1"):
        raise SystemExit(f"{name} must be 0 or 1")
for name in ("RELAX_CONDITION_CACHE_MAX_BYTES", "RELAX_CONDITION_PREFETCH_DEPTH", "RELAX_CONDITION_PREFETCH_MAX_BYTES"):
    if not os.environ[name].isdecimal():
        raise SystemExit(f"{name} must be a nonnegative integer")
if os.environ["RELAX_CONDITION_CACHE"] == "1" and int(os.environ["RELAX_CONDITION_CACHE_MAX_BYTES"]) == 0:
    raise SystemExit("An enabled cache requires a positive byte budget")
if int(os.environ["RELAX_CONDITION_PREFETCH_DEPTH"]) != 0:
    raise SystemExit("Qwen-Image SFT currently requires condition prefetch depth 0")
try:
    tolerance = float(os.environ["RELAX_CONDITION_VERIFY_RTOL"])
except ValueError:
    raise SystemExit("RELAX_CONDITION_VERIFY_RTOL must be numeric") from None
if not math.isfinite(tolerance) or not 0 <= tolerance <= 0.01:
    raise SystemExit("RELAX_CONDITION_VERIFY_RTOL must be finite and between 0 and 0.01")
runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON", "{}"))
env = runtime.setdefault("env_vars", {})
for key in ("RELAX_CONDITION_CACHE", "RELAX_CONDITION_PROFILE", "RELAX_CONDITION_CACHE_MAX_BYTES",
            "RELAX_CONDITION_CACHE_DIR", "RELAX_CONDITION_CACHE_RUN_DIR", "RELAX_CONDITION_VERIFY",
            "RELAX_CONDITION_VERIFY_RTOL", "RELAX_CONDITION_PREFETCH_DEPTH", "RELAX_CONDITION_PREFETCH_MAX_BYTES"):
    env[key] = os.environ[key]
print(json.dumps(runtime))
PYENV
)
export RUNTIME_ENV_JSON
if [[ "${DRY_RUN:-0}" != 1 ]] && compgen -G "${RELAX_CONDITION_CACHE_RUN_DIR}/events-rank*.jsonl" >/dev/null; then
    echo "Run directory already contains metrics; select a fresh RELAX_CONDITION_CACHE_RUN_DIR" >&2
    exit 1
fi
exec bash "${SCRIPT_DIR}/run-qwen-image-sft.sh" "$@"
