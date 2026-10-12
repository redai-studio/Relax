#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Mini SWE Agent with native CPU / SSD HiCache on 8 GPUs (2P2D, TP2).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${RUNTIME_ENV_JSON:?Launch through scripts/entrypoint/ray-job.sh}"
HICACHE_TIER="${HICACHE_TIER:-ssd}"
case "${HICACHE_TIER}" in
   cpu|ssd) ;;
   *) echo "HICACHE_TIER must be cpu or ssd." >&2; exit 1 ;;
esac

if [[ "${HICACHE_TIER}" == ssd ]]; then
   : "${RELAX_HICACHE_FILE_ROOT:?Set a new absolute directory on local SSD}"
   [[ "${RELAX_HICACHE_FILE_ROOT}" == /* ]] || {
      echo "RELAX_HICACHE_FILE_ROOT must be absolute." >&2; exit 1
   }
   mkdir -p "${RELAX_HICACHE_FILE_ROOT}"
   [[ -w "${RELAX_HICACHE_FILE_ROOT}" && -z "$(ls -A -- "${RELAX_HICACHE_FILE_ROOT}")" ]] || {
      echo "Choose a writable, empty cache directory for this run." >&2; exit 1
   }
   export RELAX_HICACHE_FILE_ROOT
fi

# Ray workers must receive the run root; a submitter-only export is insufficient.
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - <<'ENV'
import json
import os
runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = runtime.setdefault("env_vars", {})
if os.environ.get("RELAX_HICACHE_FILE_ROOT"):
    env["RELAX_HICACHE_FILE_ROOT"] = os.environ["RELAX_HICACHE_FILE_ROOT"]
print(json.dumps(runtime))
ENV
)
export RUN_ID="${RUN_ID:-mini-swe-hicache-${HICACHE_TIER}-$(date +%Y%m%d-%H%M%S)}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/mini_swe_agent_hicache}"
exec bash "${SCRIPT_DIR}/run_mini_swe_agent_pd.sh" \
   --sglang-config "${SCRIPT_DIR}/pd-hicache-${HICACHE_TIER}-2p2d-tp2.yaml" "$@"
