#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Mini SWE Agent with CPU HiCache in regular or PD deployment on eight GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${RUNTIME_ENV_JSON:?Launch through scripts/entrypoint/ray-job.sh}"
HICACHE_DEPLOYMENT="${HICACHE_DEPLOYMENT:-regular}"
case "${HICACHE_DEPLOYMENT}" in
   regular) config="hicache-cpu-4x-tp2.yaml" ;;
   pd) config="pd-hicache-cpu-2p2d-tp2.yaml" ;;
   *) echo "HICACHE_DEPLOYMENT must be regular or pd." >&2; exit 1 ;;
esac

export RUN_ID="${RUN_ID:-mini-swe-cpu-cache-${HICACHE_DEPLOYMENT}-$(date +%Y%m%d-%H%M%S)}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/mini_swe_agent_hicache}"
# Reuse the Ray driver and training settings; the YAML selects the topology.
exec bash "${SCRIPT_DIR}/run_mini_swe_agent_pd.sh" \
   --sglang-config "${SCRIPT_DIR}/${config}" "$@"
