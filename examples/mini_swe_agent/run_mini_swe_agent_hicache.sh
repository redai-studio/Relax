#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Mini SWE Agent with CPU or CPU + Mooncake HiCache on eight GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${RUNTIME_ENV_JSON:?Launch through scripts/entrypoint/ray-job.sh}"
HICACHE_DEPLOYMENT="${HICACHE_DEPLOYMENT:-regular}"
HICACHE_TIER="${HICACHE_TIER:-cpu}"
extra_args=()
case "${HICACHE_DEPLOYMENT}" in
   regular) config="hicache-cpu-4x-tp2.yaml" ;;
   pd) config="pd-hicache-cpu-2p2d-tp2.yaml" ;;
   pd-dp2)
      config="pd-hicache-cpu-1p1d-dp2-tp2-ep4.yaml"
      extra_args=(--rollout-num-gpus-per-engine 4 --sglang-dp-size 2
         --sglang-enable-dp-attention --sglang-ep-size 4
         --sglang-router-decode-policy consistent_hashing)
      ;;
   *) echo "HICACHE_DEPLOYMENT must be regular, pd or pd-dp2." >&2; exit 1 ;;
esac

export RUN_ID="${RUN_ID:-mini-swe-${HICACHE_TIER}-cache-${HICACHE_DEPLOYMENT}-$(date +%Y%m%d-%H%M%S)}"
export PROJECT_NAME="${PROJECT_NAME:-Relax/mini_swe_agent_hicache}"
config="${SCRIPT_DIR}/${config}"
case "${HICACHE_TIER}" in
   cpu) ;;
   mooncake)
      : "${HICACHE_MOONCAKE_CONFIG:?Set HICACHE_MOONCAKE_CONFIG to a Mooncake Store JSON file}"
      : "${EXP_ROOT:?Set EXP_ROOT to a new shared output directory}"
      config=$(python3 - "${config}" "${HICACHE_MOONCAKE_CONFIG}" "${EXP_ROOT}" <<'CONFIG'
import json
import sys
from pathlib import Path

import yaml

base, store_path, output_dir = map(Path, sys.argv[1:])
store = json.loads(store_path.read_text())
if not isinstance(store, dict) or not store.get("master_server_address"):
    raise ValueError("Mooncake config must contain master_server_address")
model = yaml.safe_load(base.read_text())
for group in model["sglang"][0]["engine_groups"]:
    if group["worker_type"] == "decode":
        continue
    overrides = group["overrides"]
    overrides["hicache_storage_backend"] = "mooncake"
    overrides["hicache_storage_prefetch_policy"] = "wait_complete"
    # Embed connection settings so Ray workers do not depend on local exports
    # or access to the submitter's original config file.
    overrides["hicache_storage_backend_extra_config"] = json.dumps(store)
output = output_dir.resolve() / "hicache-mooncake.yaml"
output.parent.mkdir(parents=True, exist_ok=True)
text = yaml.safe_dump(model, sort_keys=False)
if output.exists() and output.read_text() != text:
    raise ValueError(f"Refusing to replace a different existing config: {output}")
output.write_text(text)
print(output)
CONFIG
      )
      ;;
   *) echo "HICACHE_TIER must be cpu or mooncake." >&2; exit 1 ;;
esac

# Reuse the Ray driver and training settings; the YAML selects the topology.
exec bash "${SCRIPT_DIR}/run_mini_swe_agent_pd.sh" \
   --sglang-config "${config}" "${extra_args[@]}" "$@"
