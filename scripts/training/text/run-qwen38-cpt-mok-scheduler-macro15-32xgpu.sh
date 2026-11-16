#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export MOE_BACKEND=mok
export MOK_BACKEND_CONFIG='{"shared_expert_mode":"native","fixed_num_local_tokens":131072,"fwd_num_comm_sms":56,"bwd_num_comm_sms":56,"minibatch_size":8192,"macrobatch_size":1572864,"schedule_capacity_multiplier":0.5,"all_gather_top_experts_chunk_bytes":2048}'

exec bash "${REPO_ROOT}/examples/models/qwen3.8-flash-next/scripts/run-qwen38-qsa-cpt-256k-32xgpu-blackwell.sh" "$@"
