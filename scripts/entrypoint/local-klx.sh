#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Default environment configuration for local single-node development on
# Kunlunxin P800. Mirrors scripts/entrypoint/local.sh, but drops NVIDIA-only
# bits that don't apply to P800:
#   - NVLink detection (relies on `nvidia-smi topo`)
#   - RUNTIME_ENV_JSON (NCCL/NVSHMEM-specific tuning env vars)
#
# This script handles process cleanup, environment setup, and Ray cluster startup.
# It is designed to be *sourced* by run-*.sh scripts when no external entrypoint
# (spmd-multinode.sh or ray-job.sh) has been used.
#
# When an existing Ray cluster is detected (RAY_ADDRESS set and `ray status` OK),
# this script delegates to `ray-job.sh` (source mode) instead of starting a new
# local Ray head node.
#
# Usage (from a run script):
#   source scripts/entrypoint/local-p800.sh
#
# Environment variables:
#   NUM_GPUS               - Number of P800 cards to use (optional, auto-detect from CUDA_VISIBLE_DEVICES)
#   CUDA_VISIBLE_DEVICES   - Comma-separated device IDs (e.g., "0,1,2,3" → 4 cards)
#   MASTER_ADDR            - Head node IP address (default: 127.0.0.1)
#   MEGATRON               - Path to Megatron-LM (default: /root/Megatron-LM/)
#   RELAX                  - Path to Relax project (default: ../../)

# Guard: skip if already sourced by another entrypoint
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

_LOCAL_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ── delegate to ray-job.sh when inside an existing Ray cluster ─────────────
# When RAY_ADDRESS is set AND `ray status` succeeds, we're already part of an
# externally-managed Ray cluster. Skip local Ray startup / process cleanup and
# fall through to ray-job.sh (source mode) for env setup.
if [ -n "${RAY_ADDRESS:-}" ] && timeout 5 ray status >/dev/null 2>&1; then
    echo "=== Detected existing Ray cluster (RAY_ADDRESS=$RAY_ADDRESS); delegating to ray-job.sh ==="
    # shellcheck source=./ray-job.sh
    source "${_LOCAL_SH_DIR}/ray-job.sh"
    return 0 2>/dev/null || exit 0
fi

set -eo pipefail

# ── process cleanup ─────────────────────────────────────────────────────────
echo "=== Cleaning up stale processes ==="
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 3
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true

set -x

# ── environment setup ───────────────────────────────────────────────────────
unset MASTER_ADDR 2>/dev/null || true
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export RELAX=${RELAX:-${_LOCAL_SH_DIR}/../../}
export PYTHONPATH=${RELAX}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${_LOCAL_SH_DIR}/../models"

# ── NVLink detection ────────────────────────────────────────────────────────
# Not applicable on P800 (no `nvidia-smi topo`).

# ── GPU count detection ───────────────────────────────────────────────────────
# Priority: NUM_GPUS env > CUDA_VISIBLE_DEVICES > default 8
if [ -z "${NUM_GPUS:-}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        # Count cards from CUDA_VISIBLE_DEVICES (comma-separated)
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')
    else
        NUM_GPUS=8
    fi
fi

# ── Ray cluster startup (single node) ──────────────────────────────────────
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
echo "Starting Ray head node: MASTER_ADDR=$MASTER_ADDR, NUM_GPUS=$NUM_GPUS"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

# ── set entrypoint mode ────────────────────────────────────────────────────
export RELAX_ENTRYPOINT_MODE="local"

# Runtime env (RUNTIME_ENV_JSON) is intentionally omitted on P800: the NVIDIA
# NCCL/NVSHMEM-specific tuning vars used on CUDA do not apply here.
