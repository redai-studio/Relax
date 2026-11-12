#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Entrypoint / source helper for Ray Job tasks.
# The Ray cluster is already running. This script MUST NOT kill ray or stop the
# cluster. It only cleans up residual python/sglang processes and then sets up
# the environment for running training against an existing Ray cluster.
#
# Two usage modes:
#   1) Entry-point mode — first argument is a .sh script path:
#        bash scripts/entrypoint/ray-job-npu.sh <run-script> [extra-args...]
#      Sets up env, cleans residual processes, then execs the run script.
#
#      Example:
#        bash scripts/entrypoint/ray-job-npu.sh scripts/training/text/run-qwen35-9B-8xgpu-async.sh
#        bash scripts/entrypoint/ray-job-npu.sh scripts/training/text/run-qwen35-9B-8xgpu-async.sh --lr 5e-7
#
#   2) Source mode — no .sh script arg (like local.sh):
#        source scripts/entrypoint/ray-job-npu.sh
#      Sets up env only, so the caller can continue execution.
#
# Environment variables (optional):
#   MEGATRON      - Path to Megatron-LM (default: /root/Megatron-LM/)
#   RELAX         - Path to Relax project (default: ../../)

# Guard: skip if already sourced by another entrypoint
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

# ── mode detection ──────────────────────────────────────────────────────────
# Entry-point mode: directly executed AND first arg is an existing .sh file.
# Otherwise act as a sourced setup script.
_RAY_JOB_RUN_SCRIPT=""
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _RAY_JOB_FIRST_ARG="${1:-}"
    if [ -n "$_RAY_JOB_FIRST_ARG" ] && [ -f "$_RAY_JOB_FIRST_ARG" ] && [[ "$_RAY_JOB_FIRST_ARG" == *.sh ]]; then
        _RAY_JOB_RUN_SCRIPT="$_RAY_JOB_FIRST_ARG"
        shift
    else
        echo "Usage: $0 <run-script.sh> [extra-args...]" >&2
        exit 1
    fi
fi

set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# ── clean up residual python/sglang processes (NOT ray) ─────────────────────
# IMPORTANT: Do NOT pkill ray or run ray stop — the cluster is managed externally.
echo "=== Cleaning up residual python/sglang processes ==="
python ${DIR}/../tools/run_on_each_ray_node.py ${DIR}/../tools/kill_for_ray.sh || echo "failed"

# kill old tasks, but never kill ourselves.
# Resolve our own submission_id with two strategies:
#   1) RAY_JOB_SUBMISSION_ID env var — set by Ray ≥ 2.6 via `ray job submit`,
#      but empty on some platforms (e.g. QS wrappers that strip env vars).
#   2) Fallback: Ray's job_supervisor redirects driver stdout/stderr to
#      /tmp/ray/session_latest/logs/job-driver-<sub_id>.log, so readlink fd 1/2
#      recovers <sub_id>.
# If both fail, SKIP cleanup — never kill blindly, because that suicides the job.
SELF_SUB_ID="${RAY_JOB_SUBMISSION_ID:-}"
if [ -z "$SELF_SUB_ID" ]; then
    for _fd in 1 2; do
        _path=$(readlink -f "/proc/self/fd/${_fd}" 2>/dev/null || true)
        if [[ "$_path" =~ /job-driver-(.+)\.(log|out|err)$ ]]; then
            SELF_SUB_ID="${BASH_REMATCH[1]}"
            break
        fi
    done
fi
echo "=== Own ray submission_id: ${SELF_SUB_ID:-<unknown>} ==="
if [ -z "$SELF_SUB_ID" ]; then
    echo "WARNING: could not detect own submission_id; skipping old-job cleanup to avoid suicide."
else
    ray job list \
      | grep RUNNING \
      | grep -oP "submission_id='\\K[^']+" \
      | grep -vFx "$SELF_SUB_ID" \
      | xargs --no-run-if-empty -n1 ray job stop || true
fi

set -x

# ── environment setup ───────────────────────────────────────────────────────
# Use the first GPU node as MASTER_ADDR (prefer head node)
export MASTER_ADDR=$(ray list nodes --format json | jq -r '
  map(select(.state == "ALIVE" and (.resources_total.NPU // 0) > 0)) |
  sort_by(.is_head_node | not) |
  .[0].node_ip
')

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export MEGATRON_BRIDGE_SRC=${MEGATRON_BRIDGE_SRC:-/root/Megatron-Bridge/src/}
export MINDSPEED=${MINDSPEED:-/root/MindSpeed/}
export RELAX=${RELAX:-${DIR}/../../}
export PYTHONPATH=${RELAX}:${MEGATRON_BRIDGE_SRC}:${MINDSPEED}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${DIR}/../models"

# ── entrypoint mode & runtime env ──────────────────────────────────────────
export RELAX_ENTRYPOINT_MODE="ray-job"
RAY_DEBUG=${RAY_DEBUG:-"0"}
RAY_DEBUG_POST_MORTEM=${RAY_DEBUG_POST_MORTEM:-"0"}

# Runtime env for ray-job mode (env inherited from Ray cluster)
export RUNTIME_ENV_JSON="{
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"MASTER_ADDR\": \"${MASTER_ADDR}\",
   \"RAY_DEBUG\": \"${RAY_DEBUG}\",
   \"RAY_DEBUG_POST_MORTEM\": \"${RAY_DEBUG_POST_MORTEM}\",
   \"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES\": \"1\",
   \"PYTORCH_NPU_ALLOC_CONF\": \"expandable_segments:True\"
}
}"

# Enforce terminal success for every local Ray submission from the launcher.
source "${DIR}/ray-status.sh"

echo "=== Ray-job environment ready ==="

# ── delegate to run script (entry-point mode only) ─────────────────────────
if [ -n "$_RAY_JOB_RUN_SCRIPT" ]; then
    echo "=== Launching training script: $_RAY_JOB_RUN_SCRIPT ==="
    exec bash "$_RAY_JOB_RUN_SCRIPT" "$@"
fi
