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
#        bash scripts/entrypoint/ray-job.sh <run-script> [extra-args...]
#      Sets up env, cleans residual processes, then execs the run script.
#
#      Example:
#        bash scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen35-9B-8xgpu-async.sh
#        bash scripts/entrypoint/ray-job.sh scripts/training/text/run-qwen35-9B-8xgpu-async.sh --lr 5e-7
#
#   2) Source mode — no .sh script arg (like local.sh):
#        source scripts/entrypoint/ray-job.sh
#      Sets up env only, so the caller can continue execution.
#
# Environment variables (optional):
#   MEGATRON      - Path to Megatron-LM (default: /root/Megatron-LM/)
#   RELAX         - Path to Relax project (default: ../../)
#   RELAX_KERNEL_CACHE_DIR - Shared directory for portable Inductor/Triton cache deltas.
#   RELAX_KERNEL_CACHE_KEY - Optional graph/topology profile key. Defaults to a hash of the run script and overrides.

# Guard: skip if already sourced by another entrypoint
if [ -n "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    return 0 2>/dev/null || exit 0
fi

# ── mode detection ──────────────────────────────────────────────────────────
# Entry-point mode: directly executed AND first arg is an existing .sh file.
# Otherwise act as a sourced setup script.
_RAY_JOB_RUN_SCRIPT="${_RAY_JOB_RUN_SCRIPT:-}"
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _RAY_JOB_FIRST_ARG="${1:-}"
    if [ -n "$_RAY_JOB_FIRST_ARG" ] && [ -f "$_RAY_JOB_FIRST_ARG" ] && [[ "$_RAY_JOB_FIRST_ARG" == *.sh ]]; then
        _RAY_JOB_RUN_SCRIPT="$_RAY_JOB_FIRST_ARG"
        shift
    else
        echo "Usage: $0 <run-script.sh> [extra-args...]" >&2
        exit 1
    fi
elif [ -z "${_RAY_JOB_RUN_SCRIPT}" ]; then
    for _caller in "${BASH_SOURCE[@]:1}"; do
        if [[ "${_caller}" == */scripts/training/*.sh ]]; then
            _RAY_JOB_RUN_SCRIPT="${_caller}"
            break
        fi
    done
fi

set -eo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=./kernel-cache.sh
source "${DIR}/kernel-cache.sh"

# ── clean up residual Relax/SGLang WORKER processes (NOT ray daemons) ────────
# IMPORTANT: Do NOT pkill ray or run ray stop — the cluster is managed externally.
# kill_for_ray.sh was rewritten to a WHITELIST (positive-match) that only kills
# MegatronTrainRayActor / relax.entrypoints.train / sglang workers, with a hard guard
# against ray start/raylet/gcs_server/dashboard/log_monitor/etc. The old blacklist version
# SIGKILLed `/usr/bin/python3 ... ray start --head` on the head node → GCS reset → jobs
# vanished + SSH/dashboard flapped (see project memory "ray-job.sh kills head"). The new
# script is structurally incapable of hitting a ray daemon, so it is safe to run clusterwide.
echo "=== Cleaning up residual Relax/SGLang worker processes ==="
python ${DIR}/../tools/run_on_each_ray_node.py ${DIR}/../tools/kill_for_ray.sh || echo "failed"

# ── reserve sglang port range from kernel ephemeral pool ────────────────────
# Some worker nodes ship with net.ipv4.ip_local_port_range="10000 65500", which
# includes sglang's well-known port range (15670-15900). Megatron's 294 process
# groups grab ephemeral ports for NCCL/Gloo bootstrap; on those nodes a PG can
# land on a port sglang wants and crash the engine with
# "scheduler_input_port at 15855 is not available in 120 seconds. holder=ray::MegatronTrainRayActor".
# Reserve sglang's range so the kernel never picks it for ephemeral.
echo "=== Reserving sglang port ranges on all GPU nodes ==="
# Reserve two ranges:
#   15000-16800 — sglang port range. SGLang's dp-attention schedulers use ports
#                 starting from ~15100 (base_port + offsets for DP/TP ranks), so
#                 the range must start well below 15400 to cover all scheduler
#                 input/output/NCCL bootstrap ports.
#   30000-32768 — secondary safe zone (fallback if sglang port_base needs adjustment)
python ${DIR}/../tools/run_on_each_ray_node.py --timeout 30 "sysctl -w net.ipv4.ip_local_reserved_ports=15000-20000,30000-32768" || echo "reserve_ports failed (non-fatal)"

# Two run scenarios, distinguished by whether we are inside a ray job driver:
#   A) Entry-point mode — `bash ray-job.sh <run-script>`: this script runs in the
#      launcher shell BEFORE our own `ray job submit`, so every RUNNING relax job
#      in the list is a stale prior job — none is us. Safe to stop them all.
#   B) Driver mode — `ray job submit -- bash ray-job.sh ...`: this script runs
#      inside our own driver, so we MUST exclude our own submission_id or we
#      suicide. Resolve it with two strategies:
#        1) RAY_JOB_SUBMISSION_ID env var — set by Ray ≥ 2.6 via `ray job submit`.
#        2) Fallback: Ray's job_supervisor redirects driver stdout/stderr to
#           /tmp/ray/session_latest/logs/job-driver-<sub_id>.log, so readlink
#           fd 1/2 recovers <sub_id>.
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
echo "=== Own ray submission_id: ${SELF_SUB_ID:-<none — pre-submit entry-point mode>} ==="
# Collect submission_ids of RUNNING relax training jobs (skip placeholder jobs).
# `ray job list` emits one JobDetails(...) per line; jq is unavailable for this
# Python-repr output, so match on the same line: status RUNNING + entrypoint
# contains `relax.entrypoints.train`, then extract submission_id.
_OLD_RELAX_JOBS=$(ray job list 2>/dev/null \
    | grep RUNNING \
    | grep -F 'relax.entrypoints.train' \
    | grep -oP "submission_id='\\K[^']+" || true)
if [ -n "$SELF_SUB_ID" ]; then
    # Driver mode: never stop ourselves.
    _OLD_RELAX_JOBS=$(printf '%s\n' "$_OLD_RELAX_JOBS" | grep -vFx "$SELF_SUB_ID" || true)
fi
if [ -z "$_OLD_RELAX_JOBS" ]; then
    echo "=== No stale relax training jobs to stop ==="
else
    echo "=== Stopping stale relax training jobs ==="
    printf '%s\n' "$_OLD_RELAX_JOBS" | xargs --no-run-if-empty -n1 ray job stop || true
fi
ray serve shutdown -y

# ── remove orphan placement groups ──────────────────────────────────────────
# `ray job stop` / `ray serve shutdown` do NOT delete placement groups. When a
# prior driver dies abnormally (e.g. SIGKILL leaving a zombie), its PGs stay in
# CREATED state forever — GCS still thinks the owner is alive — and keep the GPUs
# reserved, so the next training's PG request hangs in Pending Demands and this
# script's process/job cleanup above cannot free it. At this point (before our
# own `ray job submit`) every CREATED PG is necessarily stale, so remove them.
echo "=== Removing orphan placement groups ==="
python - <<'PY' || echo "orphan PG cleanup failed (non-fatal)"
import ray
from ray.util.placement_group import remove_placement_group, placement_group_table, PlacementGroup

ray.init(address="auto", log_to_driver=False)
try:
    removed = 0
    for pg_id, info in placement_group_table().items():
        if info.get("state") == "CREATED":
            try:
                remove_placement_group(PlacementGroup(ray._raylet.PlacementGroupID(bytes.fromhex(pg_id))))
                removed += 1
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                print(f"  failed to remove PG {pg_id}: {exc!r}")
    print(f"  removed {removed} orphan placement group(s)")
finally:
    ray.shutdown()
PY

set -x

# ── environment setup ───────────────────────────────────────────────────────
# Use the first GPU node as MASTER_ADDR (prefer head node).
# NOTE: assignment is split from `export` on purpose — `export VAR=$(...)`
# always returns 0 (export's own exit code), which would mask failures of
# the command substitution and defeat `set -eo pipefail` set above.
MASTER_ADDR=$(ray list nodes --format json | jq -r '
  map(select(.state == "ALIVE" and (.resources_total.GPU // 0) > 0)) |
  sort_by(.is_head_node | not) |
  .[0].node_ip
')
if [ -z "$MASTER_ADDR" ] || [ "$MASTER_ADDR" = "null" ]; then
    echo "ERROR: failed to resolve MASTER_ADDR (no ALIVE GPU node returned by 'ray list nodes')." >&2
    exit 1
fi
export MASTER_ADDR

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON=${MEGATRON:-/root/Megatron-LM/}
export RELAX=${RELAX:-${DIR}/../../}
export PYTHONPATH=${RELAX}:$MEGATRON:$RELAX:${PYTHONPATH:-}
export MODEL_CONFIG_DIR="${DIR}/../models"

# ── NVLink detection ────────────────────────────────────────────────────────
if nvidia-smi -L 2>/dev/null | grep -q GPU; then
    NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
else
    NVLINK_COUNT=0
fi
if [ "$NVLINK_COUNT" -gt 0 ]; then
    export HAS_NVLINK=1
else
    export HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ── entrypoint mode & runtime env ──────────────────────────────────────────
export RELAX_ENTRYPOINT_MODE="ray-job"
RAY_DEBUG=${RAY_DEBUG:-"0"}
RAY_DEBUG_POST_MORTEM=${RAY_DEBUG_POST_MORTEM:-"0"}

# Runtime env for ray-job mode (env inherited from Ray cluster)
NVSHMEM_LIB_PATH="${NVSHMEM_LIB_PATH:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib}"
# torch lib path is required for fake_int4_quant_cuda.so to find libc10.so / libtorch.so
TORCH_LIB_PATH="${TORCH_LIB_PATH:-/usr/local/lib/python3.12/dist-packages/torch/lib}"
CURRENT_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${NVSHMEM_LIB_PATH}:${TORCH_LIB_PATH}"

relax_kernel_cache_configure "${_RAY_JOB_RUN_SCRIPT}" "$@"
relax_kernel_cache_start_agents cluster

# Cap OMP/MKL/OpenBLAS threads (default 24) to avoid CPU oversubscription when colocating multiple Ray actors per node.
export RUNTIME_ENV_JSON="{
\"worker_process_setup_hook\": \"relax.utils.logging_utils.install_asyncio_noise_filter\",
\"env_vars\": {
   \"PYTHONUNBUFFERED\": \"1\",
   \"PYTHONPATH\": \"${PYTHONPATH}\",
   \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
   \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
   \"OMP_NUM_THREADS\": \"${OMP_NUM_THREADS:-24}\",
   \"MKL_NUM_THREADS\": \"${MKL_NUM_THREADS:-24}\",
   \"OPENBLAS_NUM_THREADS\": \"${OPENBLAS_NUM_THREADS:-24}\",
   \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
   \"MASTER_ADDR\": \"${MASTER_ADDR}\",
   \"RAY_DEBUG\": \"${RAY_DEBUG}\",
   \"RAY_DEBUG_POST_MORTEM\": \"${RAY_DEBUG_POST_MORTEM}\",
   \"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK\": \"${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-32}\",
   \"NVSHMEM_DISABLE_NCCL\": \"${NVSHMEM_DISABLE_NCCL:-1}\",
   \"SGLANG_HEALTH_CHECK_TIMEOUT\": \"${SGLANG_HEALTH_CHECK_TIMEOUT:-180}\",
   \"INDEXER_ROPE_NEOX_STYLE\": \"${INDEXER_ROPE_NEOX_STYLE:-0}\",
   \"NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME\": \"${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${NCCL_SOCKET_IFNAME}}\",
   \"NVTE_USE_CUTLASS_GROUPED_GEMM\": \"${NVTE_USE_CUTLASS_GROUPED_GEMM:-0}\",
   \"NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK\": \"${NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK:-1}\",
   \"LD_LIBRARY_PATH\": \"${CURRENT_LD_LIBRARY_PATH}\"
}
}"

relax_kernel_cache_inject_runtime_env

echo "=== Ray-job environment ready ==="

# ── delegate to run script (entry-point mode only) ─────────────────────────
if [ -n "$_RAY_JOB_RUN_SCRIPT" ]; then
    echo "=== Launching training script: $_RAY_JOB_RUN_SCRIPT ==="
    exec bash "$_RAY_JOB_RUN_SCRIPT" "$@"
fi
