#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Entrypoint / source helper for Ray Job tasks.
# The Ray cluster is already running. This script MUST NOT kill ray or stop the
# cluster. It only cleans up residual python/sglang processes and then sets up
# the environment for running training against an existing Ray cluster.
#
# Submissions are serialised machine-wide by an exclusive flock on
# /root/autodl-tmp/relax-ray-gpu.lock; a second submission fails fast instead of
# interleaving its cluster-wide cleanup with an in-flight submission. The lock
# is held by the submitting shell, so a SIGKILLed submission releases it while
# its Ray job may still run — see the KNOWN LIMITATION note below.
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
#   RELAX_GPU_LOCK_WAIT - Seconds to wait for the shared GPU lock at
#                         /root/autodl-tmp/relax-ray-gpu.lock before failing fast
#                         (default: 0 = fail immediately when another job holds it).
#   RELAX_GPU_LOCK_PROJECT / RELAX_GPU_LOCK_COMMAND - Optional tags recorded in
#                         the lock's holder sidecar for the other party.

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

# ── physical GPU-cluster lock ───────────────────────────────────────────────
# One Ray cluster / four GPUs are shared by several projects, and the cleanup
# below is cluster-wide (it stops competing training jobs and removes their
# placement groups). Coordination by convention alone already cost hours, so
# every submission through this launcher must own an exclusive flock on
# ${RELAX_GPU_LOCK_FILE} for the whole cleanup + submit + wait window. The lock
# file is created on first use and is never unlinked, so all participants always
# lock the same inode. Contention fails fast by default: the second submission
# prints the current holder (sidecar ${RELAX_GPU_LOCK_FILE}.holder) and exits 75
# (EX_TEMPFAIL) BEFORE any cluster cleanup, instead of queueing forever.
# RELAX_GPU_LOCK_WAIT=<seconds> opts into a short bounded wait instead.
#
# The lock lives on fd 200 and is inherited across `exec bash <run-script>` and
# by the `ray job submit` client that the run script waits on, so the submitting
# process tree holds it for the whole cleanup + submit + wait window and the
# kernel releases it on every exit path: normal completion, any error,
# SIGINT/SIGTERM/SIGHUP (shell traps below), and even SIGKILL where no trap can
# run.
#
# KNOWN LIMITATION — this lock serialises SUBMISSIONS, not Ray jobs. A Ray
# driver/worker is spawned by the raylet, NOT by the submitting shell, so it is
# not a descendant and does not inherit fd 200. If the submitting shell is
# SIGKILLed the kernel drops the flock while the submitted Ray job keeps
# running, and a later submission then takes a free lock and its cluster-wide
# cleanup may stop that live job. A free lock therefore does NOT prove the
# cluster is idle: check `ray job list` / `ray job status` before assuming so.
RELAX_GPU_LOCK_FILE="${RELAX_GPU_LOCK_FILE:-/root/autodl-tmp/relax-ray-gpu.lock}"
RELAX_GPU_LOCK_HOLDER_FILE="${RELAX_GPU_LOCK_FILE}.holder"
RELAX_GPU_LOCK_WAIT="${RELAX_GPU_LOCK_WAIT:-0}"
_RELAX_GPU_LOCK_OWNED=0

# Idempotent release: drop the sidecar only if it is still ours, then close
# fd 200 so the kernel releases the flock. Used by the EXIT and signal traps.
_relax_gpu_lock_release() {
    if [ "${_RELAX_GPU_LOCK_OWNED}" != "1" ]; then
        return 0
    fi
    if [ -f "${RELAX_GPU_LOCK_HOLDER_FILE}" ] \
        && grep -qx "pid=$$" "${RELAX_GPU_LOCK_HOLDER_FILE}" 2>/dev/null; then
        rm -f "${RELAX_GPU_LOCK_HOLDER_FILE}"
    fi
    exec 200>&- 2>/dev/null || true
    _RELAX_GPU_LOCK_OWNED=0
    return 0
}

# Show who holds the lock so the blocked party can act instead of guessing.
_relax_gpu_lock_report_holder() {
    if [ -s "${RELAX_GPU_LOCK_HOLDER_FILE}" ]; then
        echo "  current holder (${RELAX_GPU_LOCK_HOLDER_FILE}):" >&2
        sed 's/^/    /' "${RELAX_GPU_LOCK_HOLDER_FILE}" >&2
        _holder_pid="$(sed -n 's/^pid=//p' "${RELAX_GPU_LOCK_HOLDER_FILE}" | head -n 1)"
        if [ -n "${_holder_pid}" ] && ! kill -0 "${_holder_pid}" 2>/dev/null; then
            echo "    (recorded pid ${_holder_pid} is gone; this holder info is stale)" >&2
        fi
    else
        echo "  current holder: unknown (no holder info recorded, but the flock is held)" >&2
    fi
}

# Re-entrancy guard: a nested or repeated invocation must reuse the ancestor's
# lock, never contend with itself. Any one of these is sufficient:
#   1. the RELAX_ENTRYPOINT_MODE guard at the top already short-circuits nested
#      ray-job.sh invocations;
#   2. RELAX_GPU_LOCK_HELD, exported below by the invocation that owns the lock;
#   3. fd 200 inherited from that invocation and still pointing at the lock file
#      (re-flocking an inherited descriptor is a no-op success, never a block).
if [ -n "${RELAX_GPU_LOCK_HELD:-}" ] \
    || [ "$(readlink -m "/proc/self/fd/200" 2>/dev/null || true)" = "$(readlink -m "${RELAX_GPU_LOCK_FILE}")" ]; then
    echo "=== GPU lock already held by this process tree (${RELAX_GPU_LOCK_HELD:-inherited fd 200}); reusing it ==="
else
    if ! exec 200>>"${RELAX_GPU_LOCK_FILE}"; then
        echo "ERROR: cannot open GPU lock file ${RELAX_GPU_LOCK_FILE}; refusing to touch the cluster." >&2
        exit 1
    fi
    if [ "${RELAX_GPU_LOCK_WAIT}" -gt 0 ] 2>/dev/null; then
        flock -w "${RELAX_GPU_LOCK_WAIT}" 200 && _relax_gpu_lock_ok=1 || _relax_gpu_lock_ok=0
    else
        flock -n 200 && _relax_gpu_lock_ok=1 || _relax_gpu_lock_ok=0
    fi
    if [ "${_relax_gpu_lock_ok}" != "1" ]; then
        echo "ERROR: another job holds the GPU lock ${RELAX_GPU_LOCK_FILE} — refusing to submit (no cluster cleanup was performed)." >&2
        _relax_gpu_lock_report_holder
        if [ "${RELAX_GPU_LOCK_WAIT}" -gt 0 ] 2>/dev/null; then
            echo "  waited ${RELAX_GPU_LOCK_WAIT}s; retry later or unset RELAX_GPU_LOCK_WAIT to fail immediately" >&2
        fi
        exec 200>&- 2>/dev/null || true
        exit 75
    fi
    _RELAX_GPU_LOCK_OWNED=1
    {
        echo "pid=$$"
        echo "ppid=${PPID}"
        echo "host=$(hostname 2>/dev/null || echo unknown)"
        echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "cwd=$(pwd)"
        echo "project=${RELAX_GPU_LOCK_PROJECT:-${_RAY_JOB_RUN_SCRIPT:-<source-mode>}}"
        echo "run_script=${_RAY_JOB_RUN_SCRIPT:-<none>}"
        echo "submit_command=${RELAX_GPU_LOCK_COMMAND:-bash ${0} ${_RAY_JOB_RUN_SCRIPT} $*}"
        echo "note=lock guards concurrent submissions only; a SIGKILLed submitter releases it while its Ray job may still run, so check 'ray job list' before treating a free lock as an idle cluster"
    } > "${RELAX_GPU_LOCK_HOLDER_FILE}"
    export RELAX_GPU_LOCK_HELD="$$"
    echo "=== GPU lock acquired: ${RELAX_GPU_LOCK_FILE} (pid $$, project ${RELAX_GPU_LOCK_PROJECT:-${_RAY_JOB_RUN_SCRIPT:-source-mode}}) ==="
    # Held for the rest of this shell's life and across `exec`. NOTE: `exec`
    # discards these traps, but the inherited fd still holds the flock, so the
    # lock is released by the kernel when the exec'd job tree exits.
    trap '_relax_gpu_lock_release' EXIT
    trap '_relax_gpu_lock_release; trap - EXIT; exit 130' INT
    trap '_relax_gpu_lock_release; trap - EXIT; exit 143' TERM
    trap '_relax_gpu_lock_release; trap - EXIT; exit 129' HUP
fi

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
