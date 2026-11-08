#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Generates RUNTIME_ENV_JSON for `ray job submit --runtime-env-json=` on PPU.
#
# Source it from a run-*.sh **after** the shell-level env vars are exported: the
# ${...} below expand at source time.  The image is a plain /usr/local install,
# so site-packages is spelled out rather than derived from CONDA_PREFIX.
#
# PPU notes:
#   - Masquerades as CUDA: `nccl` resolves to the PCCL wrapper; PCCL_* knobs are
#     listed by `strings /usr/local/pccl/lib/libpccl.so | grep -E '^PCCL_[A-Z_]+$'`.
#     Kunlunxin-only variables (XMLIR_*, BKCL_*, XSGL_*, XPU_*, ...) are dropped.
#   - Weight offload: pause() keeps a region's content only when it was captured
#     with `enable_cpu_backup`, hence --sglang-enable-weights-cpu-backup on the
#     VMM line.  Decode-graph pages live under the `cuda_graph` tag; replaying
#     them after release segfaults in hggcGraphLaunch unless HGGC_DEVICE_GRAPH=2.
#   - DeepEP's 100 s default is below the first step's uneven compile time -> 600 s.
#
# Usage (from a run script, after all `export`s and before `ray job submit`):
#   source "${SCRIPT_DIR}/../../entrypoint/runtime-env-ppu.sh"
#   ray job submit ... ${RUNTIME_ENV_JSON:+--runtime-env-json="${RUNTIME_ENV_JSON}"} ...
#
# Extension hook (optional): set `EXTRA_ENV_VARS_JSON` before sourcing to append
# `"KEY": "VAL"` pairs to `env_vars` (no leading or trailing comma).

# ── infer cpu threads num ──────────────────────────────────────────────────
NUM_GPUS_TOTAL="${NUM_GPUS_TOTAL:-8}"
if [ -z "${CPU_THREADS_PER_ACTOR:-}" ]; then
    _cores_per_socket=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket:/ {gsub(/ /,"",$2); print $2; exit}')
    _sockets=$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\):/ {gsub(/ /,"",$2); print $2; exit}')
    if [ -n "$_cores_per_socket" ] && [ -n "$_sockets" ] && [ "${_sockets}" -gt 0 ]; then
        _total_phys=$((_cores_per_socket * _sockets))
        CPU_THREADS_PER_ACTOR=$((_total_phys / NUM_GPUS_TOTAL))
        # clamp to [4, 64], avoid bad values
        [ "${CPU_THREADS_PER_ACTOR}" -lt 4 ] && CPU_THREADS_PER_ACTOR=4
        [ "${CPU_THREADS_PER_ACTOR}" -gt 64 ] && CPU_THREADS_PER_ACTOR=64
    else
        CPU_THREADS_PER_ACTOR=24
    fi
fi
echo "[cpu-threads] NUM_GPUS_TOTAL=${NUM_GPUS_TOTAL} CPU_THREADS_PER_ACTOR=${CPU_THREADS_PER_ACTOR}"
export CPU_THREADS_PER_ACTOR

# `PYTORCH_CUDA_ALLOC_CONF` defaults to expandable_segments:False, which both
# saver lines accept; a script that needs it (VMM only) opts in per actor via
# --train-env-vars.
#
# Transformer Engine grouped GEMM: the wheel (flytiger-eco/TransformerEngine-for-sail
# 2.18+v0.1.0, NVTE_CUDA_ARCHS="80;89") only has the CUTLASS path, so forward and
# dgrad use CUTLASS-RTC, wgrad falls back to cuBLAS, and RTC finds its headers
# under site-packages/include/cutlass3.
export RUNTIME_ENV_JSON="{
  \"worker_process_setup_hook\": \"relax.utils.logging_utils.install_asyncio_noise_filter\",
  \"env_vars\": {
    \"PYTHONPATH\": \"${WORKDIR}/Relax:${MEGATRON:-/root/Megatron-LM}:${SCRIPT_DIR}:$PYTHONPATH\",
    \"PYTHONUNBUFFERED\": \"1\",
    \"LD_LIBRARY_PATH\":\"${PPU_SITE_PACKAGES:-/usr/local/lib/python3.12/site-packages}:${PPU_SDK:-/usr/local/PPU_SDK}/lib:${PPU_SDK:-/usr/local/PPU_SDK}/CUDA_SDK/lib64:/usr/local/pccl/nccl_wrapper/lib:/usr/local/pccl/lib:$LD_LIBRARY_PATH\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"${CUDA_DEVICE_MAX_CONNECTIONS:-1}\",
    \"OPENBLAS_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"OMP_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"MKL_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"NUMEXPR_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"TOKENIZERS_PARALLELISM\": \"true\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"NCCL_IB_HCA\": \"${NCCL_IB_HCA:-mlx5}\",
    \"NCCL_IB_GID_INDEX\": \"${NCCL_IB_GID_INDEX:-3}\",
    \"NCCL_DEBUG\": \"${NCCL_DEBUG:-WARN}\",
    \"PCCL_DISABLE_FC_GRAPH\": \"${PCCL_DISABLE_FC_GRAPH:-1}\",
    \"PCCL_CE_ENABLE_AR\": \"${PCCL_CE_ENABLE_AR:-0}\",
    \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}\",
    \"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\",
    \"RAY_OVERRIDE_JOB_RUNTIME_ENV\": \"1\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME:-eth0}\",
    \"TP_SOCKET_IFNAME\": \"${TP_SOCKET_IFNAME:-eth0}\",
    \"SGLANG_HEALTH_CHECK_TIMEOUT\": \"${SGLANG_HEALTH_CHECK_TIMEOUT:-180}\",
    \"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK\": \"${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-32}\",
    \"DEEPEP_TIMEOUT_SECONDS\": \"${DEEPEP_TIMEOUT_SECONDS:-600}\",
    \"SGLANG_IS_FLASHINFER_AVAILABLE\": \"false\",
    \"PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT\": \"${PYTORCH_ENABLE_PPU_ELEMENTWISE_OPT:-1}\",
    \"NVTE_USE_CUTLASS_GROUPED_GEMM\": \"${NVTE_USE_CUTLASS_GROUPED_GEMM:-1}\",
    \"NVTE_USE_CUTLASS_GROUPED_GEMM_RTC\": \"${NVTE_USE_CUTLASS_GROUPED_GEMM_RTC:-1}\",
    \"NVTE_FORCE_RTC_NO_FALLBACK\": \"${NVTE_FORCE_RTC_NO_FALLBACK:-1}\",
    \"NVTE_CUTLASS3_INCLUDE_DIR\": \"${NVTE_CUTLASS3_INCLUDE_DIR:-/usr/local/lib/python3.12/site-packages/include/cutlass3}\",
    \"NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK\": \"${NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK:-1}\",
    \"SGLANG_NUMA_BIND_V2\": \"${SGLANG_NUMA_BIND_V2:-1}\",
    \"SGLANG_SET_CPU_AFFINITY\": \"${SGLANG_SET_CPU_AFFINITY:-0}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}\"${EXTRA_ENV_VARS_JSON:+,
    ${EXTRA_ENV_VARS_JSON}}
   }
}"

# Merges compiler-cache entries; no-op unless the caller configured a cache.
relax_kernel_cache_inject_runtime_env
