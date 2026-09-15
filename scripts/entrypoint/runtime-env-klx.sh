#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Generates RUNTIME_ENV_JSON for `ray job submit --runtime-env-json=`.
#
# This file is meant to be *sourced* by a run-*.sh script **after** all
# shell-level env vars (WORKDIR, CONDA_PREFIX, NCCL_SOCKET_IFNAME,
# GLOO_SOCKET_IFNAME, TP_SOCKET_IFNAME, BKCL_RDMA_NICS, XMLIR_MEMCPY_RETRY_SYNC,
# etc.) have been exported, so the ${...} references below expand against the
# caller's environment at source time.
#
# Usage (from a run script, after all `export`s and before `ray job submit`):
#   source "${SCRIPT_DIR}/../../entrypoint/runtime-env-klx.sh"
#   ray job submit ... ${RUNTIME_ENV_JSON:+--runtime-env-json="${RUNTIME_ENV_JSON}"} ...
#
# Extension hook (optional):
#   Callers may set `EXTRA_ENV_VARS_JSON` **before** sourcing this file to
#   inject additional entries into `env_vars` without polluting the shared
#   template. The value must be a comma-separated fragment of `"KEY": "VAL"`
#   pairs (no leading or trailing comma). Example:
#     EXTRA_ENV_VARS_JSON="\"WANDB_SILENT\": \"true\",
#         \"WANDB_DISABLE_GIT\": \"true\""
#     source runtime-env-klx.sh

# ── infer cpu threads num ──────────────────────────────────────────────────
NUM_GPUS_TOTAL="${NUM_GPUS_TOTAL:-8}"
if [ -z "${CPU_THREADS_PER_ACTOR:-}" ]; then
    _cores_per_socket=$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket:/ {gsub(/ /,"",$2); print $2; exit}')
    _sockets=$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\):/ {gsub(/ /,"",$2); print $2; exit}')
    if [ -n "${_cores_per_socket}" ] && [ -n "${_sockets}" ] && [ "${_sockets}" -gt 0 ]; then
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

export RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${WORKDIR}/TransferQueue:${WORKDIR}/Megatron-LM/:${SCRIPT_DIR}:${WORKDIR}/Megatron-Bridge/src/:$PYTHONPATH\",
    \"LD_LIBRARY_PATH\":\"${CONDA_PREFIX}/xcudart/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/xtorch_ops:${CONDA_PREFIX}/lib/python3.10/site-packages/torch_xmlir/:${CONDA_PREFIX}/lib/python3.10/site-packages/torch_xmlir/xre/so\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"OPENBLAS_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"OMP_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"MKL_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"NUMEXPR_NUM_THREADS\": \"${CPU_THREADS_PER_ACTOR}\",
    \"TOKENIZERS_PARALLELISM\": \"true\",
    \"NCCL_CUMEM_ENABLE\": \"0\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME:-eth0}\",
    \"NCCL_IB_HCA\": \"mlx5\",
    \"NCCL_IB_GID_INDEX\": \"3\",
    \"CUDA_DEVICE_ORDER\": \"OAM_ID\",
    \"CUDA_ENABLE_P2P_NO_UVA\": \"0\",
    \"CUDA_FAKE_UVA_ENABLE\": \"1\",
    \"CUDART_DUMMY_REGISTER\": \"1\",
    \"XPU_FORCE_USERMODE_LAUNCH\": \"1\",
    \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}\",
    \"XPU_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}\",
    \"XMLIR_FA_GEMM_TYPE\": \"float\",
    \"XBLAS_FC_HBM_VERSION\": \"40\",
    \"XMLIR_ENABLE_FAST_FC\": \"${XMLIR_ENABLE_FAST_FC:-0}\",
    \"XMLIR_USE_HYDRA_LINEAR\": \"${XMLIR_USE_HYDRA_LINEAR:-0}\",
    \"XTE_DISABLE_MOE_DW_FUSION\": \"${XTE_DISABLE_MOE_DW_FUSION:-0}\",
    \"XMLIR_PARALLEL_SAVE_MEMORY\": \"false\",
    \"XMLIR_DISABLE_CUDA_ALLOCATOR\": \"false\",
    \"XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL\": \"0\",
    \"XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL\": \"False\",
    \"XMLIR_DUMP_FALLBACK_OP_LIST_BOOL\": \"true\",
    \"XMLIR_DIST_ASYNC_ISEND_IRECV\": \"false\",
    \"XMLIR_BATCH_PARALLEL\": \"false\",
    \"XMLIR_MATMUL_FAST_MODE\": \"${XMLIR_MATMUL_FAST_MODE:-0}\",
    \"XPU_FORCE_SHARED_DEVICE_CONTEXT\": \"1\",
    \"BKCL_RDMA_PROXY_DISABLE\": \"1\",
    \"BKCL_USE_AR\": \"1\",
    \"BKCL_RING_OPT\": \"1\",
    \"BKCL_FLAT_RING\": \"1\",
    \"BKCL_CCIX_RING\": \"1\",
    \"BKCL_TREE_THRESHOLD\": \"1048576\",
    \"BKCL_CCIX_BUFFER_GM\": \"1\",
    \"BKCL_FORCE_L3_RDMA\": \"0\",
    \"BKCL_RING_BUFFER_GM\": \"1\",
    \"BKCL_ENABLE_XDR\": \"1\",
    \"BKCL_RDMA_FORCE_TREE\": \"1\",
    \"BKCL_XLINK_D2D\": \"0\",
    \"BKCL_XLINK_ETH\": \"0\",
    \"BKCL_XLINK_C2C\": \"1\",
    \"BKCL_TRANS_UNSUPPORTED_DATATYPE\": \"1\",
    \"BKCL_KL3_TURBO_MODE\": \"1\",
    \"BKCL_RING_BUFFER_SIZE\": \"2097152\",
    \"ALLREDUCE_ASYNC\": \"false\",
    \"ALLGATHER_ASYNC\": \"false\",
    \"ALLREDUCE_FUSION\": \"0\",
    \"BKCL_TIMEOUT\": \"400000\",
    \"CUDA_DISABLE_PRINTF\": \"1\",
    \"BKCL_RDMA_VERBS\": \"1\",
    \"BKCL_RDMA_NICS\": \"${BKCL_RDMA_NICS:-"bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7"}\",
    \"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES\": \"1\",
    \"TORCH_XCCL_DEFAUTL_PG_TIMEOUT_MILSEC\": \"7200000\",
    \"CUDA_ERROR_LEVEL\": \"0\",
    \"HYDRA_FULL_ERROR\": \"1\",
    \"XMLIR_ENABLE_NEW_PG\": \"1\",
    \"TORCH_XCCL_HEARTBEAT_TIMEOUT_SEC\": \"1800\",
    \"TORCH_XCCL_ENABLE_TIMING\": \"1\",
    \"TORCH_FR_BUFFER_SIZE\": \"2000\",
    \"TORCH_XCCL_TRACE_BUFFER_SIZE\": \"2000\",
    \"VERL_LOGGING_LEVEL\": \"DEBUG\",
    \"BKCL_ALL_TO_ALL_OPT\": \"1\",
    \"SGLANG_IS_FLASHINFER_AVAILABLE\": \"false\",
    \"USE_MOE_FC_V3\": \"1\",
    \"XMLIR_DIST_SINGLETON_STREAM\": \"1\",
    \"SGL_CPU_QUANTIZATION\": \"0\",
    \"XSGL_ENABLE_MEM_SAVER\": \"0\",
    \"XPU_ENABLE_CTX_LAZY_INIT\": \"1\",
    \"XPU_SUPPORT_IPC_EVENT\": \"1\",
    \"XSGL_USE_TORCH_CAUSAL_CONV\": \"1\",
    \"TRACE_WEIGHT_PATHS\": \"0\",
    \"TRITON_SKIP_AUTOTUNE\": \"1\",
    \"FLA_USE_NAIVE\": \"1\",
    \"FORCE_DISABLE_FLA\": \"1\",
    \"DISABLE_CAST_CACHE\": \"1\",
    \"FORCE_NN_LINEAR\": \"${FORCE_NN_LINEAR:-0}\",
    \"USE_FUSED_GATED_DELTA_RULE\": \"1\",
    \"XSGL_TRANSPOSE_SSM_STATE\": \"1\",
    \"XSGL_TRANSPOSE_CONV_STATE\": \"1\",
    \"XSGL_FUSE_SPLIT_NORM_ROPE_NEOX\": \"1\",
    \"XSGL_MOE_UNSTABLE_TOPK\": \"${XSGL_MOE_UNSTABLE_TOPK:-1}\",
    \"XSGL_QUANT_ALL_GATHER\": \"${XSGL_QUANT_ALL_GATHER:-0}\",
    \"XPU_FLASH_ATTENTION_DECODER_USE_BALANCE\": \"1\",
    \"XMLIR_FORCE_USE_XPU_GRAPH\": \"1\",
    \"RAY_OVERRIDE_JOB_RUNTIME_ENV\":\"1\",
    \"RELAX_SKIP_TORCH_MEMORY_SAVER\": \"1\",
    \"XMLIR_MEMCPY_RETRY_SYNC\": \"${XMLIR_MEMCPY_RETRY_SYNC:-true}\",
    \"HYDRAX_USE_PROTEUS\": \"0\",
    \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME:-"eth0"}\",
    \"TP_SOCKET_IFNAME\": \"${TP_SOCKET_IFNAME:-"eth0"}\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"1\",
    \"HEALTH_GENERATE_TOPK\": \"-1\"${EXTRA_ENV_VARS_JSON:+,
    ${EXTRA_ENV_VARS_JSON}}
   }
}"
