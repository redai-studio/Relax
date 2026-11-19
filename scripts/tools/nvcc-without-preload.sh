#!/bin/sh
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Keep training preloads out of DeepEP compiler subprocesses.
unset LD_PRELOAD
nvcc=${RELAX_DEEPEP_NVCC:-}
if [ -z "$nvcc" ]; then
    # Use the CUDA installation on this worker, as DeepEP does.
    cuda_home=${CUDA_HOME:-${CUDA_PATH:-}}
    if [ -n "$cuda_home" ]; then
        nvcc="$cuda_home/bin/nvcc"
    else
        nvcc=$(command -v nvcc) || nvcc=/usr/local/cuda/bin/nvcc
    fi
fi
exec "$nvcc" "$@"
