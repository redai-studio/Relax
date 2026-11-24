#!/bin/sh
# Training-library preloads (NCCL + jemalloc) can crash the CUDA compiler.
# Unset only in this compiler process; the calling training process is unchanged.
unset LD_PRELOAD
exec "$(dirname "$(readlink -f "$0")")/nvcc.real" "$@"
