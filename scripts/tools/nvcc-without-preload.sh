#!/bin/sh
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Keep training preloads out of DeepEP compiler subprocesses.
unset LD_PRELOAD
nvcc_real="${RELAX_NVCC_REAL:?Set RELAX_NVCC_REAL to the original NVCC executable}"
# The target may already be an image-installed NVCC wrapper.
unset RELAX_NVCC_REAL
exec "$nvcc_real" "$@"
