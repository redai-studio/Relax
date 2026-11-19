#!/bin/sh
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Keep training preloads out of DeepEP compiler subprocesses.
unset LD_PRELOAD
exec "${RELAX_DEEPEP_NVCC:?missing worker NVCC path}" "$@"
