#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

# Export a supported torch_dist checkpoint using one CPU process.
# CHECKPOINT_PATH=/path/to/iter_0000100 MODEL_PATH=/path/to/original-hf \
# OUTPUT_DIR=/path/to/new-hf bash scripts/tools/convert_checkpoint_to_hf_cpu.sh
# Add --dry-run to check metadata/mappings first. --help lists memory controls.
# --read-workers 4 --load-batch-size-gb 2 control checkpoint I/O independently
# of --cpu-threads 2. Batches shrink to fit --memory-budget-gb; individual
# conversions may exceed the batch target. --max-shard-size-gb controls output.
# Loading overlaps writing when the budget permits; --no-prefetch disables it.
# Reader workspace is reserved per batch and capped by its source file count.
# Frozen tensors are copied through a bounded buffer without materialization.
# OUTPUT_DIR directly contains the HF model. Existing directories are refused.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES=""
exec "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/convert_checkpoint_to_hf_cpu.py" "$@"
