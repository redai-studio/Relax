#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

GYM_ROOT="${GYM_ROOT:-/opt/nemo-gym}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/nemo-gym}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
GYM_DATA="${GYM_ROOT}/benchmarks/gsm8k/data/gsm8k_benchmark.jsonl"
SHARED_GYM_DATA="${OUTPUT_DIR}/gsm8k_benchmark.jsonl"
RELAX_DATA="${OUTPUT_DIR}/gsm8k_relax.jsonl"

cd "${GYM_ROOT}"
"${GYM_ROOT}/.venv/bin/gym" eval prepare --benchmark gsm8k

mkdir -p "${OUTPUT_DIR}"
cp "${GYM_DATA}" "${SHARED_GYM_DATA}"

/usr/bin/python3 \
    "${EXAMPLE_DIR}/scripts/convert_dataset.py" \
    --input "${SHARED_GYM_DATA}" \
    --output "${RELAX_DATA}"

awk 'NF { count++ } END { print count, FILENAME }' "${SHARED_GYM_DATA}"
awk 'NF { count++ } END { print count, FILENAME }' "${RELAX_DATA}"
