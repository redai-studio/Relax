#!/usr/bin/env bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/tools/convert_qwen35_mtp_checkpoint_to_hf.sh \
    INPUT_CHECKPOINT_DIR ORIGIN_HF_DIR [OUTPUT_HF_DIR]

Environment variables:
  PYTHON_BIN  Python executable to use (default: python3)
  FORCE=1     Allow the converter to overwrite an existing output directory

If OUTPUT_HF_DIR is omitted, the output is written next to the input checkpoint
as INPUT_CHECKPOINT_DIR-hf.
EOF
}

if (( $# < 2 || $# > 3 )); then
    usage >&2
    exit 2
fi

INPUT_DIR="${1%/}"
ORIGIN_HF_DIR="${2%/}"
OUTPUT_DIR="${3:-${INPUT_DIR}-hf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${INPUT_DIR}/.metadata" ]]; then
    echo "error: ${INPUT_DIR} is not a torch distributed checkpoint directory" >&2
    exit 1
fi
if [[ ! -f "${ORIGIN_HF_DIR}/config.json" ]]; then
    echo "error: ${ORIGIN_HF_DIR} is not a Hugging Face model directory" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${SCRIPT_DIR}/convert_torch_dist_to_hf_bridge.py"
CONVERT_ARGS=(
    --input-dir "${INPUT_DIR}"
    --origin-hf-dir "${ORIGIN_HF_DIR}"
    --output-dir "${OUTPUT_DIR}"
)
if [[ "${FORCE:-0}" == "1" ]]; then
    CONVERT_ARGS+=(--force)
fi

echo "Input checkpoint: ${INPUT_DIR}"
echo "Original HF model: ${ORIGIN_HF_DIR}"
echo "Output HF model: ${OUTPUT_DIR}"
"${PYTHON_BIN}" "${CONVERTER}" "${CONVERT_ARGS[@]}"

"${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path


output_dir = Path(sys.argv[1])
with (output_dir / "config.json").open() as f:
    config = json.load(f)
with (output_dir / "model.safetensors.index.json").open() as f:
    weight_index = json.load(f)

text_config = config.get("text_config", config)
mtp_num_layers = text_config.get("mtp_num_hidden_layers", 0)
mtp_keys = [key for key in weight_index["weight_map"] if "mtp" in key.lower()]
if mtp_num_layers <= 0:
    raise RuntimeError("converted config does not enable MTP")
if not mtp_keys:
    raise RuntimeError("converted checkpoint index does not contain MTP weights")

print(f"Validated HF export: mtp_num_hidden_layers={mtp_num_layers}, mtp_keys={len(mtp_keys)}")
PY

echo "Conversion complete: ${OUTPUT_DIR}"
