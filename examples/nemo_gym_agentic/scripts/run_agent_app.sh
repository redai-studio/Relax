#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${EXAMPLE_DIR}"

: "${RELAX_INPUT_JSON:?RELAX_INPUT_JSON must be set by Relax}"
: "${RELAX_OUTPUT_JSON:?RELAX_OUTPUT_JSON must be set by Relax}"
: "${RELAX_SESSION_ID:?RELAX_SESSION_ID must be set by Relax}"
: "${RELAX_SESSION_IO_DIR:?RELAX_SESSION_IO_DIR must be set by Relax}"
: "${RELAX_BASE_URL:?RELAX_BASE_URL must be set by Relax}"
: "${NEMO_GYM_URL:?NEMO_GYM_URL must point to the shared NeMo Gym gateway}"

exec python -m app.client \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
