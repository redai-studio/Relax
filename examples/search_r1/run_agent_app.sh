#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

exec python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
