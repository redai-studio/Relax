#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Per-session Search-QA agent launcher. Relax spawns this once per rollout session
# via `--agent-command`, injecting RELAX_BASE_URL / RELAX_SESSION_ID /
# RELAX_INPUT_JSON / RELAX_OUTPUT_JSON.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
if [ -f "${CONDA_HOME}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_HOME}/etc/profile.d/conda.sh"
    conda activate "${SEARCH_CONDA_ENV:-relax-opd-search}"
fi

export PYTHONPATH="${RELAX_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export SEARCH_RETRIEVAL_URL="${SEARCH_RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

PYTHON_BIN="${SEARCH_PYTHON_BIN:-python3}"
exec "${PYTHON_BIN}" -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
