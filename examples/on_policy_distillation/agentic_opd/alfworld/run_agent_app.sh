#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Per-session ALFWorld agent launcher. Relax spawns this once per rollout session
# via `--agent-command`, injecting RELAX_BASE_URL / RELAX_SESSION_ID /
# RELAX_INPUT_JSON / RELAX_OUTPUT_JSON. ALFWorld lives in a dedicated conda env
# (decoupled from the trainer env), so we activate it here.
#
# CONDA_HOME / ALFWORLD_CONDA_ENV / ALFWORLD_DATA are forwarded from the training
# script via `--agent-env`, so they are already in this process's environment on
# every node (multi-node safe). The defaults below are only a single-node fallback.

# Activate the ALFWorld environment (see README.md for setup). A plain venv is
# used when ALFWORLD_VENV points at one; otherwise fall back to conda.
if [ -n "${ALFWORLD_VENV:-}" ]; then
    source "${ALFWORLD_VENV}/bin/activate"
else
    CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
    source "${CONDA_HOME}/etc/profile.d/conda.sh"
    conda activate "${ALFWORLD_CONDA_ENV:-relax-opd-alfworld}"
fi

export ALFWORLD_DATA="${ALFWORLD_DATA:-/root/alfworld}"
export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

python -m app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
