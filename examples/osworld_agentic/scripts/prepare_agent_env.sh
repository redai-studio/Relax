#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Prepare the default shared Agent environment once, before submitting training.
set -euo pipefail
set +x
: "${DATA_DIR:?Set DATA_DIR to a shared persistent directory}"
: "${OSWORLD_REPO:?Set OSWORLD_REPO to the external OSWorld checkout}"

agent_venv="${DATA_DIR}/osworld-agent-venv"
mkdir -p "${DATA_DIR}"
# Serialize first-time installation when several launchers share DATA_DIR.
exec 9>"${DATA_DIR}/.osworld-agent-venv.lock"
flock 9
if [ ! -f "${agent_venv}/.relax-agent-ready" ]; then
    if [ ! -x "${agent_venv}/bin/python" ]; then
        python3 -m venv --system-site-packages "${agent_venv}"
    fi
    # Reuse the training image's torch packages without upgrading its CUDA stack.
    constraints=$(mktemp)
    trap 'rm -f "${constraints}"' EXIT
    python3 - <<'PY' > "${constraints}"
from importlib.metadata import PackageNotFoundError, version

for name in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        pass
PY
    "${agent_venv}/bin/python" -m pip install \
        -c "${constraints}" -r "${OSWORLD_REPO}/requirements.txt" qwen-agent fastmcp
    touch "${agent_venv}/.relax-agent-ready"
fi
if [ ! -x "${agent_venv}/bin/python" ]; then
    echo "Shared Agent Python is missing: ${agent_venv}/bin/python" >&2
    exit 1
fi
echo "Using shared Agent environment: ${agent_venv}"
