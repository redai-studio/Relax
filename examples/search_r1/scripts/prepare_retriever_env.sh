#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eu

RETRIEVER_ENV_ROOT="${SEARCH_R1_RETRIEVER_ENV_ROOT:-${TMPDIR:-/tmp}/search-r1-retriever}"
VENV_DIR="${RETRIEVER_ENV_ROOT}/.venv"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

mkdir -p "${RETRIEVER_ENV_ROOT}"
export UV_CACHE_DIR="${RETRIEVER_ENV_ROOT}/uv-cache"
export UV_ISOLATED=1
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv --python "$(command -v python3)" --system-site-packages "${VENV_DIR}"
fi

uv pip install --python "${VENV_DIR}/bin/python" --no-deps faiss-gpu==1.14.3

"${VENV_DIR}/bin/python" -c '
import faiss
import sys
import torch

faiss.StandardGpuResources()
print(f"python={sys.executable}")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}")
'
"${VENV_DIR}/bin/python" "${EXAMPLE_DIR}/retrieval_server.py" --help >/dev/null

echo "Search-R1 retriever environment is ready."
echo "Environment: ${VENV_DIR}"
echo "Server: ${EXAMPLE_DIR}/retrieval_server.py"
echo "Start with: bash examples/search_r1/run_retriever.sh"
