#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -eu

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
# shellcheck source=/dev/null
[ -f "${EXAMPLE_DIR}/env.sh" ] && source "${EXAMPLE_DIR}/env.sh"

APP_ENV_ROOT="${DEEPEYES_V2_APP_ENV_ROOT:-/tmp/deepeyes-v2-app-env}"
VENV_DIR="${APP_ENV_ROOT}/.venv"

mkdir -p "${APP_ENV_ROOT}"
export UV_CACHE_DIR="${APP_ENV_ROOT}/uv-cache"
export UV_ISOLATED=1
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv --python "$(command -v python3)" --system-site-packages "${VENV_DIR}"
fi

uv pip install --python "${VENV_DIR}/bin/python" "jupyter_client>=8"
"${VENV_DIR}/bin/python" -c "import jupyter_client"

echo "DeepEyes V2 app environment is ready: ${VENV_DIR}"
