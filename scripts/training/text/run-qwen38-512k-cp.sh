#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${DATA_MODE:-mixed}" in
    mixed) export SCENARIO=mixed512 ;;
    filled) export SCENARIO=filled512 ;;
    *) echo 'DATA_MODE must be mixed or filled' >&2; exit 2 ;;
esac
exec bash "${SCRIPT_DIR}/run-qwen38-long-context.sh" "$@"
