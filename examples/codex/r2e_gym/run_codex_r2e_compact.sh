#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
export CODEX_COMPACT=1
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_codex_r2e.sh" "$@"
