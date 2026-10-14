#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Keep the documented example command pointed at the maintained training script.
set -euo pipefail
RELAX_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "${RELAX_ROOT}/examples/osworld_agentic/scripts/run-qwen3vl-8B-osworld-8xgpu.sh" "$@"
