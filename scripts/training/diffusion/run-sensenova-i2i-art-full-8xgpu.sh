#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Independent ART recipe. Default is read-only preflight, never submission.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" -m scripts.training.diffusion.sensenova_art_recipe "$@"
