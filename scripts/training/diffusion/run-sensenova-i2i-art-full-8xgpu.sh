#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Independent ART recipe. Default is read-only preflight, never submission.
# Optional B1 generation/return overlap on 8 engines with N_SAMPLES=8:
#   ART_GROUP_OVERLAP=1 ART_OVERLAP_PENDING_GROUPS=3 ART_GENERATION_PREFETCH=1
# Defaults retain pending=2 and prefetch=0. The SGLang v0.5.21 patch automatically
# bitpacks eligible ART masks; rebuild the image and restart old engines to update it.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}/../../.."
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" -m scripts.training.diffusion.sensenova_art_recipe "$@"
