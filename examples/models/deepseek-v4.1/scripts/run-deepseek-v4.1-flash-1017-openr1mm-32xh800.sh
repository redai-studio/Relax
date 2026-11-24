#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Four decoder layers, frozen vision tower, colocated OpenR1 GRPO on 4 x 8 H800.
set -eo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/run-deepseek-v4.1-flash-1017-openr1mm.sh" 4 "$@"
