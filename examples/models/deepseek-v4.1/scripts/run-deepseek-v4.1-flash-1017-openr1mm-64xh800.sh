#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full 40-layer 1017 BF16 language model and frozen vision tower on 8 x 8 H800.
# DATA_DIR contains the stratified OpenR1 train and eval split. Train for 200
# updates with 16 prompts x 8 samples. Evaluate the base and then every 10 updates.
# bash scripts/entrypoint/ray-job.sh examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-openr1mm-64xh800.sh
set -eo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OPENR1_CLUSTER_GPUS=64 exec bash "${script_dir}/run-deepseek-v4.1-flash-1017-openr1mm.sh" 40 "$@"
