#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full-model OpenR1 GRPO with a frozen vision tower on 16 x 8 H800.
# By default, four 32-GPU rollout engines share the actor GPUs. Each engine spans four nodes.
# Set MODEL_DIR, DATA_DIR and SAVE_DIR on the Ray head, then use:
# bash scripts/entrypoint/ray-job.sh examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-openr1mm-128xh800.sh
set -eo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export EXP_NAME="${EXP_NAME:-deepseek-v41-1017-openr1mm-grpo-128xh800-tp${ROLLOUT_ENGINE_GPUS:-32}}"
OPENR1_CLUSTER_GPUS=128 exec bash "${script_dir}/run-deepseek-v4.1-flash-1017-openr1mm.sh" 40 "$@"
