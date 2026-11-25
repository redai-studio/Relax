#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Full-model OpenR1 GRPO with a frozen vision tower on 16 x 8 H800.
# Four BF16 rollout engines each span four nodes (attention DP4 x TP8, MoE EP32).
# To reproduce the 16-GPU routed-FP8 recipe, set ROLLOUT_ENGINE_GPUS=16
# and RELAX_DSV41_ROUTED_FP8=1. Set OPENR1_POLICY=sc-tis for Score Centering with TIS.
# Set MODEL_DIR, DATA_DIR and SAVE_DIR on the Ray head, then use:
# bash scripts/entrypoint/ray-job.sh examples/models/deepseek-v4.1/scripts/run-deepseek-v4.1-flash-1017-openr1mm-128xh800.sh
set -eo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export ROLLOUT_ENGINE_GPUS="${ROLLOUT_ENGINE_GPUS:-32}"
export RELAX_DSV41_ROUTED_FP8="${RELAX_DSV41_ROUTED_FP8:-0}"
export OPENR1_POLICY="${OPENR1_POLICY:-grpo}"
precision=bf16
if [[ ${RELAX_DSV41_ROUTED_FP8} == 1 ]]; then precision=routed-fp8; fi
export EXP_NAME="${EXP_NAME:-deepseek-v41-1017-openr1mm-128xh800-${precision}-ep${ROLLOUT_ENGINE_GPUS}-${OPENR1_POLICY}-gbs512}"
OPENR1_CLUSTER_GPUS=128 exec bash "${script_dir}/run-deepseek-v4.1-flash-1017-openr1mm.sh" 40 "$@"
