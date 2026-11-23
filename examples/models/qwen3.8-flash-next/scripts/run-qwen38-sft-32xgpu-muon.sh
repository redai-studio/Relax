#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.8 32-GPU Muon SFT. The Blackwell image needs Emerging Optimizers on every worker.
set -eo pipefail

qwen38_launcher="${BASH_SOURCE[0]}"
qwen38_driver_connections=1
qwen38_precision_flag=--no-use-precision-aware-optimizer
qwen38_optimizer_args=(
    --optimizer muon
    --muon-momentum 0.95
    --muon-nesterov
    --muon-coefficient-type polar_express
    --muon-num-ns-steps 8
    --muon-scale-mode spectral
    --muon-extra-scale-factor 0.2
    --muon-tp-mode duplicated
)
qwen38_extra_env=()
if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
    qwen38_extra_env+=("GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME")
fi
export EXP_NAME="${EXP_NAME:-qwen38-qsa-32xb300-muon}" # gitleaks:allow -- stable checkpoint identity
export QWEN38_CP_SIZE=1
source "$(dirname -- "${BASH_SOURCE[0]}")/run-qwen38-qsa-sft-32xgpu-blackwell.sh" "$@"
