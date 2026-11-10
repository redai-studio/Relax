#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# CE T17: Qwen3.5-9B DAPO-Math, full-parameter GRPO, FullyAsync elastic
# rollout.
#
# Resource layout:
#   stable:  actor 4 GPUs + two rollout engines x 1 GPU = 6 GPUs
#            The remaining 2 GPUs allow the first scale-out to stay on the
#            initial 8-GPU node, growing rollout from 2 to 4 engines.
#   elastic: additional 1-GPU engines can join on tidal workers, up to eight
#            rollout engines in total.

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")
echo "current time: ${now}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# QS owns the Ray cluster.  Use the external-Ray setup and never start a local
# head.  This RayJob is dedicated to one CE trial, so its stale-job cleanup is
# scoped to this trial's private Ray cluster.
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../entrypoint/ray-job.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen35-9B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/CE/run-qwen35-9B-elastic-rollout-async}"
EXP_DIR="${EXP_DIR:-/tmp/relax-ce}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=100}"

# The CE bootstrap installs this checkout only on the RayJob driver node.
# Upload it as the Ray job working directory so dynamically scheduled stable
# and elastic workers can import the same Relax commit before setup hooks run.
WORKING_DIR="${WORKING_DIR:-$(cd -- "${RELAX}" && pwd)}"

# Stable and elastic workers can lack a common RDMA fabric.  RDMA_DISABLE is
# retained as the task-level intent; NCCL/Gloo variables are the effective
# transport controls and must be propagated into every Ray worker.
export RDMA_DISABLE="${RDMA_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
# NCCL 2.28.9 can pass an AF_UNIX address into the Socket transport when this
# mixed CE topology inherits the platform default of four channels.  A live
# Stable/Elastic probe succeeds with one channel, so pin socket-only rollout
# communication here.  Keep the override recipe-scoped so other Relax tasks
# retain their existing defaults.
SOCKET_NCCL_CHANNELS="${RELAX_SOCKET_NCCL_CHANNELS:-1}"
export NCCL_MIN_NCHANNELS="${SOCKET_NCCL_CHANNELS}"
export NCCL_MAX_NCHANNELS="${SOCKET_NCCL_CHANNELS}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-${NCCL_SOCKET_IFNAME}}"

RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

runtime = json.loads(os.environ["RUNTIME_ENV_JSON"])
runtime.setdefault("env_vars", {}).update(
    {
        "RDMA_DISABLE": os.environ["RDMA_DISABLE"],
        "NCCL_IB_DISABLE": os.environ["NCCL_IB_DISABLE"],
        "NCCL_NET_PLUGIN": os.environ["NCCL_NET_PLUGIN"],
        "NCCL_SOCKET_FAMILY": os.environ["NCCL_SOCKET_FAMILY"],
        "NCCL_SOCKET_IFNAME": os.environ["NCCL_SOCKET_IFNAME"],
        "NCCL_MIN_NCHANNELS": os.environ["NCCL_MIN_NCHANNELS"],
        "NCCL_MAX_NCHANNELS": os.environ["NCCL_MAX_NCHANNELS"],
        "GLOO_SOCKET_IFNAME": os.environ["GLOO_SOCKET_IFNAME"],
        "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": os.environ[
            "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"
        ],
    }
)
print(json.dumps(runtime, separators=(",", ":")))
PY
)"
export RUNTIME_ENV_JSON

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}/Qwen3.5-9B"
   --ref-load "${MODEL_DIR}/Qwen3.5-9B"
   --megatron-to-hf-mode bridge
   --warm-hf-checkpoint-page-cache
)

PROMPT_SET="${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_SET}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rollout-seed 42
   --rm-type dapo
   --reward-key score
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --use-fault-tolerance
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 10240
   --no-rope-fusion
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --scale-out-timeout 1800
   --scale-out-partial-success-policy keep_partial
   --scale-in-drain-timeout 60
   --scale-weight-sync-precheck
)

WANDB_ARGS=(
   --use-clearml
   --use-metrics-service
   --tb-project-name "${PROJECT_NAME}"
   --tb-experiment-name "qwen35-9B-elastic-async-${now}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --seed 42
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Explicit opt-out for recovery-only runs on all-tidal Ray clusters.
if [[ "${CE_ENABLE_AFFINITY:-1}" == "0" ]]; then
    MISC_ARGS+=(--no-enable-affinity)
fi

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 4], "rollout": [1, 2], "advantages": [1, 0]}' \
   --max-staleness 2 \
   --num-data-storage-units 1 \
   --num-iters-per-train-update 32 \
   --ref-actor-config '{"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1, "expert_model_parallel_size": 1, "max_tokens_per_gpu": 10240, "sequence_parallel": false, "only_load_weight": true}' \
   --fully-async \
   --use-health-check \
   --autoscaler-config "${CE_AUTOSCALER_CONFIG:-${SCRIPT_DIR}/configs/qwen35-9b-elastic-autoscaler.yaml}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" 2>&1 | tee "log/qwen35-9B-elastic-async-${now}.log"
