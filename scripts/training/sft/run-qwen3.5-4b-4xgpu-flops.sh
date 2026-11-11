#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.5-4B FLOPs scheduling benchmark on 4 GPUs.
# Set MODEL_DIR, DATA_DIR, SAVE_DIR, RAY_ADDRESS and RAY_JOB_ADDRESS, then:
# RELAX_SKIP_CLUSTER_CLEANUP=1 RAY_NO_WAIT=1 bash scripts/entrypoint/ray-job.sh "$0"
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    source "${SCRIPT_DIR}/../../entrypoint/ray-job.sh"
fi
source "${RELAX_ROOT}/scripts/models/qwen35-4B.sh"
: "${MODEL_DIR:?Set MODEL_DIR}"
: "${DATA_DIR:?Set DATA_DIR}"
: "${SAVE_DIR:?Set SAVE_DIR}"
: "${RAY_JOB_ADDRESS:?Set the verified Ray Jobs HTTP address}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export CLEARML_OFFLINE_MODE="${CLEARML_OFFLINE_MODE:-1}"
MODE="${FLOPS_BENCH_MODE:-perf}"
ARM="${FLOPS_BENCH_ARM:-baseline}"
LAYOUT="${FLOPS_BENCH_LAYOUT:-default}"
PROJECT_NAME="${PROJECT_NAME:-Relax/sft/flops-benchmark}"
EXP_NAME="${EXP_NAME:-qwen3.5-4b-flops-${LAYOUT}-${MODE}-${ARM}}"
WORKLOAD="${FLOPS_BENCH_WORKLOAD:-Qwen3.5-4B/stratified-gbs16-cap24576}"
export RELAX_FLOPS_MANIFEST="${DATA_DIR}/${WORKLOAD}/manifest.json"
mapfile -t FROZEN < <(python3 - "${RELAX_FLOPS_MANIFEST}" <<'PYMETA'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
for key in ('batch_size', 'capacity', 'seed'):
    if type(m[key]) is not int or m[key] <= 0:
        raise ValueError(f'Invalid frozen {key}')
    print(m[key])
PYMETA
)
[[ "${#FROZEN[@]}" == 3 ]]
BUDGET="${FLOPS_BENCH_BUDGET:-${FROZEN[1]}}"
LOAD_NAME="${FLOPS_BENCH_LOAD_NAME:-qwen3.5-4b-flops-warm-native}"
SAVE_ARGS=()
HOOK_ARGS=()
PROFILE_ARGS=()
case "$MODE" in
    warm)
        END_STEP="${FLOPS_BENCH_END_STEP:-20}"
        LOAD_NAME="$EXP_NAME"
        SAVE_ARGS=(
            --save-interval 20
            --max-actor-ckpt-to-keep 1
        )
        ;;
    capture)
        END_STEP="${FLOPS_BENCH_END_STEP:-21}"
        export RELAX_FLOPS_CAPTURE_ROLLOUT="${RELAX_FLOPS_CAPTURE_ROLLOUT:-20}"
        export RELAX_FLOPS_CAPTURE_DIR="${SAVE_DIR}/${EXP_NAME}/capture"
        export RELAX_FLOPS_INITIAL_STATE_ID
        RELAX_FLOPS_INITIAL_STATE_ID="$(python3 - "${SAVE_DIR}/${LOAD_NAME}/fingerprint.json" <<'PYID'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())['initial_state_id'])
PYID
)"
        HOOK_ARGS=(
            --custom-megatron-before-train-step-hook-path scripts.tools.flops_benchmark.capture.before_train_step
        )
        ;;
    audit)
        END_STEP="${FLOPS_BENCH_END_STEP:-23}"
        export RELAX_FLOPS_AUDIT_DIR="${SAVE_DIR}/${EXP_NAME}/audit"
        HOOK_ARGS=(
            --custom-megatron-before-train-step-hook-path scripts.tools.flops_benchmark.audit.before_train_step
        )
        PROFILE_ARGS=(
            --use-pytorch-profiler
            --profile-step-start 0
            --profile-step-end 2
            --profile-output-dir "${SAVE_DIR}/${EXP_NAME}/profiles"
        )
        ;;
    perf)
        END_STEP="${FLOPS_BENCH_END_STEP:-50}"
        ;;
    *) echo "Unknown FLOPS_BENCH_MODE: $MODE" >&2; exit 2 ;;
esac
BALANCE_ARGS=()
case "$ARM" in
    baseline) ;;
    flops) BALANCE_ARGS=(--balance-by-flops) ;;
    *) echo "Unknown FLOPS_BENCH_ARM: $ARM" >&2; exit 2 ;;
esac
TP=1
PP=1
CP=1
LAYOUT_ARGS=()
case "$LAYOUT" in
    default) ;;
    tp)
        TP=2
        LAYOUT_ARGS=(--sequence-parallel)
        ;;
    pp) PP=2 ;;
    vpp)
        PP=2
        LAYOUT_ARGS=(
            --num-virtual-stages-per-pipeline-rank 2
            --overlap-p2p-comm
        )
        ;;
    cp)
        CP=2
        LAYOUT_ARGS=(--linear-cp-mode headwise)
        ;;
    *) echo "Unknown dense layout: $LAYOUT" >&2; exit 2 ;;
esac
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3.5-4B"
    --ref-load "${MODEL_DIR}/Qwen3.5-4B"
    --megatron-to-hf-mode bridge
    --load "${SAVE_DIR}/${LOAD_NAME}"
    --save "${SAVE_DIR}/${EXP_NAME}"
    --use-checkpoint-opt-param-scheduler
    --num-rollout "$END_STEP"
)
SFT_ARGS=(
    --loss-type sft
    --prompt-data "${DATA_DIR}/${WORKLOAD}/samples.jsonl"
    --input-key problem
    --label-key generated_solution
    --global-batch-size "${FROZEN[0]}"
    --seed "${FROZEN[2]}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "$BUDGET"
    --balance-data
    --sft-oversize-strategy keep
    --sft-prefetch-buffer-size 256
    --sft-prefetch-num-workers 4
)
PERF_ARGS=(
    --bf16
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --context-parallel-size "$CP"
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --calculate-per-token-loss
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --no-rope-fusion
    --cross-entropy-loss-fusion
    --sft-chunked-logits
    --sft-logits-chunk-size 1024
    --attention-dropout 0
    --hidden-dropout 0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --min-lr 1e-5
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --use-distributed-optimizer
    --optimizer-cpu-offload
    --use-precision-aware-optimizer
)
MONITOR_ARGS=(
    --use-clearml
    --use-metrics-service
    --tb-project-name "$PROJECT_NAME"
    --tb-experiment-name "${EXP_NAME}-$(date +%Y%m%d-%H%M%S)"
)
# Worker processes need the same capture paths and offline tracking setting.
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON="$(python3 - <<'PYENV'
import json, os
runtime = json.loads(os.environ.get('RUNTIME_ENV_JSON', '{}'))
env = runtime.setdefault('env_vars', {})
env.update({k: v for k, v in os.environ.items() if k.startswith('RELAX_FLOPS_')})
env['CLEARML_OFFLINE_MODE'] = os.environ['CLEARML_OFFLINE_MODE']
print(json.dumps(runtime))
PYENV
)"
now="$(date +%Y%m%d-%H%M%S)"
mkdir -p "${SCRIPT_DIR}/log"
ray job submit ${RAY_NO_WAIT:+--no-wait} \
    --address "$RAY_JOB_ADDRESS" \
    --submission-id "${FLOPS_BENCH_JOB_ID:-${EXP_NAME}-${now}}" \
    ${WORKING_DIR:+--working-dir "$WORKING_DIR"} \
    --runtime-env-json "$RUNTIME_ENV_JSON" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"sft": [1, 0], "actor": [1, 4]}' \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    "${SFT_ARGS[@]}" \
    "${BALANCE_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${LAYOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${HOOK_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${MONITOR_ARGS[@]}" \
    "$@" 2>&1 | tee "${SCRIPT_DIR}/log/${EXP_NAME}-${now}.log"
