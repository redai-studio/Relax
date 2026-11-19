#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
# Qwen3.6-35B-A3B multimodal SFT on 4 x 8 Blackwell GPUs with Megatron Muon.
# Every training node needs a Blackwell image with FLA 0.4.2 and emerging-optimizers
# compatible with the image's Megatron. Ray training workers must be able to import both.
# MODEL_DIR=/models/Qwen3.6-35B-A3B DATA_DIR=/data/parquet SAVE_DIR=/checkpoints \
#   bash scripts/training/sft/run-qwen36-35B-A3B-32xgpu-muon.sh
# Set DRY_RUN=1 to inspect the Ray job. Reuse EXP_NAME to resume the same run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RELAX="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the Qwen3.6-35B-A3B HF checkpoint directory}"
: "${DATA_DIR:?Set DATA_DIR to the multimodal SFT dataset}"
: "${SAVE_DIR:?Set SAVE_DIR to a persistent shared checkpoint root}"
export RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8265}"
export RAY_NO_WAIT="${RAY_NO_WAIT-}"
export RELAX_SKIP_CLUSTER_CLEANUP="${RELAX_SKIP_CLUSTER_CLEANUP:-1}"
if [[ "${DRY_RUN:-0}" != 1 && -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "$RELAX/scripts/entrypoint/ray-job.sh" "${BASH_SOURCE[0]}" "$@"
fi
source "$RELAX/scripts/models/qwen36-35B-A3B.sh"

PROJECT_NAME="${PROJECT_NAME:-Relax/sft/qwen36}"
EXP_NAME="${EXP_NAME:-qwen36-35B-A3B-muon-native-32gpu}"
RUN_NAME="$EXP_NAME-$(date +%Y%m%d-%H%M%S)-$$"
CHECKPOINT_DIR="$SAVE_DIR/$EXP_NAME"

# Select the image's Blackwell FLA build before Python imports Megatron's native GDN.
# The standard Ray entrypoint supplies the other paths and the worker setup hook.
FLA_ROOT="${RELAX_LEGACY_FLA_ROOT:-/opt/relax/compat/fla-0.4.2}"
if [[ "${DRY_RUN:-0}" != 1 && ! -f "$FLA_ROOT/fla/ops/gated_delta_rule/chunk.py" ]]; then
    echo "Missing Blackwell FLA runtime at $FLA_ROOT. Set RELAX_LEGACY_FLA_ROOT if installed elsewhere." >&2
    exit 1
fi
RUNTIME_ENV_JSON="$(python3 - "$FLA_ROOT" <<'PY'
import json
import os
import sys

runtime = json.loads(os.environ.get("RUNTIME_ENV_JSON") or "{}")
env = runtime.setdefault("env_vars", {})
paths = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", "")).split(os.pathsep)
env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([sys.argv[1], *filter(None, paths)]))
# Each preprocessing process handles one sample. Keep nested BLAS pools single-threaded.
workers = dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"), "1")
workers["TOKENIZERS_PARALLELISM"] = "false"
workers["RELAX_PROCESSOR_POOL_SIZE"] = os.environ.get("RELAX_PROCESSOR_POOL_SIZE", "96")
env.update(workers)
forward = env.get("RELAX_PROPAGATE_ENV_VARS", os.environ.get("RELAX_PROPAGATE_ENV_VARS", "")).split(",")
env["RELAX_PROPAGATE_ENV_VARS"] = ",".join(dict.fromkeys([*filter(None, forward), *workers]))
print(json.dumps(runtime))
PY
)"

CKPT_ARGS=(
    --hf-checkpoint "$MODEL_DIR"
    --ref-load "$MODEL_DIR"
    --megatron-to-hf-mode bridge
    --load "$CHECKPOINT_DIR"
    --save "$CHECKPOINT_DIR"
    --save-interval "${SAVE_INTERVAL:-50}"
    --max-actor-ckpt-to-keep 5
)

SFT_ARGS=(
    --loss-type sft
    --num-rollout "${TRAIN_STEPS:-50}"
    --prompt-data "$DATA_DIR"
    --input-key messages
    --metadata-key meta
    --tool-key tools
    --multimodal-keys '{"image":"images"}'
    --image-min-token-num 64
    --image-max-token-num 16384
    --apply-chat-template-kwargs '{"enable_thinking":false,"preserve_thinking":true,"add_non_thinking_prefix":true}'
    --sft-ignore-empty-think
    --freeze-vision-model
    --freeze-vision-projection
    --enable-mtp-training
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
    --mtp-detach-paths none
    --global-batch-size 800
    --seq-length 262144
    --max-tokens-per-gpu 65536
    --sft-oversize-strategy skip
    --use-dynamic-batch-size
    --balance-data
    --sft-packing-strategy binpack
    --sft-max-in-flight-steps 3
    --sft-prefetch-buffer-size 2048
    --sft-prefetch-num-workers "${SFT_PREFETCH_NUM_WORKERS:-96}"
    --sft-prefetch-chunk-size 128
    --sft-tq-timeout-minutes 60
    --eval-size 0.01
    --eval-interval 0
)

# 32 / (TP1 x PP2 x CP4) = 4 data replicas. Use the model's Megatron all-to-all dispatcher.
PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --context-parallel-size 4
    --linear-cp-mode headwise
    --expert-model-parallel-size 4
    --expert-tensor-parallel-size 1
    --decoder-first-pipeline-num-layers 20
    --decoder-last-pipeline-num-layers 20
    --data-pad-size-multiplier 8192
    --recompute-granularity full
    --recompute-method block
    --recompute-num-layers 20
    --recompute-loss-function
    --sft-chunked-logits
    --sft-logits-chunk-size 1024
    --log-probs-chunk-size 2048
    --calculate-per-token-loss
    --cross-entropy-loss-fusion
    --no-rope-fusion
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-4
    --overlap-grad-reduce
    --overlap-param-gather
)

OPTIMIZER_ARGS=(
    --bf16
    --optimizer muon
    --use-distributed-optimizer
    --no-use-precision-aware-optimizer
    --no-store-param-remainders
    --muon-momentum 0.95
    --muon-nesterov
    --muon-coefficient-type polar_express
    --muon-num-ns-steps 8
    --muon-scale-mode spectral
    --muon-extra-scale-factor 0.2
    --lr 2e-5
    --min-lr 1e-6
    --lr-decay-style cosine
    --lr-warmup-fraction 0.05
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --seed 42
)

TRAIN_ARGS=(
    python3 -m relax.entrypoints.train
    --resource '{"sft":[1,0],"actor":[1,32]}'
    --num-data-storage-units 8
    --train-env-vars '{"PYTORCH_ALLOC_CONF":"expandable_segments:True"}'
    --use-health-check
    --use-clearml
    --use-metrics-service
    --tb-project-name "$PROJECT_NAME"
    --tb-experiment-name "$RUN_NAME"
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${SFT_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
)
# Older launch commands include this script-level flag. Multimodal SFT is the default.
for arg in "$@"; do
    if [[ "$arg" != --multimodal ]]; then TRAIN_ARGS+=("$arg"); fi
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    python3 - "$RUNTIME_ENV_JSON" "${TRAIN_ARGS[@]}" <<'PY'
import json
import sys

print(json.dumps({"runtime_env": json.loads(sys.argv[1]), "argv": sys.argv[2:]}, indent=2))
PY
    exit 0
fi

RAY_ARGS=(
    job submit
    --address="${RAY_JOBS_ADDRESS:-$RAY_ADDRESS}"
    --submission-id="${SUBMISSION_ID:-$RUN_NAME}"
    --runtime-env-json="$RUNTIME_ENV_JSON"
)
if [[ -n "$RAY_NO_WAIT" ]]; then RAY_ARGS+=(--no-wait); fi
if [[ -n "${WORKING_DIR:-}" ]]; then RAY_ARGS+=(--working-dir "$WORKING_DIR"); fi
mkdir -p "$SCRIPT_DIR/log"
ray "${RAY_ARGS[@]}" -- "${TRAIN_ARGS[@]}" 2>&1 | tee "$SCRIPT_DIR/log/$RUN_NAME.log"
